"""SUPERSEDED 2026-11-04 by colab_queue.py (per-period sync, resume, VM-identity guard). Kept as the record of the first, lost attempt.

Run one queued campaign recipe on a Colab CLI GPU VM and bring it home for the normal publisher.

The VM mirrors the local absolute paths (input, raw input, runs/<experiment>), so the
runner, its fit reports and the publisher's path checks are byte-for-byte the local ones.
Only the interpreter differs: /content/colab_run/.venv (Python 3.12, versions pinned to the
local .venv in requirements.txt).

  start  build experiment.py/run.sh exactly like launch.launch, copy them, start a VM tmux job
  watch  poll every --interval s: mirror status/log, keep the kernel active, and when
         train.exit appears collect the run, validate it with publish.validate_run in a staging
         root, install it into runs/<experiment> (the local controller then publishes it), and
         stop the VM

It never edits model code, features, seed or rounds.
"""
from __future__ import annotations

import argparse
import ast
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VM_PYTHON = "/content/colab_run/.venv/bin/python"
MONITOR = REPO / "tools/sunrun_monitor.py"


def now():
    return datetime.now(UTC).isoformat()


def emit(event, **fields):
    print(json.dumps({"utc": now(), "event": event, **fields}), flush=True)


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class VM:
    def __init__(self, work, session):
        self.work, self.session = work, session
        self.config = work / "ssh_config"
        self.host = "colab-" + session

    def ssh(self, command, input=None, timeout=300, check=True):
        result = subprocess.run(["ssh", "-F", str(self.config), self.host, command], input=input,
                                capture_output=True, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(f"ssh exit {result.returncode}: {result.stderr.decode(errors='replace')[-300:]}")
        return result

    def put(self, local, remote, timeout=1800):
        subprocess.run(["scp", "-F", str(self.config), "-p", str(local), f"{self.host}:{remote}"],
                       check=True, capture_output=True, timeout=timeout)

    def get(self, remote, local, timeout=3600):
        subprocess.run(["scp", "-F", str(self.config), "-p", f"{self.host}:{remote}", str(local)],
                       check=True, capture_output=True, timeout=timeout)

    def keepalive(self):
        subprocess.run(["colab", "exec", "-s", self.session, "--timeout", "60"], input=b"1\n",
                       capture_output=True, timeout=120)


def identity(path):
    stat = Path(path).stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def runner_source(family, experiment):
    """Identical transformation to launch.launch."""
    source = (HERE / f"{family}.py").read_text()
    main_entry = ast.parse(source).body[-1]
    if not isinstance(main_entry, ast.If) or "__name__" not in ast.unparse(main_entry.test):
        raise ValueError("Runner must have a standalone main guard")
    return ("\n".join(source.splitlines()[:main_entry.lineno - 1]) + '\nif __name__ == "__main__":\n'
            '    import sys\n    if "--experiment" not in sys.argv:\n'
            f'        sys.argv.extend(["--experiment", {experiment!r}])\n    main()\n')


def start(args):
    work, experiment = Path(args.work).resolve(), args.experiment
    vm = VM(work, args.session)
    registry = json.loads((HERE / "registry.json").read_text())
    if experiment in registry:
        raise ValueError("Remove the experiment from the local registry first (controller would launch it too)")
    manifest = json.loads((HERE / "manifest.json").read_text())
    entry = next(e for e in manifest["experiments"] if e["experiment"] == experiment)
    if entry.get("state") != "assigned_colab":
        raise ValueError("Manifest must mark the experiment assigned_colab")
    family = args.family
    out = HERE / "runs" / experiment
    if out.exists():
        raise FileExistsError(f"Local run directory already exists: {out}")
    data, raw = HERE / "input/df_ffill_trainy.parquet", REPO / "data-v2/input/df.parquet"
    source = runner_source(family, experiment)
    command = [VM_PYTHON, str(out / "experiment.py"), "--data", str(data), "--out", str(out),
               "--experiment", experiment, "--threads", str(args.threads)]
    if family in ("strategies", "foundation", "direct", "tcn", "special", "special_hsspeak"):
        command += ["--raw-data", str(raw)]
    command += ["--device", "cuda:0"]
    script = ("#!/bin/bash\nset -o pipefail\ncd /content/colab_run\n"
              + shlex.join(["env", "CUDA_VISIBLE_DEVICES=0", *command]) + " > "
              + shlex.quote(str(out / "train.log")) + " 2>&1\ncode=$?\nprintf '%s\\n' $code > "
              + shlex.quote(str(out / "train.exit")) + "\nexit $code\n")
    staged = work / "launch"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "experiment.py").write_text(source)
    (staged / "run.sh").write_text(script)
    versions = vm.ssh(VM_PYTHON + " -c " + shlex.quote(
        "import json,sys,importlib.metadata as m,torch;print(json.dumps(dict(python=sys.version.split()[0],"
        "gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,**{p:m.version(p) for p in "
        "('numpy','pandas','torch','pytabkit','structured-data-models','pyarrow','huggingface_hub')})))")).stdout
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    record = {"experiment": experiment, "family": family, "host": "colab", "tmux_session": f"colab:{args.session}:train",
              "started_utc": now(), "command": command, "output": str(out),
              "baseline_submission_utc": entry["original_submission_utc"], "threads": args.threads,
              "input_identity": identity(data), "raw_input_identity": identity(raw), "training_target_ffill": True,
              "source_snapshot": str(out / "experiment.py"), "source_bytes": len(source.encode()),
              "gpu": 0, "gpu_lease": None, "training_host": "colab",
              "colab": {"session": args.session, "vm_environment": json.loads(versions),
                        "requirements": str(work / "requirements.txt"),
                        "input_copy_check": str(work / "input-check-colab.json"),
                        "path_layout": "VM mirrors local absolute input/run paths; no metadata rebasing"},
              "source_git_commit": git}
    (staged / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    vm.ssh(f"mkdir -p {shlex.quote(str(out))} && test -z \"$(ls -A {shlex.quote(str(out))})\"")
    vm.put(staged / "experiment.py", out / "experiment.py")
    vm.put(staged / "experiment.py", out / "runner_at_launch.py")
    vm.put(staged / "run.sh", out / "run.sh")
    vm.put(staged / "launch.json", out / "launch.json")
    q = shlex.quote
    vm.ssh(f"chmod 444 {q(str(out / 'experiment.py'))} {q(str(out / 'runner_at_launch.py'))} && "
           f"cmp {q(str(out / 'experiment.py'))} {q(str(out / 'runner_at_launch.py'))} && "
           f"tmux new-session -d -s train bash {q(str(out / 'run.sh'))}")
    atomic(work / "status.json", {"experiment": experiment, "host": "colab", "session": args.session,
                                  "state": "running", "started_utc": record["started_utc"], "updated": time.time()})
    emit("started", experiment=experiment, session=args.session, command=command)


def snapshot(vm):
    result = vm.ssh("python3 - --collect colab", input=MONITOR.read_bytes(), timeout=120)
    text = result.stdout.decode().strip()
    return json.loads(text[text.index("{"):])


def collect(vm, work, experiment):
    out = HERE / "runs" / experiment
    recovery = work / "collected"
    recovery.mkdir(exist_ok=True)
    archive = recovery / f"{experiment}.tar.gz"
    q = shlex.quote
    vm.ssh(f"tar -C {q(str(out.parent))} -czf /content/colab_run/{experiment}.tar.gz {q(experiment)}", timeout=1800)
    vm.get(f"/content/colab_run/{experiment}.tar.gz", archive)
    original = recovery / experiment
    if original.exists():
        shutil.rmtree(original)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != experiment or ".." in parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unexpected archive member")
        stream.extractall(recovery, filter="data")
    return original


def install(work, experiment, original):
    """Validate a copy under a staging HERE, then expose it as runs/<experiment>."""
    sys.path.insert(0, str(HERE))
    import publish
    out = HERE / "runs" / experiment
    if out.exists():
        raise FileExistsError("Local run directory appeared; refusing overwrite")
    record = json.loads((work / "launch" / "launch.json").read_text())
    if record["input_identity"] != identity(HERE / "input/df_ffill_trainy.parquet"):
        raise ValueError("Local prepared input changed since launch")
    if (original / "experiment.py").read_bytes() != (original / "runner_at_launch.py").read_bytes():
        raise ValueError("Collected runner differs from its launch snapshot")
    record.update(finished_collected_utc=now(), artifact_location="local")
    with tempfile.TemporaryDirectory(prefix=experiment + "-", dir=work) as temporary:
        root = Path(temporary)
        (root / "runs").mkdir()
        staged = root / "runs" / experiment
        shutil.copytree(original, staged)
        (staged / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        (root / "input").symlink_to(HERE / "input", target_is_directory=True)
        old = publish.HERE
        try:
            publish.HERE = root
            publish.validate_run(experiment)
        finally:
            publish.HERE = old
        staged.rename(out)
    return out


def watch(args):
    work, experiment = Path(args.work).resolve(), args.experiment
    vm = VM(work, args.session)
    status_path = work / "status.json"
    failures = 0
    while True:
        status = json.loads(status_path.read_text())
        try:
            snap = snapshot(vm)
            done = vm.ssh(f"cat {shlex.quote(str(HERE / 'runs' / experiment / 'train.exit'))} 2>/dev/null; "
                          "tmux has-session -t train 2>/dev/null && echo ALIVE", check=False).stdout.decode()
            failures = 0
        except Exception as error:  # VM gone or tunnel busy
            failures += 1
            status.update(updated=time.time(), error=str(error)[:200], consecutive_failures=failures)
            if failures >= 10:
                status["state"] = "lost"
                atomic(status_path, status)
                emit("vm_lost", error=str(error)[:200])
                raise SystemExit(3)
            atomic(status_path, status)
            time.sleep(args.interval)
            continue
        lines = done.split()
        code = lines[0] if lines and lines[0] != "ALIVE" else None
        status.update(updated=time.time(), snapshot=snap, error=None, consecutive_failures=0,
                      vm_tmux_alive="ALIVE" in lines)
        if code is None and "ALIVE" not in lines:
            status["state"] = "lost"
            atomic(status_path, status)
            emit("training_session_lost")
            raise SystemExit(4)
        if code is None:
            atomic(status_path, status)
            vm.keepalive()
            time.sleep(args.interval)
            continue
        status.update(state="collecting", exit_code=code)
        atomic(status_path, status)
        original = collect(vm, work, experiment)
        emit("collected", experiment=experiment, exit_code=code, original=str(original))
        if code != "0":
            status.update(state="failed", collected=str(original))
            atomic(status_path, status)
            subprocess.run(["colab", "stop", "-s", args.session], capture_output=True, timeout=300)
            emit("training_failed", exit_code=code)
            raise SystemExit(5)
        out = install(work, experiment, original)
        emit("installed", experiment=experiment, out=str(out))
        subprocess.run(["colab", "stop", "-s", args.session], capture_output=True, timeout=300)
        status.update(state="installed", installed=str(out), vm_stopped_utc=now())
        atomic(status_path, status)
        emit("vm_stopped", session=args.session)
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["start", "watch"])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--family", default="foundation")
    parser.add_argument("--session", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interval", type=float, default=120)
    args = parser.parse_args()
    (start if args.action == "start" else watch)(args)


if __name__ == "__main__":
    main()
