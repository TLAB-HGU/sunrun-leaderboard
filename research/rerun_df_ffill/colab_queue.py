"""Colab CLI GPU queue for campaign recipes, with per-period sync and resume on VM loss.

  add EXPERIMENT   move a not-yet-launched recipe from the local registry into colab-queue.json
  run              driver loop (one job at a time, stays alive for future queue items)
  status           print queue state

Per job and attempt the driver
  1. creates a fresh L4 session (name cq<attempt>-<experiment>), records its endpoint, and before
     every SSH call checks that `colab sessions` still lists that exact endpoint with the requested
     GPU. `colab ssh --proxy-mode` silently creates a CPU runtime when the session is gone, so a
     missing/changed endpoint is treated as VM loss; host keys are pinned per attempt as well;
  2. builds a Python 3.12 venv pinned to colab_runs/requirements.txt (the local .venv versions),
     prefetches the runner's pinned checkpoint, copies the inputs to the same absolute paths as
     locally and checks bytes/rows/columns/null counts/numeric sums against the local files;
  3. starts the runner exactly as launch.launch would (attempt 1: one full invocation). Each
     finished period (F*_metadata.json written last) is copied home immediately. If the VM is lost,
     the next attempt uploads the finished periods and runs only the missing ones with --period,
     then --merge. Periods are independent fits (seed 42 each), so the outputs are the same recipe;
  4. after train.exit: collects the run, validates it with publish.validate_run in a staging root,
     installs runs/<experiment> (the local controller publishes it) and stops the session.
Model code, features, seeds and rounds are never modified.
"""
from __future__ import annotations

import argparse
import ast
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QUEUE = HERE / "colab-queue.json"
RUNS = HERE / "colab_runs"
REQUIREMENTS = RUNS / "requirements.txt"
MONITOR = REPO / "tools/sunrun_monitor.py"
VM_ROOT = "/content/colab_run"
VM_PYTHON = VM_ROOT + "/.venv/bin/python"
DATA = HERE / "input/df_ffill_trainy.parquet"
RAW = REPO / "data-v2/input/df.parquet"
PERIODS = ("F1", "F2", "F3")
RAW_FAMILIES = ("strategies", "foundation", "direct", "tcn", "special", "special_hsspeak")
MAX_ATTEMPTS = 4
POLL_SECONDS = 60
LOST_AFTER_FAILURES = 5
MIN_BALANCE_CU = 3.0
SESSION_LINE = re.compile(r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)\s+\|\s+Hardware:\s+(?P<hardware>\S+)", re.M)

PQ_CHECK = r'''
import json, os, sys
import pyarrow.parquet as pq, pyarrow.compute as pc
out = {}
for p in sys.argv[1:]:
    t = pq.read_table(p)
    sums = {}
    for c in t.column_names:
        if str(t[c].type).startswith(("int", "uint", "float", "double")):
            s = pc.sum(t[c]).as_py()
            sums[c] = None if s is None else float(s)
    out[p] = {"bytes": os.path.getsize(p), "rows": t.num_rows, "names": t.column_names,
              "nulls": {c: t[c].null_count for c in t.column_names}, "sums": sums}
print(json.dumps(out))
'''

PREFETCH = r'''
import json, sys
import huggingface_hub
original = huggingface_hub.hf_hub_download
def download(*args, **kwargs):
    kwargs.pop("local_files_only", None)
    return original(*args, **kwargs)
huggingface_hub.hf_hub_download = download
sys.path.insert(0, sys.argv[1])
import experiment
if not hasattr(experiment, "checkpoint"):
    print(json.dumps(None)); raise SystemExit
family = experiment.EXPERIMENTS[sys.argv[2]][0]
identity = experiment.checkpoint(family)
if identity:
    import os
    identity["bytes"] = os.path.getsize(identity["local_path"])
print(json.dumps(identity))
'''


def now():
    return datetime.now(UTC).isoformat()


def log(event, **fields):
    print(json.dumps({"utc": now(), "event": event, **fields}), flush=True)


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def identity(path):
    stat = Path(path).stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class QueueFile:
    """colab-queue.json, edited only under an exclusive lock."""

    def __enter__(self):
        self.lock = (HERE / ".colab-queue.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.data = read_json(QUEUE, {"items": []})
        return self

    def save(self):
        atomic(QUEUE, self.data)

    def __exit__(self, *exc):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()

    def item(self, experiment):
        return next(i for i in self.data["items"] if i["experiment"] == experiment)


def update_item(experiment, **fields):
    with QueueFile() as q:
        q.item(experiment).update(fields, updated_utc=now())
        q.save()


# ------------------------------------------------------------------ colab / ssh primitives
def colab(*args, timeout=300, input=None):
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout, input=input)


def sessions():
    result = colab("sessions", timeout=90)
    return {m["name"]: (m["endpoint"], m["hardware"]) for m in SESSION_LINE.finditer(result.stdout)}


def balance():
    match = re.search(r"Current balance:\s*([0-9.]+)", colab("usage", timeout=90).stdout)
    return float(match.group(1)) if match else None


class Lost(Exception):
    """The attempt's VM is gone or replaced; resume on a new VM."""


class VM:
    def __init__(self, work, name, gpu):
        self.work, self.name, self.gpu = work, name, gpu
        self.endpoint = None
        self.config = work / f"ssh_config.{name}"
        self.host = "colab-" + name

    def create(self):
        result = colab("new", "-s", self.name, "--gpu", self.gpu, timeout=900)
        if "READY" not in result.stdout + result.stderr:
            raise Lost("session create failed: " + (result.stdout + result.stderr)[-200:])
        listed = sessions().get(self.name)
        if not listed or listed[1] != self.gpu:
            raise Lost(f"created session is not {self.gpu}: {listed}")
        self.endpoint = listed[0]
        self.config.write_text(
            f"Host {self.host}\n  User root\n  ProxyCommand colab ssh --proxy-mode -s {self.name}\n"
            f"  StrictHostKeyChecking accept-new\n  UserKnownHostsFile {self.work}/known_hosts.{self.name}\n"
            "  BatchMode yes\n  ConnectTimeout 60\n  ServerAliveInterval 30\n  ServerAliveCountMax 4\n"
            f"  ControlMaster auto\n  ControlPath ~/.ssh/cm-{self.name}\n  ControlPersist 900\n")
        return self.endpoint

    def guard(self):
        try:
            listed = sessions().get(self.name)
        except subprocess.TimeoutExpired:
            return  # listing is flaky; ssh host-key pinning still protects against a replaced VM
        if listed is None:
            raise Lost("session no longer listed")
        if listed != (self.endpoint, self.gpu):
            self.stop()  # an auto-created replacement under our name; never run anything on it
            raise Lost(f"session replaced by {listed}")

    def ssh(self, command, input=None, timeout=600, guard=True):
        if guard:
            self.guard()
        result = subprocess.run(["ssh", "-F", str(self.config), self.host, command], input=input,
                                capture_output=True, timeout=timeout)
        if result.returncode == 255:
            raise ConnectionError("ssh: " + result.stderr.decode(errors="replace")[-200:])
        return result

    def check(self, command, **kwargs):
        result = self.ssh(command, **kwargs)
        if result.returncode:
            raise RuntimeError(f"remote exit {result.returncode}: "
                               + (result.stderr or result.stdout).decode(errors="replace")[-400:])
        return result.stdout.decode()

    def put(self, local, remote, timeout=1800):
        self.guard()
        subprocess.run(["scp", "-F", str(self.config), "-p", str(local), f"{self.host}:{remote}"],
                       check=True, capture_output=True, timeout=timeout)

    def get(self, remote, local, timeout=1800):
        self.guard()
        subprocess.run(["scp", "-F", str(self.config), "-p", f"{self.host}:{remote}", str(local)],
                       check=True, capture_output=True, timeout=timeout)

    def stop(self):
        try:
            colab("stop", "-s", self.name, timeout=300)
        except subprocess.TimeoutExpired:
            pass
        subprocess.run(["ssh", "-F", str(self.config), "-O", "exit", self.host], capture_output=True, timeout=30)


# ------------------------------------------------------------------ job pieces
def runner_source(family, experiment):
    """Identical transformation to launch.launch."""
    source = (HERE / f"{family}.py").read_text()
    entry = ast.parse(source).body[-1]
    if not isinstance(entry, ast.If) or "__name__" not in ast.unparse(entry.test):
        raise ValueError("Runner must have a standalone main guard")
    return ("\n".join(source.splitlines()[:entry.lineno - 1]) + '\nif __name__ == "__main__":\n'
            '    import sys\n    if "--experiment" not in sys.argv:\n'
            f'        sys.argv.extend(["--experiment", {experiment!r}])\n    main()\n')


def base_command(item, out):
    command = [VM_PYTHON, str(out / "experiment.py"), "--data", str(DATA), "--out", str(out),
               "--experiment", item["experiment"], "--threads", str(item["threads"])]
    if item["family"] in RAW_FAMILIES:
        command += ["--raw-data", str(RAW)]
    return command + ["--device", "cuda:0"]


def run_script(item, out, remaining):
    env = ["env", "CUDA_VISIBLE_DEVICES=0"]
    command = base_command(item, out)
    log_path, exit_path = shlex.quote(str(out / "train.log")), shlex.quote(str(out / "train.exit"))
    head = f"#!/bin/bash\nset -o pipefail\ncd {VM_ROOT}\n"
    if len(remaining) == len(PERIODS):  # attempt without prior periods: the exact local invocation
        return (head + shlex.join(env + command) + f" > {log_path} 2>&1\ncode=$?\n"
                f"printf '%s\\n' $code > {exit_path}\nexit $code\n")
    lines = [head + "code=0"]
    for period in remaining:
        lines.append(f"[ $code -eq 0 ] && {{ {shlex.join(env + command + ['--period', period])} >> {log_path} 2>&1"
                     " || code=$?; }")
    lines.append(f"[ $code -eq 0 ] && {{ {shlex.join(env + command + ['--merge'])} >> {log_path} 2>&1 || code=$?; }}")
    lines.append(f"printf '%s\\n' $code > {exit_path}\nexit $code\n")
    return "\n".join(lines)


def local_input_check(work):
    cache = RUNS / "input-check-local.json"
    current = {"data": identity(DATA), "raw": identity(RAW)}
    cached = read_json(cache, {})
    if cached.get("identity") != current:
        result = subprocess.run([str(REPO / ".venv/bin/python"), "-c", PQ_CHECK, str(DATA), str(RAW)],
                                capture_output=True, text=True, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        cached = {"identity": current, "check": json.loads(result.stdout)}
        atomic(cache, cached)
    return cached["check"]


def provision(vm, item, work, out, record):
    q = shlex.quote
    vm.check(f"mkdir -p {q(str(DATA.parent))} {q(str(RAW.parent))} {q(str(out))} {VM_ROOT}")
    vm.put(REQUIREMENTS, VM_ROOT + "/requirements.txt")
    vm.check(f"cd {VM_ROOT} && (test -x .venv/bin/python || uv venv -q --python 3.12 .venv) && "
             "uv pip install -q --python .venv/bin/python --index-strategy unsafe-best-match "
             "--extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt", timeout=2400)
    environment = json.loads(vm.check(VM_PYTHON + " -c " + q(
        "import json,sys,importlib.metadata as m,torch;assert torch.cuda.is_available();"
        "print(json.dumps(dict(python=sys.version.split()[0],gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,"
        "packages={d.metadata['Name']:d.version for d in m.distributions()})))")))
    norm = lambda name: name.lower().replace("_", "-")
    pinned = dict(line.split("==", 1) for line in REQUIREMENTS.read_text().split() if "==" in line)
    installed = {norm(k): v for k, v in environment["packages"].items()}
    mismatched = {k: (v, installed.get(norm(k))) for k, v in pinned.items()
                  if (installed.get(norm(k)) or "").split("+")[0] != v}
    if mismatched:
        raise RuntimeError(f"VM packages differ from the pinned local versions: {mismatched}")
    for path in (DATA, RAW):
        vm.put(path, path)
    remote_check = json.loads(vm.check(f"{VM_PYTHON} - {q(str(DATA))} {q(str(RAW))}", input=PQ_CHECK.encode(), timeout=900))
    local_check = local_input_check(work)
    if remote_check != local_check:
        raise RuntimeError("Input copy differs from the local files (bytes/rows/columns/nulls/sums)")
    atomic(work / f"input-check.{vm.name}.json", remote_check)
    for name in ("experiment.py", "runner_at_launch.py"):
        vm.put(work / "launch/experiment.py", out / name)
    vm.put(work / "launch/launch.json", out / "launch.json")
    vm.check(f"chmod 444 {q(str(out / 'experiment.py'))} {q(str(out / 'runner_at_launch.py'))} && "
             f"cmp {q(str(out / 'experiment.py'))} {q(str(out / 'runner_at_launch.py'))}")
    checkpoint = json.loads(vm.check(f"{VM_PYTHON} - {q(str(out))} {q(item['experiment'])}", input=PREFETCH.encode(),
                                     timeout=1800))
    if checkpoint:
        local_file = next((Path.home() / ".cache/huggingface/hub").glob(
            f"models--{checkpoint['repo'].replace('/', '--')}/snapshots/{checkpoint['revision']}/{checkpoint['filename']}"), None)
        if local_file is not None and local_file.stat().st_size != checkpoint["bytes"]:
            raise RuntimeError("Pinned checkpoint size differs from the local copy")
    record.setdefault("colab_attempts", []).append({
        "session": vm.name, "endpoint": vm.endpoint, "gpu": environment["gpu"], "python": environment["python"],
        "cuda": environment["cuda"], "checkpoint": checkpoint, "input_check": str(work / f"input-check.{vm.name}.json"),
        "provisioned_utc": now()})
    return environment


def restore_periods(vm, work, out, done):
    """Upload previously synced finished periods (and the log) to a fresh VM."""
    if not done:
        return
    synced = work / "synced"
    archive = work / "resume.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for period in done:
            for path in sorted(synced.glob(f"{period}_*")):
                tar.add(path, arcname=path.name)
        if (synced / "train.log").exists():
            tar.add(synced / "train.log", arcname="train.log")
    vm.put(archive, VM_ROOT + "/resume.tar.gz")
    vm.check(f"tar -C {shlex.quote(str(out))} -xzf {VM_ROOT}/resume.tar.gz")


def remote_state(vm, out):
    code = ("import json,os,subprocess;o=" + repr(str(out)) + ";"
            "e=os.path.join(o,'train.exit');"
            "print(json.dumps(dict(exit=open(e).read().strip() if os.path.exists(e) else None,"
            "alive=subprocess.run(['tmux','has-session','-t','train'],capture_output=True).returncode==0,"
            "done=[p for p in ('F1','F2','F3') if os.path.exists(os.path.join(o,p+'_metadata.json'))])))")
    return json.loads(vm.check("python3 -c " + shlex.quote(code), timeout=120))


def sync_period(vm, work, out, period):
    synced = work / "synced"
    synced.mkdir(exist_ok=True)
    name = f"{period}.tar.gz"
    names = [f"{period}_metadata.json", f"{period}_predictions.npz", f"{period}_models"]
    vm.check(f"cd {shlex.quote(str(out))} && tar --ignore-failed-read -czf {VM_ROOT}/{name} "
             + " ".join(shlex.quote(n) for n in names), timeout=900)
    vm.get(f"{VM_ROOT}/{name}", work / name)
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        with tarfile.open(work / name) as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if not parts or ".." in parts or not parts[0].startswith(period + "_") or not (member.isfile() or member.isdir()):
                    raise ValueError("Unexpected period archive member")
            tar.extractall(temporary, filter="data")
        for path in Path(temporary).iterdir():
            target = synced / path.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            path.rename(target)


def collect(vm, work, out, experiment):
    collected = work / "collected"
    collected.mkdir(exist_ok=True)
    archive = collected / f"{experiment}.{vm.name}.tar.gz"
    vm.check(f"tar -C {shlex.quote(str(out.parent))} -czf {VM_ROOT}/final.tar.gz {shlex.quote(experiment)}", timeout=1800)
    vm.get(f"{VM_ROOT}/final.tar.gz", archive, timeout=3600)
    original = collected / experiment
    if original.exists():
        shutil.rmtree(original)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != experiment or ".." in parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unexpected archive member")
        stream.extractall(collected, filter="data")
    return original


def install(work, experiment, original, record):
    sys.path.insert(0, str(HERE))
    import publish
    out = HERE / "runs" / experiment
    if out.exists():
        raise FileExistsError("Local run directory appeared; refusing overwrite")
    if record["input_identity"] != identity(DATA):
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
        previous = publish.HERE
        try:
            publish.HERE = root
            publish.validate_run(experiment)
        finally:
            publish.HERE = previous
        staged.rename(out)
    return out


def write_status(work, item, state, vm=None, **fields):
    status = read_json(work / "status.json", {})
    status.update(experiment=item["experiment"], host="colab", state=state, updated=time.time(),
                  session=vm.name if vm else status.get("session"), **fields)
    atomic(work / "status.json", status)
    update_item(item["experiment"], state=state, session=status.get("session"),
                attempts=item.get("attempts", 0), periods_done=status.get("periods_done", []))


def keepalive_async(vm):
    def touch():
        try:
            colab("exec", "-s", vm.name, "--timeout", "45", timeout=90, input="1\n")
        except Exception:
            pass
    threading.Thread(target=touch, daemon=True).start()


def lb_status(experiment, status, note):
    try:
        sys.path.insert(0, str(REPO / "leaderboard"))
        import lbstatus
        lbstatus.leaderboard_submissions = lambda entries, dataset: {}
        lbstatus.commit(lambda e: lbstatus.upsert(e, "seongeun", experiment, status, note=note),
                        message=f"Fresh-df ffill rerun {status}: {experiment} (Colab queue)")
    except Exception as error:
        log("lbstatus_failed", experiment=experiment, error=str(error)[:200])


# ------------------------------------------------------------------ one attempt / one job
def attempt(item, work, number):
    experiment = item["experiment"]
    out = HERE / "runs" / experiment
    record = json.loads((work / "launch/launch.json").read_text())
    done = sorted(p for p in PERIODS if (work / "synced" / f"{p}_metadata.json").exists())
    remaining = [p for p in PERIODS if p not in done]
    (work / "synced").mkdir(exist_ok=True)
    vm = VM(work, f"cq{number}-{experiment.removeprefix('e2000fs_')}"[:40].replace("_", "-"), item["gpu"])
    write_status(work, item, "provisioning", vm, periods_done=done, attempt=number)
    try:
        vm.create()
        log("session_ready", experiment=experiment, session=vm.name, endpoint=vm.endpoint)
        provision(vm, item, work, out, record)
        restore_periods(vm, work, out, done)
        script = run_script(item, out, remaining)
        (work / f"run.{vm.name}.sh").write_text(script)
        vm.put(work / f"run.{vm.name}.sh", out / "run.sh")
        if remaining != list(PERIODS):
            record.setdefault("recovery", []).append({"session": vm.name, "reused_periods": done,
                                                      "run_periods": remaining, "utc": now(),
                                                      "rule": "finished periods kept; missing periods rerun with --period, then --merge"})
        atomic(work / "launch/launch.json", record)
        vm.put(work / "launch/launch.json", out / "launch.json")
        vm.check(f"tmux new-session -d -s train bash {shlex.quote(str(out / 'run.sh'))}")
        write_status(work, item, "running", vm, periods_done=done, attempt=number, started_utc=now())
        log("started", experiment=experiment, session=vm.name, remaining=remaining)
        failures = 0
        while True:
            time.sleep(POLL_SECONDS)
            try:
                state = remote_state(vm, out)
                for period in state["done"]:
                    if period not in done:
                        sync_period(vm, work, out, period)
                        done.append(period)
                        log("period_synced", experiment=experiment, period=period)
                vm.get(str(out / "train.log"), work / "synced/train.log", timeout=300)
                snapshot = json.loads(vm.check("python3 - --collect colab", input=MONITOR.read_bytes(), timeout=120)
                                      .strip().split("\n")[-1])
                failures = 0
                write_status(work, item, "running", vm, periods_done=sorted(done), snapshot=snapshot, error=None)
            except Lost:
                raise
            except Exception as error:
                failures += 1
                write_status(work, item, "running", vm, error=f"poll failure {failures}: {str(error)[:160]}")
                if failures >= LOST_AFTER_FAILURES:
                    raise Lost(f"{failures} consecutive poll failures: {error}")
                continue
            if state["exit"] is not None:
                write_status(work, item, "collecting", vm, exit_code=state["exit"])
                original = collect(vm, work, out, experiment)
                if state["exit"] != "0":
                    return "failed", f"exit {state['exit']}; logs in {original}"
                installed = install(work, experiment, original, record)
                atomic(work / "launch/launch.json", record)
                return "installed", str(installed)
            if not state["alive"]:
                raise Lost("training tmux session ended without train.exit")
            keepalive_async(vm)
    finally:
        vm.stop()
        log("session_stopped", session=vm.name)


def prepare(item, work):
    experiment, out = item["experiment"], HERE / "runs" / item["experiment"]
    launch = work / "launch"
    if (launch / "launch.json").exists():
        return
    manifest = read_json(HERE / "manifest.json", {})
    entry = next(e for e in manifest["experiments"] if e["experiment"] == experiment)
    source = runner_source(item["family"], experiment)
    launch.mkdir(parents=True, exist_ok=True)
    (launch / "experiment.py").write_text(source)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    record = {"experiment": experiment, "family": item["family"], "host": "colab", "tmux_session": "colab:train",
              "started_utc": now(), "command": base_command(item, out), "output": str(out),
              "baseline_submission_utc": entry["original_submission_utc"], "threads": item["threads"],
              "input_identity": identity(DATA), "raw_input_identity": identity(RAW), "training_target_ffill": True,
              "source_snapshot": str(out / "experiment.py"), "source_bytes": len(source.encode()),
              "gpu": 0, "gpu_lease": None, "training_host": "colab", "requested_gpu": item["gpu"],
              "path_layout": "VM mirrors local absolute input/run paths; no metadata rebasing",
              "requirements": REQUIREMENTS.read_text().split(), "source_git_commit": git}
    atomic(launch / "launch.json", record)


def run_job(item):
    experiment = item["experiment"]
    work = RUNS / experiment
    work.mkdir(parents=True, exist_ok=True)
    prepare(item, work)
    lb_status(experiment, "진행", f"fresh df+ffill; Colab CLI {item['gpu']} queue, original full recipe, pinned local library versions")
    while item.get("attempts", 0) < MAX_ATTEMPTS:
        cu = balance()
        if cu is not None and cu < MIN_BALANCE_CU:
            write_status(work, item, "blocked", error=f"Colab balance {cu} CU < {MIN_BALANCE_CU}")
            return "blocked"
        item["attempts"] = item.get("attempts", 0) + 1
        update_item(experiment, attempts=item["attempts"])
        try:
            outcome, detail = attempt(item, work, item["attempts"])
        except Lost as error:
            log("vm_lost", experiment=experiment, attempt=item["attempts"], error=str(error)[:300])
            write_status(work, item, "lost_retrying", error=str(error)[:300])
            continue
        except Exception as error:
            log("attempt_error", experiment=experiment, attempt=item["attempts"], error=f"{type(error).__name__}: {error}"[:400])
            write_status(work, item, "lost_retrying", error=f"{type(error).__name__}: {str(error)[:300]}")
            continue
        write_status(work, item, outcome, detail=detail, error=None if outcome == "installed" else detail)
        if outcome == "installed":
            log("installed", experiment=experiment, out=detail)
        else:
            lb_status(experiment, "실패", f"Colab run {detail[:120]}; logs preserved")
        return outcome
    write_status(work, item, "failed", error=f"VM lost {MAX_ATTEMPTS} times")
    lb_status(experiment, "중단", f"Colab VM lost {MAX_ATTEMPTS} times; synced periods preserved")
    return "failed"


def run(args):
    with (HERE / ".colab-queue-driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log("driver_started")
        while True:
            with QueueFile() as q:
                item = next((dict(i) for i in q.data["items"] if i.get("state") in (
                    "queued", "lost_retrying", "provisioning", "running", "collecting")), None)
            if item is None:
                time.sleep(args.idle)
                continue
            try:
                outcome = run_job(item)
            except Exception as error:
                log("job_error", experiment=item["experiment"], error=f"{type(error).__name__}: {error}"[:400])
                outcome = "failed"
                update_item(item["experiment"], state="failed", error=str(error)[:300])
            log("job_done", experiment=item["experiment"], outcome=outcome)
            if outcome == "blocked":
                time.sleep(600)


def add(args):
    experiment = args.experiment
    if (HERE / "runs" / experiment).exists():
        raise FileExistsError("Experiment already launched locally")
    with QueueFile() as q:
        if any(i["experiment"] == experiment for i in q.data["items"]):
            raise ValueError("Already queued for Colab")
        registry_path = HERE / "registry.json"
        registry = json.loads(registry_path.read_text())
        family = args.family or (registry.get(experiment) or {}).get("family")
        if family is None:
            raise ValueError("Unknown family; pass --family")
        if experiment in registry:
            backup = HERE / f"registry.before-colab-{experiment}.json"
            shutil.copy2(registry_path, backup)
            registry.pop(experiment)
            atomic(registry_path, registry)
        manifest_path = HERE / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["experiments"]:
            if entry["experiment"] == experiment:
                entry["state"] = "assigned_colab"
                entry["assignment_note"] = f"{now()[:10]} moved to the Colab CLI GPU queue (colab-queue.json)"
        atomic(manifest_path, manifest)
        q.data["items"].append({"experiment": experiment, "family": family, "gpu": args.gpu, "threads": args.threads,
                                "state": "queued", "attempts": 0, "added_utc": now()})
        q.save()
    print(json.dumps({"queued": experiment, "family": family, "gpu": args.gpu}))


def status(args):
    print(json.dumps(read_json(QUEUE, {"items": []}), indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("add")
    a.add_argument("experiment")
    a.add_argument("--family")
    a.add_argument("--gpu", default="L4")
    a.add_argument("--threads", type=int, default=2)
    r = sub.add_parser("run")
    r.add_argument("--idle", type=float, default=60)
    sub.add_parser("status")
    args = parser.parse_args()
    {"add": add, "run": run, "status": status}[args.action](args)


if __name__ == "__main__":
    main()
