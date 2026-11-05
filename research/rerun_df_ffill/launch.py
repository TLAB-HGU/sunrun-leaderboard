"""Launch one verified recipe in its own tmux session and record real exit status."""
from __future__ import annotations

import argparse
import ast
import fcntl
import json
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = "/home/yehwan/sunrun-rerun-20261103"
REMOTE_PYTHON = "/home/yehwan/power_forecast/.venv/bin/python"
LOCAL_PYTHON = "/home/t-lab01/sunrun/.venv/bin/python"
DL_FAMILIES = ("foundation", "tcn")
GPU0_LEASE = HERE.parents[1] / "leaderboard/store/ch-breakthrough-v2/gpu-leases/gpu0.lock"


def physics_dependency():
    """Admit HSSPEAK only after complete fresh GONG/PFSS/HUXt reproduction."""
    import numpy as np
    import pandas as pd

    output = HERE / "input/huxt_features.parquet"
    metadata_path = output.with_suffix(".metadata.json")
    if (HERE / "gong_cache/exit.code").read_text().strip() != "0":
        raise ValueError("GONG physics did not finish successfully")
    metadata = json.loads(metadata_path.read_text())
    columns = ["origin", "huxt_max", "huxt_argmax", "huxt_h1", "huxt_h24", "huxt_h48", "huxt_h72"]
    if (metadata.get("stage") not in ("all", "forecast")
            or metadata.get("old_huxt_predictions_used") is not False
            or metadata.get("output_columns") != columns
            or metadata.get("output") != str(output.resolve())
            or not metadata.get("original_sources") or not metadata.get("versions")):
        raise ValueError("GONG physics provenance is incomplete or incompatible")
    frame = pd.read_parquet(output, columns=columns)
    origins = pd.DatetimeIndex(pd.to_datetime(frame.origin, utc=True))
    dates = pd.read_parquet(HERE / "input/df_ffill_trainy.parquet", columns=["origin"], filters=[("h", "==", 1)])
    required = pd.DatetimeIndex(pd.to_datetime(dates.origin, utc=True)).unique().sort_values()
    required = required[required >= pd.Timestamp("2024-06-01", tz="UTC")]
    if (not len(required) or origins.has_duplicates or origins.hasnans
            or not required.isin(origins).all()
            or not np.isfinite(frame[columns[1:]].to_numpy()).all()
            or not frame.huxt_argmax.between(1, 72).all()):
        raise ValueError("GONG physics lacks complete finite roll6/query coverage")
    if (metadata.get("rows") != len(frame) or metadata.get("requested_origins") != len(required)
            or pd.Timestamp(metadata["first_origin"]) != origins.min()
            or pd.Timestamp(metadata["last_origin"]) != origins.max()
            or pd.Timestamp(metadata["origin_start"]) != required.min()
            or pd.Timestamp(metadata["origin_end"]) != required.max()):
        raise ValueError("GONG physics metadata disagrees with input/output coverage")
    def identity(path):
        stat = path.stat()
        return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"features": identity(output), "metadata": identity(metadata_path),
            "required_origins": len(required), "provenance": metadata}


def gpu0_available():
    """Admit GPU0 only when its lease, memory, and utilization are available."""
    GPU0_LEASE.parent.mkdir(parents=True, exist_ok=True)
    with GPU0_LEASE.open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            result = subprocess.run(["nvidia-smi", "--id=0",
                                     "--query-gpu=memory.free,memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True,
                                    text=True, check=False, timeout=15)
            if result.returncode:
                return False
            free, used, utilization = [int(v.strip()) for v in result.stdout.strip().split(",")]
            return free >= 8192 and used < 1024 and utilization < 10
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return False


def validate_device(host, family, gpu):
    if host != "local":
        raise ValueError("Campaign execution is restricted to this local host")
    if family in DL_FAMILIES:
        if gpu != 0:
            raise ValueError("Neural models require local GPU0; CPU fallback is forbidden")
        if not gpu0_available():
            raise RuntimeError("GPU0 is busy or leased; leave the CUDA experiment queued")
    elif gpu is not None:
        raise ValueError("GBM and other CPU recipes must not receive a GPU")


def run_script(command, out, gpu=None):
    executed = ["env", "CUDA_VISIBLE_DEVICES=", *command]
    if gpu is not None:
        GPU0_LEASE.parent.mkdir(parents=True, exist_ok=True)
        executed = ["flock", "-w", "120", str(GPU0_LEASE), "env", "CUDA_VISIBLE_DEVICES=0", *command]
    return ("#!/bin/bash\nset -o pipefail\n" + shlex.join(executed) + " > "
            + shlex.quote(str(out / "train.log")) + " 2>&1\ncode=$?\nprintf '%s\\n' $code > "
            + shlex.quote(str(out / "train.exit")) + "\nexit $code\n")


def remote(command, **kwargs):
    raise RuntimeError("golab2 is decommissioned: all remote operations are forbidden; use local recovered files")


def install_resume(record):
    """Use verified per-block recovery without modifying the frozen model code."""
    if record["host"] != "local":
        raise ValueError("Only local recovery is permitted; golab2 is decommissioned")
    if record["family"] not in ("calib11", "top30") or record["experiment"].endswith("_histprior"):
        return
    out = Path(record["output"])
    local_out = HERE / "runs" / record["experiment"]
    helper = (HERE / "resume_run.py").read_bytes()
    command = [REMOTE_PYTHON if record["host"] == "golab2" else LOCAL_PYTHON,
               str(out / "resume_run.py"), "--source", str(out / "experiment.py"),
               "--launch-source", str(out / "runner_at_launch.py"), "--data", record["input_identity"]["path"],
               "--data-bytes", str(record["input_identity"]["bytes"]),
               "--data-mtime-ns", str(record["input_identity"]["mtime_ns"]),
               "--family", record["family"], "--experiment", record["experiment"],
               "--out", str(out), "--threads", str(record["threads"])]
    script = run_script(command, out)
    if record["host"] == "golab2":
        remote("cat > " + shlex.quote(str(out / "resume_run.new.py")), input=helper)
        remote("chmod 444 " + shlex.quote(str(out / "resume_run.new.py")) + " && mv -f "
               + shlex.quote(str(out / "resume_run.new.py")) + " " + shlex.quote(str(out / "resume_run.py")))
        remote("cat > " + shlex.quote(str(out / "run.new.sh")), input=script.encode())
        remote("mv -f " + shlex.quote(str(out / "run.new.sh")) + " " + shlex.quote(str(out / "run.sh")))
    else:
        temporary = out / "resume_run.new.py"
        temporary.write_bytes(helper)
        temporary.chmod(0o444)
        temporary.replace(out / "resume_run.py")
        (out / "run.new.sh").write_text(script)
        (out / "run.new.sh").replace(out / "run.sh")
    record["recovery_command"] = command
    record["recovery_helper"] = "resume_run.py; original model kernels and input unchanged"
    (local_out / "launch.json").write_text(json.dumps(record, indent=2) + "\n")


def resume(record):
    """Retry an interrupted owned run using the same immutable source and input."""
    name, host = record["experiment"], record["host"]
    if host != "local":
        raise ValueError("Only local resume is permitted; golab2 is decommissioned")
    if record["family"] == "special_hsspeak":
        if host != "local" or physics_dependency() != record.get("physics_input"):
            raise ValueError("HSSPEAK physics input changed since launch")
    validate_device(host, record["family"], record.get("gpu"))
    if record["family"] in DL_FAMILIES:
        command = record["command"]
        if "--device" not in command or command[command.index("--device") + 1:command.index("--device") + 2] != ["cuda:0"]:
            raise ValueError("Neural resume command must explicitly select cuda:0")
    install_resume(record)
    out = Path(record["output"])
    if "recovery_command" not in record:
        (out / "run.sh").write_text(run_script(record["command"], out, record.get("gpu")))
    session = record["tmux_session"]
    check = ('from pathlib import Path\nimport json,subprocess\np=Path(' + repr(str(out)) + ')\n'
             's=' + repr(session) + '\n'
             'assert subprocess.run(["tmux","has-session","-t",s],capture_output=True).returncode!=0\n'
             'assert (p/"experiment.py").read_bytes()==(p/"runner_at_launch.py").read_bytes()\n'
             'data=Path(' + repr(record["input_identity"]["path"]) + ')\n'
             'assert data.stat().st_size==' + str(record["input_identity"]["bytes"]) + '\n'
             'assert data.stat().st_mtime_ns==' + str(record["input_identity"]["mtime_ns"]) + '\n'
             'for proc in Path("/proc").iterdir():\n'
             ' if not proc.name.isdigit():continue\n'
             ' try:args=(proc/"cmdline").read_bytes().decode().split(chr(0))\n'
             ' except (FileNotFoundError,PermissionError,ProcessLookupError):continue\n'
             ' assert str(p/"experiment.py") not in args,"Original training process is still alive"\n'
             'n=len(list(p.glob("train.interrupted-*.log")))+1\n'
             'if(p/"train.log").exists():(p/"train.log").rename(p/("train.interrupted-%d.log"%n))\n'
             'if(p/"train.exit").exists():(p/"train.exit").rename(p/("train.interrupted-%d.exit"%n))\n')
    if host == "golab2":
        remote(REMOTE_PYTHON + " -", input=check.encode())
        remote("tmux new-session -d -s " + shlex.quote(session) + " bash " + shlex.quote(str(out / "run.sh")))
    else:
        subprocess.run([LOCAL_PYTHON, "-"], input=check.encode(), check=True)
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", str(out / "run.sh")], check=True)
    local = HERE / "runs" / name
    with (local / "restarts.jsonl").open("a") as log:
        log.write(json.dumps({"utc": datetime.now(UTC).isoformat(), "reason": "lost session or host restart",
                              "source_and_input_revalidated": True}) + "\n")


def launch(experiment, family, host, threads, gpu=None):
    validate_device(host, family, gpu)
    manifest = json.loads((HERE / "manifest.json").read_text())
    entry = next(e for e in manifest["experiments"] if e["experiment"] == experiment)
    if experiment in manifest.get("user_excluded_experiments", []) or entry.get("state") == "excluded_by_user":
        raise ValueError("Experiment removed from execution by user instruction")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", experiment) or threads < 1:
        raise ValueError("Invalid experiment or thread count")
    physics = None
    if family == "special_hsspeak":
        if host != "local" or gpu is not None:
            raise ValueError("HSSPEAK requires the original local CPU recipe and verified local physics")
        physics = physics_dependency()
    script = HERE / f"{family}.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    root = Path(REMOTE) if host == "golab2" else HERE
    out = root / "runs" / experiment
    local_out = HERE / "runs" / experiment
    if local_out.exists() and any(local_out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {local_out}")
    local_out.mkdir(parents=True, exist_ok=True)
    launch_record = local_out / "launch.json"
    if launch_record.exists():
        raise FileExistsError(f"Already launched: {experiment}; inspect its recorded state")
    source = script.read_text()
    main_entry = ast.parse(source).body[-1]
    if not isinstance(main_entry, ast.If) or "__name__" not in ast.unparse(main_entry.test):
        raise ValueError("Runner must have a standalone main guard")
    source = ("\n".join(source.splitlines()[:main_entry.lineno - 1]) + '\nif __name__ == "__main__":\n'
              '    import sys\n    if "--experiment" not in sys.argv:\n'
              f'        sys.argv.extend(["--experiment", {experiment!r}])\n    main()\n')
    compile(source, str(out / "experiment.py"), "exec")
    (local_out / "experiment.py").write_text(source)
    (local_out / "runner_at_launch.py").write_text(source)
    (local_out / "experiment.py").chmod(0o444)
    (local_out / "runner_at_launch.py").chmod(0o444)
    command = [REMOTE_PYTHON if host == "golab2" else LOCAL_PYTHON,
               str(out / "experiment.py"), "--data", str(root / "input/df_ffill_trainy.parquet"),
               "--out", str(out), "--experiment", experiment, "--threads", str(threads)]
    if family in ("strategies", "foundation", "direct", "tcn", "special", "special_hsspeak"):
        command += ["--raw-data", str(root / "raw/df.parquet" if host == "golab2"
                                       else HERE.parents[1] / "data-v2/input/df.parquet")]
    if family == "tda":
        command += ["--raw-suvi-dir", str(HERE.parents[1] / "data-v2/raw/SUVI/nc/g18")]
    if physics is not None:
        command += ["--huxt-features", physics["features"]["path"]]
    if gpu is not None:
        command += ["--device", "cuda:0"]
    session = "srff-" + experiment
    run_script_text = run_script(command, out, gpu)
    if host == "golab2":
        remote("mkdir " + shlex.quote(str(out)))
        remote("cat > " + shlex.quote(str(out / "experiment.py")), input=source.encode())
        remote("cat > " + shlex.quote(str(out / "runner_at_launch.py")), input=source.encode())
        remote("chmod 444 " + shlex.quote(str(out / "experiment.py")) + " " + shlex.quote(str(out / "runner_at_launch.py")))
        remote("cat > " + shlex.quote(str(out / "run.sh")), input=run_script_text.encode())
        identity_script = ('from pathlib import Path\nimport json\np=Path(' + repr(str(root / "input/df_ffill_trainy.parquet"))
                           + ')\ns=p.stat()\nprint(json.dumps(dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)))\n')
        identity = json.loads(remote(REMOTE_PYTHON + " -", input=identity_script.encode(), capture_output=True).stdout)
    else:
        (out / "run.sh").write_text(run_script_text)
        data = HERE / "input/df_ffill_trainy.parquet"
        stat = data.stat()
        identity = {"path": str(data), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if identity["bytes"] != (HERE / "input/df_ffill_trainy.parquet").stat().st_size:
        raise ValueError("Training input size differs from the prepared snapshot")
    record = {"experiment": experiment, "family": family, "host": host, "tmux_session": session,
              "started_utc": datetime.now(UTC).isoformat(), "command": command,
              "output": str(out), "baseline_submission_utc": entry["original_submission_utc"],
              "threads": threads, "input_identity": identity, "training_target_ffill": True,
              "source_snapshot": str(out / "experiment.py"), "source_bytes": len(source.encode()),
              "gpu": gpu, "gpu_lease": str(GPU0_LEASE) if gpu is not None else None}
    if physics is not None:
        record["physics_input"] = physics
    launch_record.write_text(json.dumps(record, indent=2) + "\n")
    install_resume(record)
    if host == "golab2":
        remote("tmux new-session -d -s " + shlex.quote(session) + " bash " + shlex.quote(str(out / "run.sh")))
    else:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", str(out / "run.sh")], check=True)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--family", choices=("calib11", "top30", "strategies", "distributions", "advanced",
                                           "foundation", "calibration", "direct", "structured", "tcn", "special", "tda",
                                           "pooling", "special_hsspeak"), required=True)
    parser.add_argument("--host", choices=("local", "golab2"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gpu", type=int, choices=(0,))
    args = parser.parse_args()
    launch(args.experiment, args.family, args.host, args.threads, args.gpu)
