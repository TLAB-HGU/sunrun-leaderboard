"""Resume the verified retraining queue, collect finished runs, and publish them.

This is ordinary campaign state, not OMX/hidden goal state. Every completion
requires an exit file and the publisher's independent artifact checks.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import launch

HERE = Path(__file__).resolve().parent
MEMORY_GIB = {"calib11": 2.5, "top30": 1.6, "strategies": 12.0, "distributions": 3.5,
              "advanced": 2.0, "calibration": 3.0, "structured": 3.5, "direct": 8.0,
              "foundation": 6.0, "tcn": 4.0, "special": 6.0, "tda": 4.0,
              "pooling": 5.0, "special_hsspeak": 6.0}
HOST_LIMIT = {"local": 2, "golab2": 0}
EXECUTION_HOSTS = ("local",)  # User instruction: all remaining runs on this host, with DL on GPU only.


def emit(event, **fields):
    print(json.dumps({"utc": datetime.now(UTC).isoformat(), "event": event, **fields}), flush=True)


def status_update(name, status, note):
    sys.path.insert(0, str(HERE.parents[1] / "leaderboard"))
    import lbstatus
    # Status publication must not reclassify unrelated active runs using stale
    # receipts. The publisher separately verifies and retires completed runs.
    lbstatus.leaderboard_submissions = lambda entries, dataset: {}
    lbstatus.commit(lambda entries: lbstatus.upsert(entries, "seongeun", name, status, note=note),
                    message=f"Fresh-df ffill rerun {status}: {name}")


def memory_local():
    values = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    return values["MemAvailable"] / 1024**2


def memory_total():
    return next(int(line.split()[1]) / 1024**2 for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:"))


def memory_required(name, family):
    return 20.0 if name == "e2000fs_mimo" else MEMORY_GIB[family]


def records():
    return {p.parent.name: json.loads(p.read_text()) for p in (HERE / "runs").glob("*/launch.json")}


def tick(runtime):
    registry = json.loads((HERE / "registry.json").read_text())
    launches = records()
    active = {"local": 0, "golab2": 0}
    reserved = 0.0
    publication_attempted = False
    for name, record in launches.items():
        state = runtime.setdefault(name, {})
        out = HERE / "runs" / name
        if (out / "submission.json").exists():
            state["state"] = "submitted"
            continue
        host = record["host"]
        exit_path = out / "train.exit"
        if exit_path.exists():
            code = int(exit_path.read_text())
        else:
            code = None
        if code is None:
            exists = host == "local" and subprocess.run(
                ["tmux", "has-session", "-t", record["tmux_session"]],
                capture_output=True, check=False).returncode == 0
            if not exists:
                if state.get("state") != "interrupted":
                    emit("lost_session", experiment=name, host=host)
                    status_update(name, "중단", "Training session lost without an exit receipt; logs preserved, diagnosis required")
                    state["retry_after"] = time.time() + 60
                state["state"] = "interrupted"
                continue
            active[host] += 1
            if name in registry:
                reserved += memory_required(name, registry[name]["family"])
            state["state"] = "running"
            continue
        if code:
            if state.get("state") != "failed":
                emit("training_failed", experiment=name, exit_code=code)
                status_update(name, "실패", f"fresh-df ffill rerun exit {code}; error log preserved for diagnosis")
            state.update(state="failed", exit_code=code)
            continue
        state["state"] = "trained"
        # Failed publications retry without repeating training. Avoid tight
        # retry loops and leave the complete traceback for parent diagnosis.
        if publication_attempted or time.time() < state.get("publication_retry_after", 0):
            continue
        publication_attempted = True
        with (out / "publish.log").open("a") as log:
            result = subprocess.run([launch.LOCAL_PYTHON, str(HERE / "publish.py"), name], stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            state["publication_retry_after"] = time.time() + 300
            state["publication_failures"] = state.get("publication_failures", 0) + 1
            emit("publication_failed", experiment=name, exit_code=result.returncode)
        else:
            state["state"] = "submitted"
            emit("submitted", experiment=name)
    # Re-query memory after collection/publication and admit one job at a time.
    for host in EXECUTION_HOSTS:
        available = min(memory_local(), memory_total() - reserved - 3.0)
        gpu_available = (any(r["family"] in launch.DL_FAMILIES for r in registry.values())
                         and launch.gpu0_available())
        def priority(item, allow_gpu=gpu_available):
            name, recipe = item
            if runtime.get(name, {}).get("state") == "interrupted":
                return 0
            if allow_gpu and recipe["family"] == "tcn":
                return 1
            if allow_gpu and recipe["family"] == "foundation":
                return 2 if name.endswith(("pooled", "recursive")) else 3
            return 4
        ordered = sorted(registry.items(), key=priority)
        for name, recipe in ordered:
            state = runtime.setdefault(name, {})
            retry = (name in launches and state.get("state") == "interrupted"
                     and launches[name]["host"] == host and time.time() >= state.get("retry_after", 0))
            if (name in launches and not retry) or active[host] >= HOST_LIMIT[host]:
                continue
            family = recipe["family"]
            if family in launch.DL_FAMILIES and not gpu_available:
                continue
            gpu = 0 if family in launch.DL_FAMILIES else None
            if family in ("foundation", "tcn", "tda", "special_hsspeak") and host != "local":
                continue  # baseline checkpoints and CPU dependencies verified locally
            required = memory_required(name, family)
            if available < required + 3.0:
                continue
            if family == "special_hsspeak":
                try:
                    launch.physics_dependency()
                except (OSError, ValueError, KeyError) as error:
                    if state.get("dependency_note") != str(error):
                        emit("waiting_for_physics", experiment=name, reason=str(error))
                    state["dependency_note"] = str(error)
                    continue
                state.pop("dependency_note", None)
            device = "GPU0" if gpu is not None else "CPU"
            status_update(name, "진행", f"fresh df+ffill; {host} {device}, original full recipe, independent tmux/log")
            if retry:
                launch.resume(launches[name])
                emit("resumed", experiment=name, host=host)
            else:
                launch.launch(name, family, host, 2 if host == "local" else recipe["threads"], gpu=gpu)
            launches[name] = json.loads((HERE / "runs" / name / "launch.json").read_text())
            runtime[name] = {"state": "running"}
            active[host] += 1
            if gpu is not None:
                gpu_available = False
            available -= required
            emit("launched", experiment=name, family=family, host=host)
    summary = {state: sum(runtime.get(name, {}).get("state") == state for name in registry)
               for state in ("running", "trained", "submitted", "failed")}
    summary["registered_recipes"] = len(registry)
    emit("summary", **summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    runtime_path = HERE / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    while True:
        try:
            summary = tick(runtime)
            runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
            if args.once or summary["submitted"] == summary["registered_recipes"]:
                return
        except Exception:
            emit("controller_error", error=traceback.format_exc())
            runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
            if args.once:
                raise
        time.sleep(30)


if __name__ == "__main__":
    with (HERE / ".controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
