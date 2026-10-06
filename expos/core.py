"""Experiment OS: an append-only, hash-chained experiment graph plus the only sanctioned actions on it.

Every state change is one JSON line in research/expos/graph.jsonl whose `hash` covers the previous line's hash,
so edits to history are detectable (`verify`). Agents act only through these functions / the CLI.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
ROOT = LB.parent
RESEARCH = Path(os.environ.get("EXPOS_RESEARCH", ROOT / "research"))
GRAPH = RESEARCH / "expos" / "graph.jsonl"
RUNS = RESEARCH / "expos" / "runs"
LEASE_DIR = LB / "store/ch-breakthrough-v2/gpu-leases"  # shared with existing experiment code

PROTECTED = ["scorer/scoring.py", "store/ch-v1/ace.parquet", "experiments/mse20_hss_arrival/diagnostics.py",
             "experiments/mse20_scope_expansion/diagnostics.py", "experiments/mse20_scope_expansion/protocol.py",
             "expos/core.py", "expos/evaluate.py", "expos/cli.py", "expos/experience.py", "expos/archive.py", "expos/axes.py", "expos/rocv.py"]
# Official-period truth or official predictions must never feed an experiment.
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay.parquet", "official_check", "official-check",
             "predictions_official_truth"]
KINDS = ("reference", "pilot", "full", "final")
VERDICTS = ("accept", "reject", "needs-verify")


class ExposError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def _locked():
    GRAPH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _lines():
    if not GRAPH.exists():
        return []
    return [json.loads(l) for l in GRAPH.read_text().splitlines() if l.strip()]


def _digest(prev, body):
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()


def verify():
    """Check the hash chain; raise on any edited, dropped or reordered line."""
    prev = ""
    for i, e in enumerate(_lines()):
        body = {k: v for k, v in e.items() if k != "hash"}
        if e.get("seq") != i or e.get("prev") != prev or e.get("hash") != _digest(prev, body):
            raise ExposError(f"graph chain broken at line {i}")
        prev = e["hash"]
    return len(_lines())


def append(event, **fields):
    with _locked():
        lines = _lines()
        verify()
        prev = lines[-1]["hash"] if lines else ""
        body = {"seq": len(lines), "prev": prev, "ts": now(), "event": event, **fields}
        rec = {**body, "hash": _digest(prev, body)}
        with open(GRAPH, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def state():
    s = {"init": None, "directions": {}, "experiments": {}, "frozen": None, "official": None}
    for e in _lines():
        ev = e["event"]
        if ev == "init":
            s["init"] = e
        elif ev == "direction":
            s["directions"][e["id"]] = {**s["directions"].get(e["id"], {}), **{k: e[k] for k in e if k in
                                        ("id", "parent", "title", "rationale", "state", "reason")}}
        elif ev == "register":
            s["experiments"][e["id"]] = {**e, "runs": [], "evals": [], "reviews": [], "flags": list(e.get("flags", []))}
        elif ev in ("run", "evaluate", "review", "flag"):
            if ev == "run":
                s["experiments"][e["id"]].pop("running", None)
            x = s["experiments"][e["id"]]
            {"run": x["runs"], "evaluate": x["evals"], "review": x["reviews"]}.get(ev, x["flags"]).append(
                e if ev != "flag" else e["flag"])
        elif ev == "rocv":
            s["experiments"][e["id"]].setdefault("rocv", []).append(e)
        elif ev == "protect":  # admin re-baseline of protected hashes after an audited harness change
            s["init"] = {**s["init"], "protected": e["protected"]}
        elif ev == "start":
            s["experiments"][e["id"]]["running"] = e
        elif ev == "freeze":
            s["frozen"] = e
        elif ev == "official":
            s["official"] = e
    for x in s["experiments"].values():  # a start without a run whose process is gone was killed or crashed
        r = x.get("running")
        if r and r.get("pid"):
            try:
                os.kill(r["pid"], 0)
            except ProcessLookupError:
                x["running"] = {**r, "stale": True}
            except PermissionError:
                pass
    return s


def protected_hashes():
    return {p: sha256(LB / p) for p in PROTECTED if (LB / p).exists()}


def check_protected(s=None):
    s = s or state()
    if not s["init"]:
        raise ExposError("expos not initialised: run `python -m expos init`")
    changed = [p for p, h in s["init"]["protected"].items() if not (LB / p).exists() or sha256(LB / p) != h]
    return changed


def init(force=False):
    s = state()
    if s["init"] and not force:
        raise ExposError("already initialised")
    return append("init", protected=protected_hashes(), forbidden=FORBIDDEN)


def protect(reason):
    """Re-baseline protected hashes; only for audited harness changes made by the operator (logged in the chain)."""
    if not reason:
        raise ExposError("protect needs a reason")
    return append("protect", protected=protected_hashes(), reason=reason)


def add_direction(id, title, rationale, parent=None):
    if not re.fullmatch(r"D[0-9A-Za-z_.-]+", id):
        raise ExposError("direction id must look like D1, D1.2, D-img-a")
    s = state()
    if id in s["directions"]:
        raise ExposError(f"direction {id} exists")
    if parent and parent not in s["directions"]:
        raise ExposError(f"unknown parent {parent}")
    return append("direction", id=id, parent=parent, title=title, rationale=rationale, state="active", reason="")


def set_direction(id, state_, reason):
    if state_ not in ("active", "paused", "closed", "verify", "branch"):
        raise ExposError("bad direction state")
    if id not in state()["directions"]:
        raise ExposError(f"unknown direction {id}")
    return append("direction", id=id, state=state_, reason=reason)


def scan(text):
    return [p for p in FORBIDDEN if p in text]


def register(direction, owner, hypothesis, cmd, kind="full", code=(), reference=None, experience=()):
    s = state()
    if direction not in s["directions"] or s["directions"][direction]["state"] == "closed":
        raise ExposError(f"direction {direction} unknown or closed")
    if kind not in KINDS:
        raise ExposError(f"kind must be one of {KINDS}")
    if reference and reference not in s["experiments"]:
        raise ExposError(f"unknown reference {reference}")
    code_hashes, blob = {}, cmd
    for c in code:
        p = (LB / c) if not Path(c).is_absolute() else Path(c)
        if not p.exists():
            raise ExposError(f"code file missing: {c}")
        code_hashes[str(c)] = sha256(p)
        blob += "\n" + p.read_text(errors="ignore")
    flags = [f"forbidden:{p}" for p in scan(blob)]
    eid = f"E{len(s['experiments']) + 1:04d}"
    return append("register", id=eid, direction=direction, owner=owner, hypothesis=hypothesis, cmd=cmd, kind=kind,
                  code=code_hashes, reference=reference, experience=list(experience), flags=flags)


def _gpu_order(prefer=None):
    """Free-looking GPUs first: lowest utilisation, then lowest memory; `prefer` goes first if given."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        load = {int(i): (float(u), float(m)) for i, u, m in (l.split(",") for l in out.strip().splitlines())}
    except Exception:
        load = {0: (0, 0), 1: (0, 0)}
    order = sorted(load, key=lambda g: load[g])
    if prefer is not None and prefer in order:
        order.remove(prefer)
        order.insert(0, prefer)
    return order


@contextlib.contextmanager
def gpu_lease(n=1, prefer=None, timeout=1800):
    """Hold exclusive fcntl leases on n GPUs (shared lock files with other experiment code); yields their ids."""
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        held = []
        for gpu in _gpu_order(prefer):
            h = open(LEASE_DIR / f"gpu{gpu}.lock", "w")
            try:
                fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append((gpu, h))
            except OSError:
                h.close()
            if len(held) == n:
                break
        if len(held) == n:
            try:
                yield [g for g, _ in held]
            finally:
                for _, h in held:
                    fcntl.flock(h, fcntl.LOCK_UN)
                    h.close()
            return
        for _, h in held:
            fcntl.flock(h, fcntl.LOCK_UN)
            h.close()
        if time.time() - t0 > timeout:
            raise ExposError(f"no {n} free GPU lease(s) within timeout")
        time.sleep(10)


def run(eid, timeout=3600, gpu=False, gpus=1, prefer=None):
    s = state()
    x = s["experiments"].get(eid)
    if not x:
        raise ExposError(f"unknown experiment {eid}")
    changed = check_protected(s)
    if changed:
        append("flag", id=eid, flag=f"protected_modified:{','.join(changed)}")
        raise ExposError(f"protected files changed: {changed}")
    for c, h in x["code"].items():
        p = (LB / c) if not Path(c).is_absolute() else Path(c)
        if not p.exists() or sha256(p) != h:
            raise ExposError(f"code changed since register: {c} (register a new experiment)")
    out = RUNS / eid
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "EXPOS_OUT": str(out), "EXPOS_EXP": eid, "EXPOS_KIND": x["kind"]}
    log = out / f"run{len(x['runs']) + 1}.log"
    t0 = time.time()
    lease = gpu_lease(gpus, prefer) if (gpu or gpus > 1) else contextlib.nullcontext(None)
    with lease as dev:
        if dev is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, dev))
            env["EXPOS_GPUS"] = env["CUDA_VISIBLE_DEVICES"]
        append("start", id=eid, pid=os.getpid(), gpu=dev)
        with open(log, "w") as f:
            try:
                rc = subprocess.run(["bash", "-lc", x["cmd"]], cwd=LB, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
    outputs = {p.name: sha256(p) for p in sorted(out.glob("*.parquet")) + sorted(out.glob("*.json"))}
    gpu_rec = dev[0] if dev and len(dev) == 1 else dev
    return append("run", id=eid, rc=rc, seconds=round(time.time() - t0, 1), gpu=gpu_rec, log=str(log), outputs=outputs)


def run_background(eid, timeout=3600, gpu=False, gpus=1, prefer=None):
    """Start `run` detached so a worker can launch several experiments (or pipeline stages) at once."""
    if eid not in state()["experiments"]:
        raise ExposError(f"unknown experiment {eid}")
    out = RUNS / eid
    out.mkdir(parents=True, exist_ok=True)
    cmd = [os.sys.executable, "-m", "expos", "run", eid, "--timeout", str(timeout), "--gpus", str(gpus)]
    if gpu:
        cmd.append("--gpu")
    if prefer is not None:
        cmd += ["--prefer-gpu", str(prefer)]
    with open(out / "background.log", "a") as f:
        p = subprocess.Popen(cmd, cwd=LB, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return {"id": eid, "pid": p.pid, "log": str(out / "background.log")}


def review(eid, reviewer, verdict, notes):
    s = state()
    x = s["experiments"].get(eid)
    if not x:
        raise ExposError(f"unknown experiment {eid}")
    if verdict not in VERDICTS:
        raise ExposError(f"verdict must be one of {VERDICTS}")
    if reviewer == x["owner"]:
        raise ExposError("owner cannot review their own experiment")
    if not x["evals"]:
        raise ExposError("evaluate before review: reviewers judge recorded evidence")
    return append("review", id=eid, reviewer=reviewer, verdict=verdict, notes=notes)


def eligible(x):
    """Why an experiment cannot be frozen (empty list = eligible)."""
    why = []
    if not x["evals"]:
        return ["not evaluated"]
    ev = x["evals"][-1]
    if x["flags"]:
        why.append(f"integrity flags {x['flags']}")
    if not ev.get("poison_ok"):
        why.append("no passing poison.json causality proof")
    if not ev.get("gate", {}).get("pass"):
        why.append(f"three-period gate failed: {ev.get('gate', {}).get('reasons')}")
    if not any(r["verdict"] == "accept" for r in x["reviews"]):
        why.append("no accepting review")
    if any(r["verdict"] == "reject" for r in x["reviews"][-1:]):
        why.append("latest review rejects")
    return why


def freeze(eid, final=None):
    s = state()
    if s["frozen"]:
        raise ExposError(f"already frozen: {s['frozen']['id']}")
    x = s["experiments"].get(eid)
    if not x:
        raise ExposError(f"unknown experiment {eid}")
    why = eligible(x)
    if why:
        raise ExposError("not eligible: " + "; ".join(why))
    src = final or eid
    f = s["experiments"].get(src)
    pred = RUNS / src / "predictions_official.parquet"
    if not f or f["flags"] or not pred.exists():
        raise ExposError(f"{src} must be clean and have produced predictions_official.parquet (no truth needed)")
    if final and (f["kind"] != "final" or f["direction"] != x["direction"]):
        raise ExposError("final run must be kind=final in the same direction")
    return append("freeze", id=eid, final=src, predictions=str(pred), predictions_sha256=sha256(pred),
                  eval=x["evals"][-1].get("summary"))
