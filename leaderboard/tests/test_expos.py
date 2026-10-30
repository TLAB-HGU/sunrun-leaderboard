import importlib
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def expos(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPOS_RESEARCH", str(tmp_path))
    from expos import core, evaluate, experience
    for m in (core, evaluate, experience):
        importlib.reload(m)
    core.init()
    core.add_direction("D1", "test", "r")
    return core, evaluate, experience


def _write_preds(core, eid, fn, step="6h"):
    filled, _ = importlib.import_module("expos.evaluate")._ace()
    out = core.RUNS / eid
    out.mkdir(parents=True, exist_ok=True)
    for name, (a, b) in importlib.import_module("expos.evaluate").PERIODS.items():
        o = pd.date_range(a, b, freq=step)
        df = pd.DataFrame({"origin": np.repeat(o, 72), "h": np.tile(np.arange(1, 73), len(o))})
        df["pred_kms"] = fn(df, filled)
        df["origin_last_input_utc"] = df.pop("origin").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.rename(columns={"h": "horizon_hours"}).to_parquet(out / f"predictions_{name}.parquet")


def persistence(df, filled):
    return filled.reindex(pd.DatetimeIndex(df["origin"])).to_numpy()


def better(df, filled):  # half-way to truth: strictly better than persistence wherever they differ
    t = filled.reindex(pd.DatetimeIndex(df["origin"] + pd.to_timedelta(df["h"], unit="h"))).to_numpy()
    return 0.5 * persistence(df, filled) + 0.5 * t


def test_chain_detects_tampering(expos):
    core, _, _ = expos
    core.add_direction("D2", "x", "y")
    assert core.verify() == 3
    lines = core.GRAPH.read_text().splitlines()
    lines[1] = lines[1].replace('"test"', '"edited"')
    core.GRAPH.write_text("\n".join(lines) + "\n")
    with pytest.raises(core.ExposError, match="chain broken"):
        core.verify()


def test_register_flags_official_truth_access(expos):
    core, _, _ = expos
    e = core.register("D1", "w1", "h", "python -c \"open('/x/truth.parquet')\"")
    assert e["flags"] == ["forbidden:truth.parquet"]
    assert "integrity flags" in core.eligible(core.state()["experiments"][e["id"]] | {"evals": [{}]})[0]


def test_run_records_outputs_and_review_rules(expos):
    core, _, _ = expos
    e = core.register("D1", "w1", "h", 'echo hi > "$EXPOS_OUT/out.json"')
    r = core.run(e["id"], timeout=30)
    assert r["rc"] == 0 and "out.json" in r["outputs"]
    with pytest.raises(core.ExposError, match="evaluate before review"):
        core.review(e["id"], "critic-1", "accept", "n")
    with pytest.raises(core.ExposError, match="owner cannot review"):
        core.review(e["id"], "w1", "accept", "n")


def test_gate_freeze_and_official_once(expos):
    core, ev, _ = expos
    ref = core.register("D1", "w0", "persistence", "true", kind="reference")["id"]
    _write_preds(core, ref, persistence)
    ev.evaluate(ref)
    cand = core.register("D1", "w1", "better", "true", reference=ref)["id"]
    _write_preds(core, cand, better)
    (core.RUNS / cand / "poison.json").write_text(json.dumps({"passed": True}))
    g = ev.evaluate(cand)["gate"]
    assert g["n_improved"] == 3 and g["mean_relative"] < -0.02
    worse = core.register("D1", "w2", "worse", "true", reference=ref)["id"]
    _write_preds(core, worse, lambda df, f: persistence(df, f) + 80.0)
    assert not ev.evaluate(worse)["gate"]["pass"]
    if not g["pass"]:  # F1 may be undefined on synthetic data; then freeze must refuse
        with pytest.raises(core.ExposError, match="gate"):
            core.freeze(cand)
        return
    with pytest.raises(core.ExposError, match="no accepting review"):
        core.freeze(cand)
    core.review(cand, "critic-1", "accept", "ok")
    with pytest.raises(core.ExposError, match="predictions_official"):
        core.freeze(cand)
    (core.RUNS / cand / "predictions_official.parquet").write_bytes(
        (core.RUNS / cand / "predictions_dev.parquet").read_bytes())
    core.freeze(cand)
    with pytest.raises(core.ExposError, match="already frozen"):
        core.freeze(cand)
    core.append("official", id=cand, results={}, gates={}, achieved=False)
    with pytest.raises(core.ExposError, match="exactly once"):
        ev.official_score()


def test_experience_add_and_query(expos):
    _, _, exp = expos
    exp.add("store/headroom-audit/report.md", "72h lead", "level oracle -50%", "predict level", ["mechanism"], "a")
    exp.add("store/x.md", "DONKI ENLIL", "redundant with plasma/CH", "do not recollect", ["data", "negative"], "a")
    with pytest.raises(Exception):
        exp.add("r", "a", "f", "i", ["nonsense"], "a")
    assert exp.query("ENLIL 중복", k=1)[0]["record"] == "store/x.md"
    assert len(exp.query(tags=["negative"])) == 1


def test_protect_rebaselines_and_gpu_order(expos, monkeypatch):
    core, _, _ = expos
    (core.LB / "expos" / "core.py").exists()
    monkeypatch.setattr(core, "protected_hashes", lambda: {"expos/core.py": "changed"})
    with pytest.raises(core.ExposError, match="reason"):
        core.protect("")
    core.protect("test")
    assert core.state()["init"]["protected"] == {"expos/core.py": "changed"}

    class R:
        stdout = "0, 90, 9000\n1, 5, 100\n"
    monkeypatch.setattr(core.subprocess, "run", lambda *a, **k: R())
    assert core._gpu_order() == [1, 0]
    assert core._gpu_order(prefer=0) == [0, 1]


def test_run_background_and_start_event(expos):
    core, _, _ = expos
    e = core.register("D1", "w1", "h", 'sleep 1; echo ok > "$EXPOS_OUT/x.json"')["id"]
    r = core.run_background(e, timeout=30)
    assert r["pid"] > 0
    import time
    for _ in range(60):
        if core.state()["experiments"][e]["runs"]:
            break
        time.sleep(0.5)
    x = core.state()["experiments"][e]
    assert x["runs"] and x["runs"][-1]["rc"] == 0 and "running" not in x


def test_pareto_front_and_niche_elites(expos):
    core, _, _ = expos
    from expos import archive
    s = {"directions": {"D0": {"parent": None}, "D1": {"parent": "D0"}, "D5": {"parent": "D1"}, "D9": {"parent": "D0"}},
         "experiments": {}}
    def exp(i, d, mr, f1, kind="full", ref="E0003", flags=()):
        s["experiments"][i] = {"id": i, "direction": d, "kind": kind, "flags": list(flags),
                               "evals": [{"gate": {"reference": ref, "mean_relative": mr, "pooled_f1": f1}}]}
    exp("A", "D5", -0.02, 0.30)   # MSE elite
    exp("B", "D9", +0.01, 0.42)   # F1 elite
    exp("C", "D1", +0.02, 0.30)   # dominated by A
    exp("D", "D1", -0.05, 0.50, ref="E0001")  # other reference: ignored
    exp("E", "D9", -0.03, 0.45, flags=["forbidden:x"])  # flagged: ignored
    p = archive.pareto(s)
    assert [q["id"] for q in p["front"]] == ["A", "B"]
    assert p["niches"]["D1/full"]["mse_elite"]["id"] == "A" and p["niches"]["D1/full"]["count"] == 2
    assert p["niches"]["D9/full"]["f1_elite"]["id"] == "B"


def test_stale_running_detected(expos):
    core, _, _ = expos
    e = core.register("D1", "w1", "h", "true")["id"]
    core.append("start", id=e, pid=999999, gpu=None)  # no such process
    assert core.state()["experiments"][e]["running"]["stale"] is True


def test_axes_explicit_tag_and_keywords(expos):
    from expos import axes
    x = {"hypothesis": "[axes: loss=huber+quantile, sel=permutation] try robust loss", "owner": "w"}
    c = axes.classify(x)
    assert c == {"loss": {"huber", "quantile"}, "sel": {"permutation"}}
    y = {"hypothesis": "average five seeds then apply Box-Cox mapping", "owner": "w"}
    cy = axes.classify(y)
    assert "seed-mean" in cy["ens"] and "distribution-map" in cy["post"]
    s = {"experiments": {f"E{i}": {"id": f"E{i}", "hypothesis": "plain xgb run", "owner": "w"} for i in range(5)}}
    r = axes.report(s=s)
    assert set(r["locked_axes"]) == set(axes.AXES) and r["n"] == 5


def test_rocv_gate(expos, monkeypatch):
    core, ev, _ = expos
    from expos import rocv
    importlib.reload(rocv)
    filled, _ = ev._ace()
    def write(eid, fn, seeds):
        d = core.RUNS / eid / "rocv"; d.mkdir(parents=True, exist_ok=True)
        for k in range(len(rocv.FOLDS)):
            a, b = rocv.fold_bounds(k)
            o = pd.date_range(a, b, freq="12h")
            df = pd.DataFrame({"origin": np.repeat(o, 72), "h": np.tile(np.arange(1, 73), len(o))})
            for s in seeds:
                df2 = df.copy(); df2["pred_kms"] = fn(df2, filled, s)
                df2["origin_last_input_utc"] = df2.pop("origin").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                df2.rename(columns={"h": "horizon_hours"}).to_parquet(d / f"f{k}_s{s}.parquet")
    ref = core.register("D1", "w0", "ref", "true", kind="reference")["id"]
    write(ref, lambda df, f, s: persistence(df, f), [0, 1, 2])
    good = core.register("D1", "w1", "better", "true")["id"]
    write(good, lambda df, f, s: better(df, f) + s * 0.1, [0, 1, 2])
    g = rocv.evaluate_rocv(good, ref)["gate"]
    assert g["n_folds_improved"] == 8 and g["mean_relative"] < -0.01 and g["seed_se"] < 0.01
    few = core.register("D1", "w2", "two seeds", "true")["id"]
    write(few, lambda df, f, s: better(df, f), [0, 1])
    assert any("seeds" in r for r in rocv.evaluate_rocv(few, ref)["gate"]["reasons"])


def test_rocv_identical_seeds_rejected(expos):
    core, ev, _ = expos
    from expos import rocv
    importlib.reload(rocv)
    filled, _ = ev._ace()
    def write(eid, fn, seeds):
        d = core.RUNS / eid / "rocv"; d.mkdir(parents=True, exist_ok=True)
        for k in range(len(rocv.FOLDS)):
            a, b = rocv.fold_bounds(k)
            o = pd.date_range(a, b, freq="24h")
            df = pd.DataFrame({"origin": np.repeat(o, 72), "h": np.tile(np.arange(1, 73), len(o))})
            for s in seeds:
                df2 = df.copy(); df2["pred_kms"] = fn(df2, filled, s)
                df2["origin_last_input_utc"] = df2.pop("origin").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                df2.rename(columns={"h": "horizon_hours"}).to_parquet(d / f"f{k}_s{s}.parquet")
    ref = core.register("D1", "w0", "ref", "true", kind="reference")["id"]
    write(ref, lambda df, f, s: persistence(df, f) + s, [0, 1, 2])
    same = core.register("D1", "w1", "seed-invariant", "true")["id"]
    write(same, lambda df, f, s: better(df, f), [0, 1, 2])
    g = rocv.evaluate_rocv(same, ref)["gate"]
    assert not g["pass"] and any("identical" in r for r in g["reasons"])


def _write_official(core, ev, eid, fn, seeds):
    folds = pd.read_parquet(core.Path(ev.OFFICIAL["snapshot"]) / "folds.parquet")
    filled, _ = ev._ace()
    o = pd.to_datetime(folds["origin_last_input_utc"], utc=True)
    df = pd.DataFrame({"origin": np.repeat(o.to_numpy(), 72), "h": np.tile(np.arange(1, 73), len(o))})
    out = core.RUNS / eid
    out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        d = df.copy()
        d["origin"] = pd.to_datetime(d["origin"], utc=True)
        d["pred_kms"] = fn(d, filled, s)
        d["origin_last_input_utc"] = d.pop("origin").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        name = "predictions_official.parquet" if s is None else f"predictions_official_s{s}.parquet"
        d.rename(columns={"h": "horizon_hours"}).to_parquet(out / name)


def test_lb_eval_gate(expos):
    core, ev, _ = expos
    from expos import lbeval
    importlib.reload(lbeval)
    core.append("official", id="E0000", results={"champion": {"events": {"f1": 0.0}}}, gates={}, achieved=False)
    base = core.register("D1", "w0", "persistence baseline", "true", kind="reference")["id"]
    _write_official(core, ev, base, lambda d, f, s: persistence(d, f), [None])
    r = lbeval.lb_eval(base, baseline=True)
    assert r["baseline"] and not r["gate"]["pass"] and r["summary"]["n_seeds"] == 1
    assert lbeval.best(core.state())["id"] == base
    good = core.register("D1", "w1", "better", "true")["id"]
    _write_official(core, ev, good, lambda d, f, s: better(d, f) + s * 0.1, [0, 1, 2])
    g = lbeval.lb_eval(good)["gate"]
    assert g["pass"] and g["rel_vs_best"] < 0, g["reasons"]
    assert lbeval.best(core.state())["id"] == good
    few = core.register("D1", "w2", "two seeds", "true")["id"]
    _write_official(core, ev, few, lambda d, f, s: better(d, f) + s * 0.1, [0, 1])
    assert any("seed" in x for x in lbeval.lb_eval(few)["gate"]["reasons"])
    same = core.register("D1", "w3", "seed-invariant", "true")["id"]
    _write_official(core, ev, same, lambda d, f, s: better(d, f), [0, 1, 2])
    assert any("identical" in x for x in lbeval.lb_eval(same)["gate"]["reasons"])


def test_lb_eval_runner_truth_access_still_flagged(expos):
    core, _, _ = expos
    e = core.register("D1", "w1", "h", "python x.py --truth /snap/truth.parquet")
    assert "forbidden:truth.parquet" in e["flags"]


def test_lb_eval_rejects_total_win_that_worsens_high_cell(expos):
    core, ev, _ = expos
    from expos import lbeval
    importlib.reload(lbeval)
    core.append("official", id="E0000", results={"champion": {"events": {"f1": 0.0}}}, gates={}, achieved=False)
    folds = pd.read_parquet(core.Path(ev.OFFICIAL["snapshot"]) / "folds.parquet")
    llh = set(pd.to_datetime(folds.loc[folds["regime"] == "low_low_high", "origin_last_input_utc"], utc=True))
    base = core.register("D1", "w0", "persistence baseline", "true", kind="reference")["id"]
    _write_official(core, ev, base, lambda d, f, s: persistence(d, f), [None])
    lbeval.lb_eval(base, baseline=True)
    def cheat(d, f, s):  # near-truth everywhere except a much worse low_low_high cell
        t = f.reindex(pd.DatetimeIndex(d["origin"] + pd.to_timedelta(d["h"], unit="h"))).to_numpy()
        bad = d["origin"].isin(llh).to_numpy()
        return np.where(bad, persistence(d, f) + 300.0, t + s * 0.1)
    c = core.register("D1", "w1", "cell cheat", "true")["id"]
    _write_official(core, ev, c, cheat, [0, 1, 2])
    r = lbeval.lb_eval(c)
    assert r["summary"]["mse_seedmean"] < lbeval.best(core.state())["mse_seedmean"]
    assert not r["gate"]["pass"] and any("low_low_high" in x for x in r["gate"]["reasons"])


def test_nn_session_rules(expos, tmp_path):
    core, ev, _ = expos
    from expos import lbeval
    importlib.reload(lbeval)
    core.add_direction("N1", "nn track", "r")
    with pytest.raises(core.ExposError, match="register only"):
        core.register("D1", "nn-w1", "h", "true", code=[])
    with pytest.raises(core.ExposError, match="register only"):
        core.register("N1", "worker-1", "h", "true")
    d = tmp_path / "nnx"; d.mkdir()
    (d / "helper.py").write_text("import xgboost as xgb\n")
    (d / "run.py").write_text("import torch\nimport helper\n")
    with pytest.raises(core.ExposError, match="banned"):
        core.register("N1", "nn-w1", "h", "python run.py", code=[str(d / "run.py")])
    (d / "helper.py").write_text("import torch.nn as nn\n")
    eid = core.register("N1", "nn-w1", "h", "python run.py", code=[str(d / "run.py")])["id"]
    core.append("official", id="E0000", results={"champion": {"events": {"f1": 0.0}}}, gates={}, achieved=False)
    _write_official(core, ev, eid, lambda dd, f, s: better(dd, f) + s * 0.1, [0, 1, 2])
    r = lbeval.lb_eval(eid)
    assert not r["gate"]["pass"] and any("screen-only" in x for x in r["gate"]["reasons"])
    for s_ in (0, 1, 2):
        (core.RUNS / eid / f"curves_s{s_}.json").write_text(json.dumps(
            {"epochs": [{"valid_loss": 2.0}, {"valid_loss": 1.0}], "early_stopped": True}))
    assert not any("screen-only" in x for x in lbeval.lb_eval(eid)["gate"]["reasons"])
    with pytest.raises(core.ExposError, match="GPU-hours"):
        core.set_direction("N1", "closed", "looks bad")
    core.set_direction("N1", "closed", "invalid: data pipeline could not load inputs")
