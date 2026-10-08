"""E0408-structure per-lead mixture with an analog residual corrector (standardised scenario distance) plus DONKI WSA-Enlil CME features.

Reproduces ExpOS experiment E0472 (direction D148), seed cell 0. It is E0466 (the E0421 recipe with each scenario subspace divided by its
pre-2026 pool standard deviation) whose base XGBoost models also see six features from NASA CCMC DONKI WSA-Enlil runs (estimated Earth
shock arrival, Kp, glancing flag), using only runs whose modelCompletionTime is <= the forecast origin. Requires this repository's
experiments/ code plus the pinned local inputs listed in CONFIG. CPU only (XGBoost device=cpu). The base fits, out-of-fold fits,
scenario replay and blend weight are computed inside this run from targets before 2026-05-29; no stored prediction file is read and no
truth is read after any origin. SUVI availability uses an operational proxy: S3 LastModified <= origin and observation end <= origin - 1h.

usage: .venv-fusion/bin/python scripts/seongeun/xgb_d76_lead_moe_resid_dv_enlil/b8ae672fdb6ff9a6dc85bc5c3f2d8b2acbc9207c98126d39ad1913f24581962c.py --output predictions.parquet
"""
import argparse, hashlib, json, os, shutil, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads('{"data":{"enlil_rule":"NASA CCMC DONKI WSA-Enlil runs with modelCompletionTime <= origin, within 7 days, Earth arrival estimates only; the API serves runs in real time","expos_experiment":"E0472 (D148: E0466 = D147 lean arm s3_std, plus live DONKI WSA-Enlil features), seed cell 0","features":"existing_all (speed history, 26-28 d recurrence, ACE plasma/IMF/EPAM, CH) + 17 forecast-time-aligned SUVI longitude strips + 6 DONKI WSA-Enlil columns (next/since Earth arrival, pending count, max Kp, glancing, lead minus next arrival), 111 columns","input_sha256":{"donki_wsa_enlil (sha256 of sorted relpath<TAB>sha256 lines: local headroom-audit/donki <= 2026-05 + store/donki-live 2026-06..2026-10)":"c47f4df13ce32098843d8a3069ee6a193af66c8df5faafc31f476559ed022129","raw_ace_snapshot_years (sha256 of sorted relpath<TAB>sha256 lines)":"2a07ba7c5740d68ae39bb635e8dcd12cd3f29dd7e732daa58cbb132ea991ead5","store/ch-v1/ace.parquet":"700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af","store/ch-v1/ch-hourly.parquet":"96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a","store/suvi-fusion/suvi-hourly-v2.parquet":"33cbe243e552ab794ea9eaad7d858b7c1908942765ce55f0160baf6df764f746"},"oof":"base refit on targets before 2026-03-01 scores OOF origins 2026-03-04..2026-05-28 (3h grid); blend weight w = grid argmin of OOF MSE per seed; the official window never enters any fitted choice","recipe":"E0408-structure base (per-lead convex mixture of E0003-line, D33 hard-switch hybrid and D45 level-mean hybrid, XGBoost depth 3, 900 trees, eta 0.03) plus an analog residual corrector: dv = y - yhat_base is replayed from the K most similar past scenarios and added as base + w*dv","scenario":"96-dim vector: past 12 h observed speed, base error (past minus hindcast) over 12 h, base predicted 72 h path; each subspace divided by its pre-2026 pool standard deviation, then equal weights; neighbours within one Carrington rotation (27.27 d) of the query excluded; pool uses targets before the cutoff only","source_sha256":{"experiments/d143_axisA_analog/a2b/resid_lib_a2b.py":"f3fbb52216280ac1a158699a60f74f6bc8541a01c9d207f61b163de892ba92c4","experiments/d143_axisA_analog/a2b/run_official_a2b.py":"63abb966de263697e0d6bdef51dfd2cf717d8565bb24a222a03b046e6f2f4575","experiments/d143_axisA_analog/a2b/spec_a2b.py":"7d6a131138b6f9e17348e269d695a3ffae37428e2398ca0dd18eb65af6b710d2","experiments/d147_e0421_lean/lean.py":"8f2ba0b9c9dba52879683ff770810e5272a216e9eba67422a08b6bf5f8c5972b","experiments/d148_enlil_live/lean_enlil.py":"20ad7d4ee78ae762b6781aea6fd8e88169c95fd67770a7680ccea49f72e5be96","experiments/d1_longitude_strips/strips.py":"50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023","experiments/headroom_audit/headroom.py":"2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f","experiments/headroom_audit/info_ablation.py":"85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199"}},"hyperparameters":{"blend_grid":[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],"blend_window":"W2 (2026-03-04..2026-05-28)","carrington_hours":654.48,"device":"cpu","dist":"std_equal","k":15,"n_jobs":6,"oof_windows_scored":["2026-01","2026-02","2026-03-04..2026-05-28"],"scenario_dim":96,"seed_cell":0,"split_h":24,"train_origin_grid":"3h from 2022-10-01","train_target_cutoff_utc":"2026-05-29T00:00:00Z","weights":{"1-6":{"d33_hsplit":0.3899845818930272,"d45_lvlmean":0.3946530834632206,"e0003_line":0.21536233464375223},"25-48":{"d33_hsplit":0.3333333333333333,"d45_lvlmean":0.3333333333333333,"e0003_line":0.3333333333333333},"49-72":{"d33_hsplit":0.3333333333333333,"d45_lvlmean":0.3333333333333333,"e0003_line":0.3333333333333333},"7-24":{"d33_hsplit":0.3436571856823209,"d45_lvlmean":0.3411415011482895,"e0003_line":0.3152013131693896}},"xgboost":{"colsample_bytree":0.7,"learning_rate":0.03,"max_depth":3,"min_child_weight":20,"n_estimators":900,"reg_lambda":5,"subsample":0.8,"tree_method":"hist"}}}')
CONFIG_SHA256 = "b8ae672fdb6ff9a6dc85bc5c3f2d8b2acbc9207c98126d39ad1913f24581962c"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def validate():
    canonical = json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != CONFIG_SHA256 or Path(__file__).stem != CONFIG_SHA256:
        raise ValueError("registered script/config identity mismatch")
    for rel, expected in CONFIG["data"]["source_sha256"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError("source identity mismatch: " + rel)
    files = sorted([str(p.relative_to(ROOT)) for p in (ROOT / "store/headroom-audit/donki").glob("WSAEnlilSimulations_*.json") if p.stem[-7:] < "2026-06"]
                   + [str(p.relative_to(ROOT)) for p in (ROOT / "store/donki-live").glob("WSAEnlilSimulations_*.json")])
    got = hashlib.sha256("".join(f"{p}\t{digest(ROOT / p)}\n" for p in files).encode()).hexdigest()
    if got != [v for k, v in CONFIG["data"]["input_sha256"].items() if k.startswith("donki")][0]:
        raise ValueError("DONKI WSA-Enlil input identity mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    validate()
    out = tempfile.mkdtemp(prefix="lb-run-")
    os.environ["EXPOS_OUT"] = out  # read by the experiment modules at import time
    sys.path.insert(0, str(ROOT / "experiments/d148_enlil_live"))
    import lean_enlil as LE
    L = LE.L
    L.ia.enlil_runs, L.A.base_grid_for, L.feature_columns = LE.enlil_runs, LE.base_grid_for, LE.feature_columns
    L.SP.SEEDS = (0,)  # seed cell 0 only
    n = int(L.SP.N_OFFICIAL_ORIGINS)
    spent = [0.0]

    def timed(fn, is_official):
        def wrap(*args, **kw):
            t0 = time.perf_counter()
            r = fn(*args, **kw)
            if is_official(args):
                spent[0] += time.perf_counter() - t0
            return r
        return wrap

    # inference = official base paths (query + 12 h hindcast), residual replay, final base prediction; fitting/loading excluded
    L.paths = timed(L.paths, lambda a: len(a[2]) == n)
    L.residual_multi = timed(L.residual_multi, lambda a: len(a[0]) == n)
    L.predict = timed(L.predict, lambda a: len(a[2]) == n * 72)
    sys.argv = [sys.argv[0], "--arm", "s3_std"]
    L.main()
    shutil.copy(Path(out) / "predictions_official_s0.parquet", a.output)
    print(json.dumps({"inference_seconds_per_fold": spent[0] / n}))


if __name__ == "__main__":
    main()
