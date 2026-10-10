# MODEL RELEASE (seongeun, 2026-10-10)
# Fitted model: xgb_d76_lead_moe_resid_dv_seongeun.pkl
# Google Drive: https://drive.google.com/open?id=1kv6pQBohWk-E2nj6EWEzgADsv19CNPCW
# SHA-256: 4053c88093269179bf31fa8b15060241ceaef4e7ddd9e08a4030f293f1effdff
# Standalone inference package: https://drive.google.com/open?id=1n0yjiDE7vBNwqBimtC8T3hlBGLCTQak2
# Access: anyone with the link. See package README for numeric inputs and usage.

"""E0408-structure per-lead mixture (depth 3) with an analog residual corrector (dv = y - yhat_base).

Reproduces ExpOS experiment E0421 (direction D143, axis A2b), seed cell 0. Requires this repository's experiments/ code plus the
pinned local inputs listed in CONFIG. CPU only (XGBoost device=cpu). The base fits, the out-of-fold fits, the scenario replay and the
blend weight are all computed inside this run from targets before 2026-05-29; no stored prediction file is read and no truth is read
after any origin. SUVI availability uses an operational proxy: S3 LastModified <= origin and observation end <= origin - 1h.

usage: .venv-fusion/bin/python scripts/seongeun/xgb_d76_lead_moe_resid_dv/285d8f41752f9ad9b0c11a867a26546ca3887d029972af3bf438b18314436609.py --output predictions.parquet
"""
import argparse, contextlib, hashlib, io, json, os, re, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads('{"data":{"expos_experiment":"E0421 (D143 axis A2b), seed cell 0","features":"existing_all (speed history, 26-28 d recurrence, ACE plasma/IMF/EPAM, CH) + 17 forecast-time-aligned SUVI longitude strips, 105 columns","input_sha256":{"donki_enlil_json (read by the shared builder; ENLIL columns are not model inputs)":"c092fff5d435333675b3ea62f966810b87d05f7b24651f601e9326d9d255eeba","raw_ace_snapshot_years (sha256 of sorted relpath<TAB>sha256 lines)":"2a07ba7c5740d68ae39bb635e8dcd12cd3f29dd7e732daa58cbb132ea991ead5","store/ch-v1/ace.parquet":"700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af","store/ch-v1/ch-hourly.parquet":"96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a","store/suvi-fusion/suvi-hourly-v2.parquet":"33cbe243e552ab794ea9eaad7d858b7c1908942765ce55f0160baf6df764f746"},"oof":"base refit on targets before 2026-03-01 scores OOF origins 2026-03-04..2026-05-28 (3h grid); blend weight w = grid argmin of OOF MSE per seed; the official window never enters any fitted choice","recipe":"E0408-structure base (per-lead convex mixture of E0003-line, D33 hard-switch hybrid and D45 level-mean hybrid, XGBoost depth 3, 900 trees, eta 0.03) plus an analog residual corrector: dv = y - yhat_base is replayed from the K most similar past scenarios and added as base + w*dv","scenario":"96-dim vector: past 12 h observed speed, base error (past minus hindcast) over 12 h, base predicted 72 h path; equal subspace weights; neighbours within one Carrington rotation (27.27 d) of the query excluded; pool uses targets before the cutoff only","source_sha256":{"experiments/d143_axisA_analog/a2b/resid_lib_a2b.py":"f3fbb52216280ac1a158699a60f74f6bc8541a01c9d207f61b163de892ba92c4","experiments/d143_axisA_analog/a2b/run_official_a2b.py":"63abb966de263697e0d6bdef51dfd2cf717d8565bb24a222a03b046e6f2f4575","experiments/d143_axisA_analog/a2b/spec_a2b.py":"7d6a131138b6f9e17348e269d695a3ffae37428e2398ca0dd18eb65af6b710d2","experiments/d1_longitude_strips/strips.py":"50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023","experiments/headroom_audit/headroom.py":"2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f","experiments/headroom_audit/info_ablation.py":"85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199"}},"hyperparameters":{"blend_grid":[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],"carrington_hours":654.48,"device":"cpu","k":15,"n_jobs":6,"scenario_dim":96,"seed_cell":0,"split_h":24,"train_origin_grid":"3h from 2022-10-01","train_target_cutoff_utc":"2026-05-29T00:00:00Z","weights":{"1-6":{"d33_hsplit":0.3899845818930272,"d45_lvlmean":0.3946530834632206,"e0003_line":0.21536233464375223},"25-48":{"d33_hsplit":0.3333333333333333,"d45_lvlmean":0.3333333333333333,"e0003_line":0.3333333333333333},"49-72":{"d33_hsplit":0.3333333333333333,"d45_lvlmean":0.3333333333333333,"e0003_line":0.3333333333333333},"7-24":{"d33_hsplit":0.3436571856823209,"d45_lvlmean":0.3411415011482895,"e0003_line":0.3152013131693896}},"xgboost":{"colsample_bytree":0.7,"learning_rate":0.03,"max_depth":3,"min_child_weight":20,"n_estimators":900,"reg_lambda":5,"subsample":0.8,"tree_method":"hist"}}}')
CONFIG_SHA256 = "285d8f41752f9ad9b0c11a867a26546ca3887d029972af3bf438b18314436609"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    validate()
    out = tempfile.mkdtemp(prefix="lb-run-")
    os.environ["EXPOS_OUT"] = out  # read by the experiment modules at import time
    sys.path.insert(0, str(ROOT / "experiments/d143_axisA_analog/a2b"))
    import run_official_a2b as A
    A.SP.SEEDS = (0,)  # seed cell 0 only
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        A.main()
    text = log.getvalue()
    print(text)
    shutil.copy(Path(out) / "predictions_official_s0.parquet", a.output)
    # inference seconds: base-path scoring plus residual replay for the official origins, excluding loading, fitting and writing
    paths = re.search(r"seed 0: paths .*?seconds=([0-9.]+)", text)
    resid = re.search(r"seed 0: residual .*?seconds=([0-9.]+)", text)
    n = CONFIG["hyperparameters"].get("official_origins", 2833)
    if paths and resid:
        print(json.dumps({"inference_seconds_per_fold": (float(paths.group(1)) + float(resid.group(1))) / n}))


if __name__ == "__main__":
    main()
