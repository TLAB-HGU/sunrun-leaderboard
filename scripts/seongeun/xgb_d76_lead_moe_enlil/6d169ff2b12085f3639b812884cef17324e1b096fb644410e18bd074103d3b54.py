# MODEL RELEASE (seongeun, 2026-10-10)
# Fitted model: xgb_d76_lead_moe_enlil_seongeun.pkl
# Google Drive: https://drive.google.com/open?id=1anwxW0Fu9dMXLSusn6iZe8MMJCRE0Xv0
# SHA-256: 2ca18e80697cc63398c5f021fedd3aa955e7cc97b6bb61c0a8300fd4128345af
# Standalone inference package: https://drive.google.com/open?id=1n0yjiDE7vBNwqBimtC8T3hlBGLCTQak2
# Access: anyone with the link. See package README for numeric inputs and usage.

"""E0147 per-lead mixture (E0003-line + D33 hard-switch hybrid + D45 level-mean hybrid) plus DONKI WSA-Enlil CME arrival features.

Reproduces ExpOS experiment E0471 (direction D148), seed cell 0: level seeds 0/1/2 and event seed 1. The model inputs add six features
from NASA CCMC DONKI WSA-Enlil runs (estimated Earth shock arrival, Kp, glancing flag), using only runs whose modelCompletionTime is
<= the forecast origin. The DONKI API serves these runs in real time (latest run used: 2026-10-07). Requires this repository's
experiments/ code plus the pinned local inputs listed in CONFIG. CPU only (XGBoost device=cpu).
SUVI availability uses an operational proxy: S3 LastModified <= origin and observation end <= origin - 1h.
Training uses only targets before 2026-05-29; no truth is read after any origin.

usage: .venv-fusion/bin/python scripts/seongeun/xgb_d76_lead_moe_enlil/6d169ff2b12085f3639b812884cef17324e1b096fb644410e18bd074103d3b54.py --output predictions.parquet
"""
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads('{"data":{"expos_experiment":"E0471 (D148, E0147 recipe + live DONKI WSA-Enlil, seed cell 0)","features":"existing_all (speed history, 26-28 d recurrence, ACE plasma/IMF/EPAM, CH) + 17 forecast-time-aligned SUVI longitude strips + 6 DONKI WSA-Enlil columns (next/since Earth arrival, pending count, max Kp, glancing, lead minus next arrival), 111 columns","input_sha256":{"donki_wsa_enlil (sha256 of sorted relpath<TAB>sha256 lines: local headroom-audit/donki <= 2026-05 + store/donki-live 2026-06..2026-10)":"c47f4df13ce32098843d8a3069ee6a193af66c8df5faafc31f476559ed022129","raw_ace_snapshot_years (sha256 of sorted relpath<TAB>sha256 lines)":"2a07ba7c5740d68ae39bb635e8dcd12cd3f29dd7e732daa58cbb132ea991ead5","store/ch-v1/ace.parquet":"700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af","store/ch-v1/ch-hourly.parquet":"96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a","store/suvi-fusion/suvi-hourly-v2.parquet":"33cbe243e552ab794ea9eaad7d858b7c1908942765ce55f0160baf6df764f746"},"recipe":"E0147 per-lead convex mixture of three members (E0003-line, D33 hard-switch hybrid, D45 level-mean hybrid), XGBoost max_depth 6, plus DONKI WSA-Enlil features from runs with modelCompletionTime <= origin; per-lead weights fixed from E0145 dev-period OOF","source_sha256":{"experiments/d141_e0147_ablation/ablate.py":"2e635c647b565506f50aed89d468f4b8ab92d4c26af305d292a9f0478bc554af","experiments/d148_enlil_live/run.py":"32458a4e5d62df1e93fabbecc028a30350eb8c36bf8cca9da4e613cd2a88d38d","experiments/d1_longitude_strips/strips.py":"50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023","experiments/d32_bandexpert/bands.py":"7266d107cad142a7b7350de87107e18b249da7cf17ab0aca3e63f81d739ce95f","experiments/d33_hsplit/hsplit.py":"99c34a39773d4b3770aa8a83deb535476f576f4468ff03d288eea63a7fe85439","experiments/d45_lvlmean/lvlmean.py":"7bba6da06f0d23ed6228a4a9f88e462888a91f709a77a4a2c88e84993b546cd4","experiments/d76_moe/moe.py":"4e0d9b3bf7a113b7ff244d790e5d40895f4f18c5a882af06d1c7904f7c3d106e","experiments/d76_moe/runner.py":"c3444acde0450700563beb70aece9baced4ff5373b410f80409de806532be40e","experiments/expos_runners/existing_all.py":"6343e35dfbb0f449f36cd74fc4ec95853637d9e3ae5ac328ed96a83841d2a73a","experiments/expos_runners/official_members.py":"b5994f20a343602be5724c3b856d35babc90897b5a814c73cdd2130333238faa","experiments/headroom_audit/headroom.py":"2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f","experiments/headroom_audit/info_ablation.py":"85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199"}},"hyperparameters":{"device":"cpu","enlil_rule":"modelCompletionTime <= origin, runs within 7 days, Earth arrival estimates only","event_seed":1,"level_seeds":[0,1,2],"n_jobs":6,"split_h":24,"train_origin_grid":"3h from 2022-10-01","train_target_cutoff_utc":"2026-05-29T00:00:00Z","weights":{"1-6":{"d33_hsplit":0.3899845818930272,"d45_lvlmean":0.3946530834632206,"e0003_line":0.21536233464375223},"25-48":{"d33_hsplit":0.3333333333333333,"d45_lvlmean":0.3333333333333333,"e0003_line":0.3333333333333333},"49-72":{"d33_hsplit":0.3333333333333333,"d45_lvlmean":0.3333333333333333,"e0003_line":0.3333333333333333},"7-24":{"d33_hsplit":0.3436571856823209,"d45_lvlmean":0.3411415011482895,"e0003_line":0.3152013131693896}},"xgboost":{"colsample_bytree":0.7,"learning_rate":0.05,"max_depth":6,"min_child_weight":20,"n_estimators":600,"reg_lambda":5,"subsample":0.8,"tree_method":"hist"}}}')
CONFIG_SHA256 = "6d169ff2b12085f3639b812884cef17324e1b096fb644410e18bd074103d3b54"


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
    lb = ROOT
    files = sorted([str(p.relative_to(lb)) for p in (lb / "store/headroom-audit/donki").glob("WSAEnlilSimulations_*.json") if p.stem[-7:] < "2026-06"]
                   + [str(p.relative_to(lb)) for p in (lb / "store/donki-live").glob("WSAEnlilSimulations_*.json")])
    got = hashlib.sha256("".join(f"{p}\t{digest(lb / p)}\n" for p in files).encode()).hexdigest()
    if got != [v for k, v in CONFIG["data"]["input_sha256"].items() if k.startswith("donki")][0]:
        raise ValueError("DONKI WSA-Enlil input identity mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    validate()
    os.environ["EXPOS_OUT"] = tempfile.mkdtemp(prefix="lb-run-")  # read by the experiment modules at import time
    sys.path.insert(0, str(ROOT / "experiments/d141_e0147_ablation"))
    import numpy as np
    import pandas as pd
    import ablate as A
    sys.path.insert(0, str(ROOT / "experiments/d148_enlil_live"))
    import run as D148
    A.ia.enlil_runs = D148.enlil_runs  # local DONKI <= 2026-05 + store/donki-live (pinned in CONFIG)
    cfg = A.ARMS["enlil"]
    df = A.build_train("3h", None, {})
    cols = A.feature_columns(df, cfg)
    tr = df[df["target_t"] < A.CUTOFF]
    meta, block = A.ST.load_inputs()
    te, _ = A.R.add_strips(A.R.base_grid(A.ORIGINS), meta, block)
    models = A.fit_cell(tr, cols, cfg, 0)
    pred = A.blend(A.member_preds(models, te, cols, cfg), te["h"].to_numpy(), cfg)
    if not np.isfinite(pred).all() or len(pred) != len(A.ORIGINS) * 72:
        raise SystemExit("non-finite or incomplete official predictions; refusing to write")
    pd.DataFrame({"origin_last_input_utc": te["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "horizon_hours": te["h"].astype(int), "pred_kms": pred.astype(float)}).to_parquet(a.output, index=False)
    sample = te[te["origin"].isin(A.ORIGINS[::14][:200])]
    t0 = time.perf_counter()
    for _, g in sample.groupby("origin"):
        A.blend(A.member_preds(models, g, cols, cfg), g["h"].to_numpy(), cfg)
    per_fold = (time.perf_counter() - t0) / sample["origin"].nunique()
    print(json.dumps({"inference_seconds_per_fold": per_fold, "train_rows": int(len(tr)), "origins": len(A.ORIGINS)}))


if __name__ == "__main__":
    main()
