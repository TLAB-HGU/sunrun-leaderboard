# MODEL RELEASE (seongeun, 2026-10-10)
# Fitted model: xgb_d33_hardswitch_rocv_seongeun.pkl
# Google Drive: https://drive.google.com/open?id=1POAalgzvaxM5T1DDgy5M65pdGYvMEqI0
# SHA-256: d31a72c801069744b824db2aee11ab6eec7428c0f0eb38551d034a08d72a0d22
# Standalone inference package: https://drive.google.com/open?id=1n0yjiDE7vBNwqBimtC8T3hlBGLCTQak2
# Access: anyone with the link. See package README for numeric inputs and usage.

"""Direct 72h XGBoost hard switch (D33/E0102 line): two short band experts (1-6h, 7-24h) for h<=24 and one long ...

Reproduces ExpOS experiment E0207 (rolling-origin 8 folds x 3 seeds) / official run E0221. Requires this repository's experiments/ code plus the pinned local inputs listed in CONFIG
(ACE, CH, raw ACE snapshot, SUVI hourly cache and SUVI metadata stage, DONKI ENLIL JSON) and an NVIDIA GPU (XGBoost device=cuda).
SUVI availability uses an operational proxy: S3 LastModified <= origin and observation end <= origin - 1h.
Training uses only targets before 2026-05-29; no truth is read after any origin.

usage: .venv-fusion/bin/python scripts/seongeun/xgb_d33_hardswitch_rocv/18121b65b2559acf339851a188d0481731e020d85555b12865c11c2209d7a14e.py --output predictions.parquet
"""
import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads('{"data":{"expos_experiment":"E0207 (rolling-origin 8 folds x 3 seeds) / official run E0221","input_sha256":{"donki_enlil_json":"d7732509f542185aea93ace7acd8a0b762f3512a94a35700588efb5911cfa033","raw_ace_snapshot_years":"695990b036c88073ed4d3f4b0338bc9a522872dba65f520884598e50e0ba7736","store/ch-v1/ace.parquet":"700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af","store/ch-v1/ch-hourly.parquet":"96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a","store/suvi-fusion/suvi-hourly-v2.parquet":"33cbe243e552ab794ea9eaad7d858b7c1908942765ce55f0160baf6df764f746"},"recipe":"Hard switch at h=24: short 1-6h and 7-24h band experts plus one long E0003-line head, all on E0003 features (existing_all + 17 SUVI longitude strips); seed cell 0","source_sha256":{"experiments/d133_t1rocv/rocv_runner.py":"d555c76facc76fc6ca71f43ae301dba777e6bd8f03a0573c01f37eb1a1a45b04","experiments/d1_longitude_strips/runner.py":"a55c59aad77c457ebfd456ea937f92f7d7ac4e93f86e75b31931ed6206d3ae40","experiments/d1_longitude_strips/strips.py":"50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023","experiments/d32_bandexpert/bands.py":"7266d107cad142a7b7350de87107e18b249da7cf17ab0aca3e63f81d739ce95f","experiments/d33_hsplit/hsplit.py":"99c34a39773d4b3770aa8a83deb535476f576f4468ff03d288eea63a7fe85439","experiments/expos_runners/d207_official.py":"0323ad169e4003cc797a99b591b162794e3742b7636062bdc48c40573eba4632","experiments/expos_runners/existing_all.py":"6343e35dfbb0f449f36cd74fc4ec95853637d9e3ae5ac328ed96a83841d2a73a","experiments/expos_runners/hsplit_official.py":"1727e2f8ae96bb6aaae37f1f1659e621c60835229429ced80f3a07160c4c3d4b","experiments/headroom_audit/headroom.py":"2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f","experiments/headroom_audit/info_ablation.py":"85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199","expos/__init__.py":"586127c1d1b4976214f1839da8ef4a036eca443e5f2c824a11d3d6f19210c941","expos/core.py":"a2f8089d77562298edeb77920dee9039912955f668c3fb5d5761a7afcca89286","expos/evaluate.py":"28d6a06448d5c4ff9f50406446fbb0cf53edb04175a4b2a556f42acfb0937cf5","expos/rocv.py":"456bc11d0b721f6f4a13421dacf5eb8f9972120e096c9adcb23064a25a4d09db"}},"hyperparameters":{"device":"cuda","seeds":{"event":0,"level":0},"split_h":24,"train_origin_grid":"3h from 2022-10-01","train_target_cutoff_utc":"2026-05-29T00:00:00Z","xgboost_base":{"colsample_bytree":0.7,"learning_rate":0.05,"max_depth":6,"min_child_weight":20,"n_estimators":600,"n_jobs":8,"random_state":0,"reg_lambda":5,"subsample":0.8,"tree_method":"hist"}}}')
CONFIG_SHA256 = "18121b65b2559acf339851a188d0481731e020d85555b12865c11c2209d7a14e"


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
    sys.path.insert(0, str(ROOT / "experiments/expos_runners"))
    import d207_official as official
    official.main()
    shutil.copy(Path(out) / "predictions_official.parquet", a.output)
    print(json.dumps(json.loads((Path(out) / "timing.json").read_text())))


if __name__ == "__main__":
    main()
