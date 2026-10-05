"""Direct 72h XGBoost: ACE speed/plasma/IMF/EPAM + CH + 26-28d recurrence + 17 forecast-time-aligned SUVI Fe195 ...

Reproduces ExpOS official run E0105 pilot (6h grid) / official run E0107. Requires this repository's experiments/ code plus the pinned local inputs listed in CONFIG
(ACE, CH, raw ACE snapshot, SUVI hourly cache and SUVI metadata stage, DONKI ENLIL JSON) and an NVIDIA GPU (XGBoost device=cuda).
SUVI availability uses an operational proxy: S3 LastModified <= origin and observation end <= origin - 1h.
Training uses only targets before 2026-05-29; no truth is read after any origin.

usage: .venv-fusion/bin/python scripts/seongeun/xgb_d45_level3mean_hardswitch/1374c46b724369aaf859f36b20a222c4abe23c47264caec67b8f1c05b0789c5f.py --output predictions.parquet
"""
import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads('{"data":{"expos_experiment":"E0105 pilot (6h grid) / official run E0107","input_sha256":{"donki_enlil_json":"d7732509f542185aea93ace7acd8a0b762f3512a94a35700588efb5911cfa033","raw_ace_snapshot_years":"695990b036c88073ed4d3f4b0338bc9a522872dba65f520884598e50e0ba7736","store/ch-v1/ace.parquet":"700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af","store/ch-v1/ch-hourly.parquet":"96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a","store/suvi-fusion/suvi-hourly-v2.parquet":"33cbe243e552ab794ea9eaad7d858b7c1908942765ce55f0160baf6df764f746"},"recipe":"E0003 features (existing_all + 17 forecast-time-aligned SUVI Fe195 longitude strips) with a horizon hard switch at h=24: h<=24 from two band experts (1-6, 7-24) averaged over level seeds 0,1,2; h>=25 from one E0003-line head (seed 1)","source_sha256":{"experiments/d1_longitude_strips/runner.py":"a55c59aad77c457ebfd456ea937f92f7d7ac4e93f86e75b31931ed6206d3ae40","experiments/d1_longitude_strips/strips.py":"50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023","experiments/d32_bandexpert/bands.py":"7266d107cad142a7b7350de87107e18b249da7cf17ab0aca3e63f81d739ce95f","experiments/d33_hsplit/hsplit.py":"99c34a39773d4b3770aa8a83deb535476f576f4468ff03d288eea63a7fe85439","experiments/d45_lvlmean/lvlmean.py":"7bba6da06f0d23ed6228a4a9f88e462888a91f709a77a4a2c88e84993b546cd4","experiments/d45_lvlmean/runner.py":"8617b444db81d62527f69ffd840971a4b7e969ebbf97e3cd9dbe9133073bd95b","experiments/expos_runners/d45_official.py":"34276d67495aed9247ace6bc15d023f859a3e7c8915bb085d81affd075dd0ec7","experiments/expos_runners/existing_all.py":"6343e35dfbb0f449f36cd74fc4ec95853637d9e3ae5ac328ed96a83841d2a73a","experiments/headroom_audit/headroom.py":"2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f","experiments/headroom_audit/info_ablation.py":"85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199"}},"hyperparameters":{"device":"cuda","event_seed":1,"fit_n_jobs":4,"level_seeds":[0,1,2],"split_h":24,"train_origin_grid":"6h from 2022-10-01","train_target_cutoff_utc":"2026-05-29T00:00:00Z","xgboost_base":{"colsample_bytree":0.7,"learning_rate":0.05,"max_depth":6,"min_child_weight":20,"n_estimators":600,"n_jobs":8,"random_state":0,"reg_lambda":5,"subsample":0.8,"tree_method":"hist"}}}')
CONFIG_SHA256 = "1374c46b724369aaf859f36b20a222c4abe23c47264caec67b8f1c05b0789c5f"


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
    import d45_official as official
    official.main()
    shutil.copy(Path(out) / "predictions_official.parquet", a.output)
    print(json.dumps(json.loads((Path(out) / "timing.json").read_text())))


if __name__ == "__main__":
    main()
