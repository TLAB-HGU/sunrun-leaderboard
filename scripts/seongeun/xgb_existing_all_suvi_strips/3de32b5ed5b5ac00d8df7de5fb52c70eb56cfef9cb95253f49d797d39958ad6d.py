"""existing_all direct XGBoost + forecast-time-aligned SUVI Fe195 longitude strips: official 72h forecasts.

Reproduces ExpOS experiment E0040 (recipe of E0003, which passed the preofficial three-period gate and review).
Requires this repository's experiments/ code plus the pinned local inputs listed in CONFIG (ACE, CH, raw ACE snapshot,
SUVI hourly cache, DONKI ENLIL JSON, SUVI metadata stage). SUVI availability uses an operational proxy:
S3 LastModified <= origin and observation end <= origin - 1h. Reads no truth after any origin.

usage: .venv-fusion/bin/python scripts/seongeun/xgb_existing_all_suvi_strips/3de32b5ed5b5ac00d8df7de5fb52c70eb56cfef9cb95253f49d797d39958ad6d.py --output predictions.parquet
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads('{"data":{"expos_experiment":"E0003 (3-period gate pass, reviewed) / official run E0040","input_sha256":{"donki_enlil_json":"d7732509f542185aea93ace7acd8a0b762f3512a94a35700588efb5911cfa033","raw_ace_snapshot_years":"695990b036c88073ed4d3f4b0338bc9a522872dba65f520884598e50e0ba7736","store/ch-v1/ace.parquet":"700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af","store/ch-v1/ch-hourly.parquet":"96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a","store/suvi-fusion/suvi-hourly-v2.parquet":"33cbe243e552ab794ea9eaad7d858b7c1908942765ce55f0160baf6df764f746"},"inputs":["ACE speed history + 26/27/28-day recurrence","ACE plasma/IMF/EPAM (raw snapshot)","SUVI CH hourly features","SUVI longitude strips (proxy availability: S3 LastModified<=origin, obs_end<=origin-1h)"],"recipe":"existing_all direct XGBoost + 17 forecast-time-aligned SUVI Fe195 longitude-strip features (5 speed assumptions 350-750 km/s)","source_sha256":{"experiments/d1_longitude_strips/runner.py":"a55c59aad77c457ebfd456ea937f92f7d7ac4e93f86e75b31931ed6206d3ae40","experiments/d1_longitude_strips/strips.py":"50224ed06ae71bc04e87581901097dcdd9d68dcf91b4a42222bdd9cc7d7a1023","experiments/expos_runners/d1_strips_official.py":"0ae97e1544268024b90f12916d21d2bea9d009e28ea429d9d709cd35e3abafdd","experiments/expos_runners/existing_all.py":"6343e35dfbb0f449f36cd74fc4ec95853637d9e3ae5ac328ed96a83841d2a73a","experiments/headroom_audit/headroom.py":"2a553f6dc02dc80a79c8ae9c67632b3a4f588f7385e19ec1b7392a283abbd19f","experiments/headroom_audit/info_ablation.py":"85cf734480900f135ed6a052f44564012ab8faccfcfdacce9089e53dd972e199"},"train_origin_grid":"3h from 2022-10-01","train_target_cutoff_utc":"2026-05-29T00:00:00Z"},"hyperparameters":{"xgboost":{"colsample_bytree":0.7,"learning_rate":0.05,"max_depth":6,"min_child_weight":20,"n_estimators":600,"n_jobs":8,"random_state":0,"reg_lambda":5,"subsample":0.8,"tree_method":"hist"}}}')
CONFIG_SHA256 = "3de32b5ed5b5ac00d8df7de5fb52c70eb56cfef9cb95253f49d797d39958ad6d"


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
    sys.path.insert(0, str(ROOT / "experiments/d1_longitude_strips"))
    import pandas as pd
    import xgboost as xgb
    import runner as d1
    if d1.ia.PARAMS != CONFIG["hyperparameters"]["xgboost"]:
        raise ValueError("xgboost params differ from CONFIG")
    df = d1.ia.build()
    meta, block = d1.ST.load_inputs()
    df, _ = d1.add_strips(df, meta, block)
    cols = d1.ia.arms(df.columns)["existing_all"] + d1.ST.strip_columns()
    tr = df[df["target_t"] < pd.Timestamp(CONFIG["data"]["train_target_cutoff_utc"])]
    model = xgb.XGBRegressor(**d1.ia.PARAMS).fit(tr[cols], tr["y"])
    origins = pd.date_range("2026-05-31 23:00", "2026-09-26 23:00", freq="h", tz="UTC")
    te, _ = d1.add_strips(d1.base_grid(origins), meta, block)
    pred = model.predict(te[cols])
    # inference timing: one 72h forecast per call (features prepared beforehand), mean over 200 origins
    sample = te[te["origin"].isin(origins[::14][:200])]
    t0 = time.perf_counter()
    for _, g in sample.groupby("origin"):
        model.predict(g[cols])
    per_fold = (time.perf_counter() - t0) / sample["origin"].nunique()
    pd.DataFrame({"origin_last_input_utc": te["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "horizon_hours": te["h"].astype(int), "pred_kms": pred.astype(float)}).to_parquet(a.output, index=False)
    print(json.dumps({"rows": int(len(te)), "origins": len(origins), "inference_seconds_per_fold": per_fold}))


if __name__ == "__main__":
    main()
