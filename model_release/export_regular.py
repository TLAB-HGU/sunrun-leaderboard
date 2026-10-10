"""Export frozen submitted recipes, retaining fitted models and parity evidence.

Run each recipe in a fresh process to isolate legacy module-global caches.
Only the GPU1 shared lease is used for recipes originally trained with CUDA.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB))

RECIPES = {
    "E0040": ("xgb_existing_all_suvi_strips", "direct", "3h"),
    "E0221": ("xgb_d33_hardswitch_rocv", "hsplit", "3h"),
    "E0107": ("xgb_d45_level3mean_hardswitch", "lvlmean", "6h"),
    "E0146": ("xgb_d71_oof_stack", "stack", "3h"),
    "E0147": ("xgb_d76_lead_moe", "moe", "3h"),
    "E0368": ("xgb_d76_lead_moe_depth4", "ablation", "3h"),
    "E0471": ("xgb_d76_lead_moe_enlil", "ablation", "3h"),
}


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_models(value, *, preserve_cuda_arithmetic=False):
    """Keep serialized fitted boosters usable without a CUDA installation."""
    if isinstance(value, dict):
        for item in value.values():
            cpu_models(item, preserve_cuda_arithmetic=preserve_cuda_arithmetic)
    elif isinstance(value, (tuple, list)):
        for item in value:
            cpu_models(item, preserve_cuda_arithmetic=preserve_cuda_arithmetic)
    elif hasattr(value, "get_booster"):
        if preserve_cuda_arithmetic:
            booster = value.get_booster()
            base = booster.attr("release_gpu_base_score")
            if base is None:
                base = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
            value.set_params(device="cpu", n_jobs=4, base_score=float(0))
            booster.set_attr(release_gpu_base_score=base)
        else:
            value.set_params(device="cpu", n_jobs=4)


def restore_cuda_models(value):
    if isinstance(value, dict):
        for item in value.values():
            restore_cuda_models(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            restore_cuda_models(item)
    elif hasattr(value, "get_booster"):
        base = value.get_booster().attr("release_gpu_base_score")
        value.set_params(device="cuda", n_jobs=4, **({"base_score": float(base)} if base is not None else {}))


@contextlib.contextmanager
def gpu1_lease():
    path = LB / "store/ch-breakthrough-v2/gpu-leases/gpu1.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        deadline = time.monotonic() + 14400
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                status = subprocess.check_output([
                    "nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True)
                memory, utilization = map(int, status.strip().split(","))
                if memory < 100 and utilization == 0:
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU1 unavailable; GPU0 is forbidden")
            print("Waiting for shared GPU1 lease / idle GPU", flush=True)
            time.sleep(30)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=RECIPES)
    ap.add_argument("--output-dir", type=Path, default=LB.parent / "model-release-20261010")
    ap.add_argument("--reuse-moe", action="store_true", help="D71 only: reuse identical verified D76 fitted heads")
    ap.add_argument("--resume-fitted", action="store_true", help="Validate and finalize saved fitted models without retraining")
    args = ap.parse_args()
    name, kind, freq = RECIPES[args.experiment]
    if args.reuse_moe and args.experiment != "E0146":
        raise ValueError("Shared D76 heads apply only to D71")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    needs_cuda = args.experiment in ("E0221", "E0107", "E0146", "E0147")
    os.environ.update(OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      CUDA_VISIBLE_DEVICES="1" if needs_cuda else "")
    os.environ.pop("EXPOS_SMOKE", None)
    os.environ.pop("D141_DEVICE", None)
    os.environ["EXPOS_OUT"] = str(out)
    from model_release.regular_runtime import predict_bundle
    script = next((LB / "scripts/seongeun" / name).glob("*.py"))
    spec = importlib.util.spec_from_file_location("submitted_recipe", script)
    submitted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submitted)
    canonical = json.dumps(submitted.CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == submitted.CONFIG_SHA256 == script.stem
    mismatches = []
    for rel, expected in submitted.CONFIG["data"]["source_sha256"].items():
        if digest(LB / rel) != expected:
            if args.experiment == "E0221" and rel.startswith("expos/"):
                mismatches.append(rel)
            else:
                raise ValueError(f"Source identity mismatch: {rel}")
    for rel, expected in submitted.CONFIG["data"]["input_sha256"].items():
        if rel.startswith("store/") and digest(LB / rel) != expected:
            raise ValueError(f"Input identity mismatch: {rel}")
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    sys.path.insert(0, str(LB / "experiments/d141_e0147_ablation"))
    import ablate as A
    sys.path.insert(0, str(LB / "experiments/expos_runners"))
    import official_members as OM
    A.SCRATCH = out / "cache" / ("enlil" if args.experiment == "E0471" else "standard")
    A.R.SCRATCH = A.SCRATCH
    arm = "enlil" if args.experiment == "E0471" else "depth4"
    if args.experiment == "E0471":
        sys.path.insert(0, str(LB / "experiments/d148_enlil_live"))
        import run as D148
        A.ia.enlil_runs = D148.enlil_runs
        # The submitted runner uses this same local+live DONKI source.
        submitted.validate()
    print(f"Building {args.experiment} {freq} feature frame", flush=True)
    df = A.build_train(freq, None, {})
    cfg = A.ARMS[arm] if kind == "ablation" else {}
    cols = A.feature_columns(df, cfg)
    tr = df[df["target_t"] < A.CUTOFF]
    meta, block = A.ST.load_inputs()
    te, _ = A.R.add_strips(A.R.base_grid(A.ORIGINS), meta, block)
    print(f"Fitting {len(tr)} rows, {len(cols)} features", flush=True)
    gpu = submitted.CONFIG["hyperparameters"].get("device") == "cuda"
    params = dict(A.ia.PARAMS, n_jobs=4 if gpu else 6, device="cuda" if gpu else "cpu")
    state = {"include_enlil": args.experiment == "E0471", "arm": arm if kind == "ablation" else None,
             "cutoff": A.CUTOFF.isoformat(),
             "train_rows": len(tr), "training_device": params["device"], "source_mismatches_outside_training": mismatches}
    with gpu1_lease() if gpu else contextlib.nullcontext():
        saved_models = None
        if args.resume_fitted:
            saved = out / f"{name}_seongeun.pkl.pending"
            if not saved.is_file():
                saved = saved.with_suffix("")
            with saved.open("rb") as stream:
                loaded = pickle.load(stream)
            assert loaded["config_sha256"] == submitted.CONFIG_SHA256 and loaded["columns"] == cols
            saved_models = loaded["models"]
        if saved_models is not None and gpu:
            restore_cuda_models(saved_models)
        if kind == "direct":
            models = saved_models if saved_models is not None else xgb.XGBRegressor(**params).fit(tr[cols], tr["y"])
            reference = models.predict(te[cols])
        elif kind == "hsplit":
            import hsplit_official as HO
            models = saved_models if saved_models is not None else tuple(xgb.XGBRegressor(**params).fit(part[cols], part["y"]) for part in
                                                                        (tr[tr.h <= 6], tr[tr.h.between(7, 24)], tr))
            reference = HO.predict(A.R, models, te, cols)
        elif kind == "ablation":
            models = saved_models if saved_models is not None else A.fit_cell(tr, cols, cfg, 0)
            reference = A.blend(A.member_preds(models, te, cols, cfg), te.h.to_numpy(), cfg)
            state["weights"] = A.WEIGHTS
        else:
            if args.reuse_moe:
                with (out / "xgb_d76_lead_moe_seongeun.pkl").open("rb") as stream:
                    source = pickle.load(stream)
                assert source["kind"] == "moe" and source["columns"] == cols
                for key in ("event_seed", "level_seeds", "train_origin_grid", "train_target_cutoff_utc", "xgboost_base"):
                    assert source["config"]["hyperparameters"][key] == submitted.CONFIG["hyperparameters"][key], key
                models = source["models"]
                restore_cuda_models(models)
                state["shared_fit_source"] = source["experiment"]
            else:
                models = saved_models if saved_models is not None else OM.fit_set(A.R, df, cols, params)
            mem = OM.members(A.R, models, te, cols)
            if kind == "lvlmean":
                reference = mem["d45_lvlmean"]
            elif kind == "stack":
                state["weights"] = submitted.CONFIG["hyperparameters"]["weights"]
                reference = sum(state["weights"][m] * mem[m] for m in A.MEMBERS)
            else:
                state["weights"] = A.WEIGHTS
                reference = A.R.MOE.blend_per_lead(mem, A.WEIGHTS, te.h.to_numpy())
        bundle = dict(format_version=1, experiment=name, config=submitted.CONFIG,
                      config_sha256=submitted.CONFIG_SHA256, kind=kind, models=models, state=state, columns=cols)
        cpu_models(models, preserve_cuda_arithmetic=gpu)
        state["prediction_device"] = "cpu"
        if gpu:
            state["cpu_prediction_mode"] = "gpu_base_score_last"
        path = out / f"{name}_seongeun.pkl.pending"
        with path.open("wb") as stream:
            pickle.dump(bundle, stream, protocol=5)
        with path.open("rb") as stream:
            restored = pickle.load(stream)
        actual = predict_bundle(restored, te)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-6)
    rows = pd.DataFrame({"origin_last_input_utc": te.origin.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "horizon_hours": te.h.astype(int), "pred_kms": actual})
    assert len(rows) == 2833 * 72 and not rows.duplicated(["origin_last_input_utc", "horizon_hours"]).any()
    evidence = out / "validation"
    evidence.mkdir(exist_ok=True)
    rows.to_parquet(evidence / f"{args.experiment}_reload.parquet", index=False)
    rows.assign(pred_kms=np.asarray(reference, dtype=float)).to_parquet(evidence / f"{args.experiment}_memory.parquet", index=False)
    te.head(72).to_parquet(evidence / f"{args.experiment}_feature_sample.parquet", index=False)
    stored = out / "references" / f"{name}.parquet"
    if not stored.is_file():
        raise FileNotFoundError(f"Authoritative submission reference required: {stored}")
    orig = pd.read_parquet(stored)
    for frame in (orig, rows):
        frame["origin_last_input_utc"] = pd.to_datetime(frame["origin_last_input_utc"], utc=True)
    merged = rows.merge(orig, on=["origin_last_input_utc", "horizon_hours"], validate="one_to_one", suffixes=("_new", "_submitted"))
    assert len(merged) == len(rows) == len(orig)
    delta = float(np.max(np.abs(merged.pred_kms_new - merged.pred_kms_submitted)))
    report = dict(experiment=args.experiment, file=path.name.removesuffix(".pending"), sha256=digest(path), rows=len(rows),
                  memory_reload_max_abs=float(np.max(np.abs(actual-reference))), submitted_max_abs=delta,
                  submitted_parity_passed=delta <= 1e-3, state=state)
    (evidence / f"{args.experiment}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if delta > 1e-3:
        raise RuntimeError("Submitted prediction mismatch; do not publish this artifact")
    path.rename(path.with_suffix(""))


if __name__ == "__main__":
    main()
