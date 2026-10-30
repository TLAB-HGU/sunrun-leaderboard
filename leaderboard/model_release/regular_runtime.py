"""Training-free inference for the released fitted regular XGBoost recipes."""
import numpy as np

MEMBERS = ("e0003_line", "d33_hsplit", "d45_lvlmean")
BANDS = (("1-6", 1, 6), ("7-24", 7, 24), ("25-48", 25, 48), ("49-72", 49, 72))


def predict_estimator(model, frame):
    """Reproduce CUDA's base-score-last float32 accumulation on CPU.

    Export records the original intercept as a booster attribute and sets the
    CPU booster's base_score to zero. The learned trees remain unchanged.
    """
    pred = model.predict(frame)
    base = model.get_booster().attr("release_gpu_base_score")
    return pred if base is None else pred + np.float32(float(base))


def predict_bundle(bundle, te):
    cols, models, kind = bundle["columns"], bundle["models"], bundle["kind"]
    missing = set(cols) - set(te.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    h = te["h"].to_numpy()
    if not np.isin(h, np.arange(1, 73)).all():
        raise ValueError("Horizons must be integers in 1..72")
    masks = (h <= 6, (h >= 7) & (h <= 24), h > 24)
    if kind == "direct":
        out = predict_estimator(models, te[cols])
    elif kind == "hsplit":
        out = np.empty(len(te), dtype=np.float64)
        for mask, model in zip(masks, models):
            if mask.any():
                out[mask] = predict_estimator(model, te.loc[mask, cols])
    else:
        short, long_, _ = models
        long_pred = np.asarray(predict_estimator(long_, te[cols]), dtype=np.float64)
        mem = {"e0003_line": long_pred}
        for name, seeds in (("d33_hsplit", (1,)), ("d45_lvlmean", (0, 1, 2))):
            p = long_pred.copy()
            for i, mask in enumerate(masks[:2]):
                if not mask.any():
                    continue
                preds = [predict_estimator(short[s][i], te.loc[mask, cols]) for s in seeds]
                if kind == "ablation":
                    p[mask] = np.mean(preds, axis=0)
                else:
                    p[mask] = sum(np.asarray(v, dtype=np.float64) for v in preds) / len(preds)
            mem[name] = p
        if kind == "lvlmean":
            out = mem["d45_lvlmean"]
        elif kind == "stack":
            out = sum(bundle["state"]["weights"][m] * mem[m] for m in MEMBERS)
        elif kind in ("moe", "ablation"):
            out = np.empty(len(te), dtype=np.float64)
            for band, lo, hi in BANDS:
                mask = (h >= lo) & (h <= hi)
                w = bundle["state"]["weights"][band]
                z = sum(w[m] for m in MEMBERS) if kind == "ablation" else 1.0
                out[mask] = sum(w[m] / z * mem[m][mask] for m in MEMBERS)
        else:
            raise ValueError(f"Unknown regular bundle kind: {kind}")
    out = np.asarray(out, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("Model produced non-finite predictions")
    return out
