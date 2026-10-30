"""Portable fitted residual and event-branch inference; no training or file reads."""
import numpy as np
import pandas as pd

from .regular_runtime import predict_estimator


def predict_mixture(fs, frame, columns, weights):
    short, long, seeds, event = fs
    h = frame['h'].to_numpy()
    line = np.asarray(long.predict(frame[columns]), dtype=float)
    switch, mean = line.copy(), line.copy()
    for band, mask in enumerate((h <= 6, (h >= 7) & (h <= 24))):
        if mask.any():
            preds = {s: np.asarray(short[s][band].predict(frame.loc[mask, columns]), dtype=float) for s in seeds}
            switch[mask] = preds[event]
            mean[mask] = sum(preds.values()) / len(seeds)
    members = {'e0003_line': line, 'd33_hsplit': switch, 'd45_lvlmean': mean}
    out = np.empty(len(frame))
    for label, (lo, hi) in zip(('1-6', '7-24', '25-48', '49-72'), ((1, 6), (7, 24), (25, 48), (49, 72))):
        mask = (h >= lo) & (h <= hi)
        out[mask] = sum(weights[label][m] * members[m][mask] for m in members)
    return out


def past_matrix(origins, filled):
    return np.array([filled.reindex(pd.date_range(o - pd.Timedelta('11h'), o, freq='h')).to_numpy(float) for o in origins])


def freeze_pool(times, paths, filled, cutoff):
    """Store past scenarios and learned-model residual outcomes strictly before cutoff."""
    times = pd.DatetimeIndex(times)
    past = past_matrix(times, filled)
    errors = np.full_like(past, np.nan)
    errors[12:] = past[12:] - paths[:-12, :12]
    outcomes = np.array([filled.reindex(pd.date_range(t + pd.Timedelta('1h'), periods=72, freq='h')).to_numpy(float) for t in times])
    mask = np.asarray(times + pd.Timedelta('72h') < cutoff)
    return {'times': times[mask], 'past': past[mask], 'errors': errors[mask], 'paths': paths[mask], 'residuals': (outcomes-paths)[mask]}


def residual_replay(origins, filled, paths, hind, state):
    pool = state['pool']
    times = pool['times']
    qp = past_matrix(origins, filled)
    qe = qp - hind[:, :12]
    scales = state['scales']
    output = np.full((len(origins), 72), np.nan)
    finite = np.isfinite(pool['past']).all(1) & np.isfinite(pool['errors']).all(1) & np.isfinite(pool['paths']).all(1)
    for i, origin in enumerate(origins):
        if not all(np.isfinite(x).all() for x in (qp[i], qe[i], paths[i])):
            continue
        mask = finite & (times >= origin-pd.Timedelta(hours=state['lookback_h'])) & (times <= origin-pd.Timedelta('655h'))
        ix = np.flatnonzero(mask)
        if len(ix):
            distance = (np.mean((pool['past'][ix]-qp[i])**2, axis=1)/scales[0]
                        + np.mean((pool['errors'][ix]-qe[i])**2, axis=1)/scales[1]
                        + np.mean((pool['paths'][ix]-paths[i])**2, axis=1)/scales[2])
            picks = ix[np.argsort(distance, kind='stable')[:state['k']]]
            output[i] = np.nanmean(pool['residuals'][picks], axis=0)
    return output


def predict_residual(bundle, origins, grid_builder, filled_speed):
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    state, cols = bundle['state'], bundle['columns']
    if (origins < pd.Timestamp(state['cutoff'])).any():
        raise ValueError('Fitted model is only valid for origins at or after its training cutoff')
    frame = grid_builder(origins)
    hind_frame = grid_builder(origins-pd.Timedelta('12h'))
    paths = predict_mixture(bundle['models']['oof'], frame, cols, state['weights']).reshape(-1, 72)
    hind = predict_mixture(bundle['models']['oof'], hind_frame, cols, state['weights']).reshape(-1, 72)
    dv = residual_replay(origins, filled_speed, paths, hind, state).ravel()
    base = predict_mixture(bundle['models']['full'], frame, cols, state['weights'])
    return np.where(np.isfinite(dv), base+state['weight']*dv, base)


def predict_branch(bundle, frame, analog_table):
    full = frame.merge(analog_table, on=['origin', 'h'], how='left', validate='one_to_one')
    level, joint, _ = bundle['models']
    h = frame['h'].to_numpy()
    lp, jp = [], []
    for seed, models in level.items():
        pred = np.empty(len(frame))
        for mask, model in zip((h <= 6, (h >= 7) & (h <= 24), h > 24), models):
            if mask.any():
                pred[mask] = predict_estimator(model, frame.loc[mask, bundle['columns']['level']])
        lp.append(pred)
        jp.append(np.asarray(predict_estimator(joint[seed], full[bundle['columns']['joint']]), dtype=float))
    return np.where(np.all(np.asarray(jp) >= 600, axis=0), np.mean(jp, axis=0), np.mean(lp, axis=0))


CH_KEEP = [
    "ch_t035_meridian30_equatorial_area",
    "ch_t035_equatorial_area",
    "ch_t035_meridian30_area",
    "ch_t045_meridian30_equatorial_area",
    "ch_t035_area",
    "ch_t045_equatorial_area",
]

CH3 = [
    "ch_t035_meridian30_equatorial_area",
    "ch_t035_equatorial_area",
    "ch_t045_meridian30_equatorial_area",
]

KEY_SETS = {
    "CH6": ["s_" + c for c in CH_KEEP],
    "CH6V2": ["s_" + c for c in CH_KEEP] + ["s_v_m24", "s_v_d24"],
    "CH3V3": ["s_" + c for c in CH3] + ["s_v_last", "s_v_m24", "s_v_d24"],
}

METRICS = ("l2", "l1")

WIN_LO_H = 20 * 24

WIN_HI_H = 35 * 24

K_TOP = 3

H_ALL = np.arange(1, 73)

def compute_d46_analog(origins, state: pd.DataFrame, filled: pd.Series,
                keys: list, metric: str, scaler: dict) -> pd.DataFrame:
    """Per-(origin,h) replay columns for one key set.

    For query origin O the pool is hourly H with O-35d <= H <= O-20d and the
    replayed target filled[H+h] always predates O (H+72h <= O-17d). Origins
    with no finite query state or no finite candidate yield NaN (branch off).
    """
    if metric not in METRICS:
        raise ValueError(f"D46 metric allows only {METRICS}, got {metric!r}")
    unknown = [k for k in keys if k not in state.columns]
    if unknown:
        raise ValueError(f"D46 keys must be state columns, unknown={unknown}")
    origins = pd.DatetimeIndex(origins, tz="UTC")
    hours = state.index
    state_vals = state.to_numpy(float)
    filled_vals = filled.reindex(hours).to_numpy(float)
    all_keys = list(state.columns)
    keys_idx = np.array([all_keys.index(k) for k in keys], dtype=int)
    mu = scaler["mean"].reindex(keys).to_numpy(float)
    sd = scaler["std"].reindex(keys).to_numpy(float)
    n = len(origins)
    best = np.full((n, 72), np.nan)
    mean3 = np.full((n, 72), np.nan)
    dist = np.full(n, np.nan)
    spread = np.full((n, 72), np.nan)
    for i, o in enumerate(origins):
        o_pos = int(hours.searchsorted(o))
        if o_pos >= len(hours) or hours[o_pos] != o:
            continue
        lo = o - pd.Timedelta(hours=int(WIN_HI_H))
        hi = o - pd.Timedelta(hours=int(WIN_LO_H))
        c_lo = int(hours.searchsorted(lo))
        c_hi = int(hours.searchsorted(hi, side="right")) - 1
        c_hi = min(c_hi, o_pos - 1)
        if c_hi < c_lo:
            continue
        q = (state_vals[o_pos, keys_idx] - mu) / sd
        if not np.isfinite(q).all():
            continue
        cand = (state_vals[c_lo: c_hi + 1][:, keys_idx] - mu) / sd
        ok = np.isfinite(cand).all(axis=1)
        if not bool(ok.any()):
            continue
        idx = np.flatnonzero(ok)
        c = cand[idx]
        if metric == "l1":
            d = np.abs(c - q).sum(axis=1)
        else:
            d = ((c - q) ** 2).sum(axis=1)
        order = np.argsort(d, kind="stable")[:K_TOP]
        picks = idx[order] + c_lo
        dsorted = d[order]
        dist[i] = float(dsorted[0])
        curves = []
        for p in picks:
            seg = filled_vals[p + 1: p + 73]
            if len(seg) < 72:
                seg = np.pad(seg, (0, 72 - len(seg)), constant_values=np.nan)
            curves.append(seg)
        curves = np.array(curves, dtype=float)
        best[i] = curves[0]
        with np.errstate(all="ignore"):
            mean3[i] = np.nanmean(curves, axis=0)
            spread[i] = np.nanstd(curves, axis=0) if len(curves) > 1 else np.zeros(72)
    df = pd.DataFrame({
        "origin": np.repeat(origins.to_numpy(), 72),
        "h": np.tile(H_ALL, n),
        "analog_best": best.ravel(),
        "analog_mean3": mean3.ravel(),
        "analog_dist": np.repeat(dist, 72),
        "analog_spread": spread.ravel(),
    })
    df["origin"] = pd.to_datetime(df["origin"], utc=True)
    return df

def build_d46_state(filled, ch):
    lo = max(filled.index.min(), ch.index.min())
    hi = min(filled.index.max(), ch.index.max())
    full = pd.date_range(lo, hi, freq="h", tz="UTC")
    filled = filled.reindex(full)
    ch = ch.reindex(full)
    frame = pd.DataFrame({"v": filled})
    for c in CH_KEEP:
        frame[c] = ch[c] if c in ch.columns else np.nan
    v = frame["v"].astype(float)
    chf = frame[[c for c in CH_KEEP if c in frame.columns]].reindex(full).ffill(limit=6)
    m6 = v.rolling(6, min_periods=1).mean()
    m24 = v.rolling(24, min_periods=1).mean()
    state = pd.DataFrame(index=full)
    for c in CH_KEEP:
        state["s_" + c] = chf[c] if c in chf.columns else np.nan
    state["s_v_last"] = v
    state["s_v_m6"] = m6
    state["s_v_m24"] = m24
    state["s_v_d24"] = m6 - m6.shift(24)
    return state, filled
