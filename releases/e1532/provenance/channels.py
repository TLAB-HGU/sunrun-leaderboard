"""Arrival/propagation channels for a champion-anchored residual head (2026-10-18, user: run both tracks).

HSS track  : HUXt (E1466 causal TD-HUXt, GONG SW-012 -> PFSS/WSA) Earth trajectory, timing-aligned with recent L1 data.
             delta = lag that best matches the HUXt nowcast history to ACE (obs+2h) over the last 72h; aligned forecast
             = HUXt(t+h+delta). Also the recent ACE-HUXt level bias.
CME track  : DONKI WSA-Enlil runs with modelCompletionTime <= origin. Per CME (latest run per cmeStartTime), the
             arrival probability by horizon Phi((h - ETA)/sigma) and since-arrival decay; sigma and amplitude are learned
             by the head (sigma chosen on training folds only).
All channels use only information available at the origin.
"""
import glob, json, ast
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

LB = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
H = np.arange(1, 73)
LAT = 2


class Huxt:
    def __init__(self):
        z = np.load(HERE / 'cache/huxt_earth.npz')
        self.origins = pd.DatetimeIndex(pd.to_datetime(z['origins_ns'], utc=True))
        self.traj = z['traj'].astype(np.float32)
        self.pos = pd.Series(np.arange(len(self.origins)), index=self.origins)
        # nowcast history: HUXt speed at t+1 issued at origin t (causal at t)
        self.now = pd.Series(self.traj[:, 0], index=self.origins + pd.Timedelta(hours=1))

    def channels(self, origins, ace_v, max_lag=48, win=72, step=2):
        """ace_v: hourly ACE speed indexed by observation time (NaN where missing).
        Per origin p, the causal HUXt series S_p(tau) = nowcast issued at tau-1 for tau <= p+1, and the forecast issued
        at p for p+1 < tau <= p+72. Lag d: ACE(tau) ~ S_p(tau + d) over the last `win` available ACE hours;
        aligned forecast = S_p(p + h + d), clipped to the last forecast hour."""
        origins = pd.DatetimeIndex(origins)
        idx = self.pos.reindex(origins).to_numpy()
        ok = np.isfinite(idx)
        raw = np.full((len(origins), 72), np.nan, np.float32)
        raw[ok] = self.traj[idx[ok].astype(int)]
        g = pd.date_range(min(self.now.index.min(), ace_v.index.min()), origins.max() + pd.Timedelta(hours=1), freq='h', tz='UTC')
        nv, av = self.now.reindex(g).to_numpy(), ace_v.reindex(g).to_numpy()
        gi = g.get_indexer(origins)
        lags = np.arange(-max_lag, max_lag + 1, step)
        n = len(origins)
        best, err, bias = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        al = np.full_like(raw, np.nan)
        pad = win + max_lag + LAT + 2
        for i in np.where(ok)[0]:
            p = gi[i]
            if p < pad:
                continue
            # S over absolute grid positions [p - pad, p + 72]
            S = np.concatenate([nv[p - pad:p + 2], raw[i, 1:]])     # nv[p+1] is the lead-1 value issued at p
            base = p - pad
            last = p - LAT
            obs = av[last - win + 1:last + 1]
            if np.isfinite(obs).sum() < 24:
                continue
            e = np.full(len(lags), np.inf)
            for j, d in enumerate(lags):
                m = S[last - win + 1 + d - base:last + 1 + d - base]
                k = np.isfinite(m) & np.isfinite(obs)
                if k.sum() >= 24:
                    e[j] = np.mean(np.abs(obs[k] - m[k]))
            if not np.isfinite(e).any():
                continue
            j = int(np.argmin(e)); d = int(lags[j]); best[i], err[i] = d, e[j]
            m = S[last - win + 1 - base:last + 1 - base]; k = np.isfinite(m) & np.isfinite(obs)
            bias[i] = np.mean(obs[k] - m[k]) if k.sum() >= 24 else np.nan
            pos = np.clip(p + H + d - base, 0, len(S) - 1)
            al[i] = S[pos]
        return dict(huxt_raw=raw, huxt_al=al, huxt_lag=best, huxt_err=err, huxt_bias=bias)


def enlil_runs():
    files = [f for f in sorted(glob.glob(str(LB / 'store/headroom-audit/donki/WSAEnlilSimulations_*.json'))) if f[-12:-5] < '2026-06']
    files += sorted(glob.glob(str(LB / 'store/donki-live/WSAEnlilSimulations_*.json')))
    rows = []
    for f in files:
        for s in json.load(open(f)):
            if not s.get('estimatedShockArrivalTime'):
                continue
            ci = s.get('cmeInputs') or []
            if isinstance(ci, str):
                ci = ast.literal_eval(ci)
            c = max(ci, key=lambda r: r.get('speed') or 0) if ci else {}
            kp = max([float(s.get(k)) for k in ('kp_90', 'kp_135', 'kp_180') if s.get(k) not in (None, 'None')] or [0])
            rows.append(dict(done=pd.Timestamp(s['modelCompletionTime']), eta=pd.Timestamp(s['estimatedShockArrivalTime']),
                             cme_start=pd.Timestamp(c['cmeStartTime']) if c.get('cmeStartTime') else pd.NaT,
                             speed=float(c.get('speed') or np.nan), lon=float(c.get('longitude') or np.nan),
                             half=float(c.get('halfAngle') or np.nan), kp=kp,
                             glancing=str(s.get('isEarthGB')) == 'True', minor=str(s.get('isEarthMinorImpact')) == 'True'))
    r = pd.DataFrame(rows).drop_duplicates()
    for c in ('done', 'eta', 'cme_start'):
        r[c] = pd.to_datetime(r[c], utc=True)
    return r.sort_values('done').reset_index(drop=True)


def cme_channels(origins, runs, sigmas=(6, 9, 12, 18, 24)):
    """Per origin: for the latest run of each CME (cme_start) with done <= origin and ETA in (origin-48h, origin+120h]."""
    origins = pd.DatetimeIndex(origins)
    out = {f'cme_phi{s}': np.zeros((len(origins), 72), np.float32) for s in sigmas}
    for k in ('cme_phi_fast', 'cme_phi_kp', 'cme_phi_glance'):
        out[k] = {s: np.zeros((len(origins), 72), np.float32) for s in sigmas}
    out['cme_n'] = np.zeros(len(origins), np.float32)
    done = runs.done.astype('int64').to_numpy(); eta = runs.eta.astype('int64').to_numpy(); HR = 3_600_000_000_000
    for i, o in enumerate(origins):
        ov = o.value
        sel = runs[(done <= ov) & (eta > ov - 48 * HR) & (eta <= ov + 120 * HR)]
        if sel.empty:
            continue
        sel = sel.sort_values('done').groupby(sel.cme_start.fillna(sel.eta)).tail(1)
        out['cme_n'][i] = len(sel)
        for _, r in sel.iterrows():
            dt = (r.eta - o) / pd.Timedelta(hours=1)
            fast = np.clip(((r.speed if np.isfinite(r.speed) else 600) - 400) / 400, 0, 3)
            for s in sigmas:
                phi = norm.cdf((H - dt) / s).astype(np.float32)
                out[f'cme_phi{s}'][i] += phi
                out['cme_phi_fast'][s][i] += phi * fast
                out['cme_phi_kp'][s][i] += phi * r.kp / 9
                out['cme_phi_glance'][s][i] += phi * float(r.glancing or r.minor)
    return out
