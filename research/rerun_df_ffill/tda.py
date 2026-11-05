"""TDA feature extraction for EUV full-disk frames (arXiv 2610.35549 replication).

Pipeline (paper Sec. 3.2, adapted to GOES-18 SUVI L1b Fe195):
    radiance -> disk crop -> 2x rebin -> clip -> asinh stretch -> [-1,1] -> invert
    -> cubical complex (gudhi) -> H0/H1 persistence diagrams
    -> fixed 14x14 persistence images (Eq. 2, w(p)=p, sigma=0.05)

The paper's absolute constants (clip 7757.31, a = 0.005, applied to AIA data
numbers) do not transfer to SUVI radiance in W m-2 sr-1; the clip is replaced by
the 99.5th percentile of the disk pixels of the same frame. The asinh shape
parameter a = 0.005 is kept because it acts on the [0,1]-normalized image.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

import numpy as np

A = 0.005
GRID = 14
SIGMA = 0.05
REBIN = 2
EDGES = np.linspace(0.0, 1.0, GRID + 1)
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])
CENT_B, CENT_P = np.meshgrid(CENT, CENT, indexing="ij")
PIX = 1.0 / GRID
NFEAT = 2 * GRID * GRID


def read_frame(path: str) -> tuple[np.ndarray, float, float, float]:
    """Return (radiance, rsun_px, crpix1, crpix2) in physics units."""
    import h5netcdf

    with h5netcdf.File(path, "r") as f:
        rad = np.array(f["RAD"][:], dtype=np.float32)
        sf = float(f["RAD"].attrs.get("scale_factor", 1.0))
        ao = float(f["RAD"].attrs.get("add_offset", 0.0))
        rs = float(f["RSUN"][()])
        if not np.isfinite(rs) or rs <= 0:
            diam = float(f["DIAM_SUN"][()])
            rs = 0.5 * diam if np.isfinite(diam) and diam > 0 else float("nan")
        c1 = float(f["CRPIX1"][()])
        c2 = float(f["CRPIX2"][()])
    return rad * sf + ao, rs, c1, c2


def _derive_geometry(rad: np.ndarray) -> tuple[float, float, float]:
    """Fallback when RSUN/CRPIX carry fill values: centroid + area-equivalent radius."""
    hi = float(np.nanmax(rad))
    if not np.isfinite(hi) or hi <= 0:
        raise ValueError("frame has no usable intensity")
    thr = max(0.05 * hi, 1e-3)
    m = np.nan_to_num(rad, nan=0.0) > thr
    if m.sum() < 1000:
        raise ValueError("disk mask too small")
    yy, xx = np.nonzero(m)
    c1 = float(xx.mean()) + 1.0
    c2 = float(yy.mean()) + 1.0
    rs = float(np.sqrt(m.sum() / np.pi))
    return rs, c1, c2


def preprocess(rad: np.ndarray, rs: float, c1: float, c2: float) -> tuple[np.ndarray, np.ndarray]:
    """Disk crop + rebin + asinh normalization + inversion (paper Sec. 3.2)."""
    if not np.isfinite(rs) or rs < 10 or not (np.isfinite(c1) and np.isfinite(c2)):
        rs, c1, c2 = _derive_geometry(rad)
    h, w = rad.shape
    cy, cx = c2 - 1.0, c1 - 1.0
    half = int(np.ceil(rs * 1.05))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        pad = 4 * half
        big = np.full((pad, pad), np.nan, dtype=np.float32)
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(h, y1), min(w, x1)
        big[ys - y0:ys - y0 + (ye - ys), xs - x0:xs - x0 + (xe - xs)] = rad[ys:ye, xs:xe]
        rad = big
        cy, cx = cy - y0, cx - x0
        y0, y1, x0, x1 = 0, pad, 0, pad
    sub = rad[y0:y1, x0:x1]
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    disk = np.hypot(xx - (cx - x0), yy - (cy - y0)) < rs
    sub = np.where(disk, sub, 0.0).astype(np.float32)
    hh, ww = (sub.shape[0] // REBIN) * REBIN, (sub.shape[1] // REBIN) * REBIN
    img = sub[:hh, :ww].reshape(hh // REBIN, REBIN, ww // REBIN, REBIN).mean(axis=(1, 3))
    dk = disk[:hh, :ww].reshape(hh // REBIN, REBIN, ww // REBIN, REBIN).mean(axis=(1, 3)) > 0.5
    hi = float(np.percentile(img[dk], 99.5)) if dk.any() else float(img.max())
    x = np.clip(img, 0.0, hi) / max(hi, 1e-6)
    norm = np.arcsinh(x / A) / np.arcsinh(1.0 / A)      # [0, 1]
    norm = 2.0 * norm - 1.0                              # [-1, 1]
    proc = norm.max() - norm                             # invert: coronal holes become peaks
    return np.ascontiguousarray(proc.astype(np.float64)), dk


def diagrams(proc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from gudhi import CubicalComplex

    cc = CubicalComplex(top_dimensional_cells=proc)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    out = []
    for k in (0, 1):
        dg = np.asarray(cc.persistence_intervals_in_dimension(k), dtype=float)
        out.append(dg[np.isfinite(dg).all(axis=1)] if len(dg) else np.zeros((0, 2)))
    return out[0], out[1]


def persistence_image(dg: np.ndarray) -> np.ndarray:
    """Paper Eq. (2) on a fixed 14x14 grid in (birth, persistence) coordinates."""
    if len(dg) == 0:
        return np.zeros((GRID, GRID))
    b = dg[:, 0]
    p = dg[:, 1] - dg[:, 0]
    ok = np.isfinite(b) & np.isfinite(p) & (p > 0)
    b, p = b[ok], p[ok]
    if len(b) == 0:
        return np.zeros((GRID, GRID))
    w = p
    db = CENT_B.ravel()[None, :] - b[:, None]
    dp = CENT_P.ravel()[None, :] - p[:, None]
    k = np.exp(-(db ** 2 + dp ** 2) / (2 * SIGMA ** 2))
    return ((w[:, None] * k).sum(axis=0) * PIX ** 2).reshape(GRID, GRID)


def features(path: str) -> tuple[np.ndarray, dict]:
    rad, rs, c1, c2 = read_frame(path)
    proc, _ = preprocess(rad, rs, c1, c2)
    d0, d1 = diagrams(proc)
    vec = np.concatenate([persistence_image(d0).ravel(), persistence_image(d1).ravel()]).astype(np.float32)
    meta = {
        "n_h0": int(len(d0)),
        "n_h1": int(len(d1)),
        "max_h1_persistence": float((d1[:, 1] - d1[:, 0]).max()) if len(d1) else 0.0,
        "shape": int(proc.shape[0]),
    }
    return vec, meta


def ace_features(series: pd.Series, origins: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """Causal ACE speed context at each origin (only rows with timestamp <= origin)."""
    s = series
    idx = s.index
    cols, names = [], []
    pos = idx.searchsorted(origins, side="right") - 1   # last row <= origin

    def at(lag):
        p = pos - lag
        ok = p >= 0
        v = np.full(len(origins), np.nan)
        v[ok] = s.to_numpy()[p[ok]]
        return v

    for lag in (0, 1, 2, 3, 6, 12, 24, 48, 72):
        cols.append(at(lag)); names.append(f"ace_lag{lag}")
    vals = s.to_numpy()
    for w in (6, 12, 24, 48, 72):
        win = np.full((len(origins), w), np.nan)
        for k in range(w):
            p = pos - k
            ok = p >= 0
            win[ok, k] = vals[p[ok]]
        cols.append(np.nanmean(win, axis=1)); names.append(f"ace_mean{w}")
        if w in (24, 72):
            cols.append(np.nanstd(win, axis=1)); names.append(f"ace_std{w}")
            cols.append(np.nanmin(win, axis=1)); names.append(f"ace_min{w}")
            cols.append(np.nanmax(win, axis=1)); names.append(f"ace_max{w}")
    lag = at(24) - at(0)
    cols.append(lag / 24.0); names.append("ace_slope24")
    doy = origins.dayofyear.to_numpy()
    cols.append(np.sin(2 * np.pi * doy / 365.25)); names.append("doy_sin")
    cols.append(np.cos(2 * np.pi * doy / 365.25)); names.append("doy_cos")
    X = np.column_stack(cols).astype(np.float64)
    return X, names

H=72
ALPHAS=(100.,1000.,10000.)
PI_COLS=[f"f{i:03d}" for i in range(392)]

CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}

QUERY_COUNTS = {"F1": 2089, "F2": 2113, "F3": 2113}

def merge(out):
    out = Path(out)
    frames = []
    for period in CUTOFFS:
        z = np.load(out / f"{period}_predictions.npz")
        origins = pd.DatetimeIndex(z["origins"], tz="UTC")
        pred = z["pred"]
        if len(origins) != QUERY_COUNTS[period] or pred.shape != (len(origins), H) or not np.isfinite(pred).all():
            raise ValueError(f"{period}: incomplete full-run predictions")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), len(origins)), "pred_kms": pred.ravel()}))
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["origin_last_input_utc", "horizon_hours"]).any():
        raise ValueError("duplicate output keys")
    result.to_parquet(out / "predictions.parquet", index=False)
    return len(result)


def file_end(path):
    """NOAA filename uses year, Julian day, HHMMSS and fractional seconds."""
    match=re.search(r'_e(\d{13})(\d*)_',Path(path).name)
    if match is None:
        raise ValueError(f'cannot parse observation end: {path}')
    stamp=pd.to_datetime(match[1],format='%Y%j%H%M%S',utc=True)
    return stamp+pd.Timedelta(seconds=float('0.'+(match[2] or '0')))


def frame_paths(directory,end=None):
    paths=sorted(Path(directory).rglob('*Fe195*.nc'))
    records=[(file_end(p),str(p.resolve())) for p in paths]
    if end is not None:
        records=[r for r in records if r[0]<=end]
    return sorted(records)


def cached_frame(item):
    """Per-file atomic cache uses source path/size/mtime, never file digests."""
    expected_end,path,cache=item
    import h5netcdf
    source=Path(path);stat=source.stat()
    folder=Path(cache)/str(expected_end.year)
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/(source.stem+'.npz')
    if output.exists():
        with np.load(output,allow_pickle=False) as z:
            if str(z['path'])==str(source) and int(z['bytes'])==stat.st_size and int(z['mtime_ns'])==stat.st_mtime_ns:
                return dict(path=path,obs_end=pd.Timestamp(int(z['obs_end']),tz='UTC'),
                            vector=z['vector'],status=str(z['status']),cached=True)
    try:
        with h5netcdf.File(source,'r') as f:
            end=pd.Timestamp(f.attrs['time_coverage_end'])
            if end.tz is None:
                end=end.tz_localize('UTC')
            else:
                end=end.tz_convert('UTC')
            if abs((end-expected_end).total_seconds())>.001:
                raise ValueError('filename and observation-end metadata disagree')
            eclipse=int(f['ECLIPSE'][()]) if 'ECLIPSE' in f.variables else 0
        if eclipse==2:
            vec,meta,status=np.empty(0,np.float32),{},'excluded_eclipse2'
        else:
            vec,meta=features(str(source));status='ok'
            if vec.shape!=(392,) or not np.isfinite(vec).all():
                raise ValueError('invalid persistence-image features')
        temporary=output.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,path=str(source),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                                obs_end=end.value,vector=vec,status=status,metadata=json.dumps(meta),
                                recipe='git6b85acc0b original tda.py; A=.005/grid14/sigma.05/rebin2')
        os.replace(temporary,output)
        return dict(path=path,obs_end=end,vector=vec,status=status,cached=False)
    except Exception as exc:
        error=dict(path=path,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,error=f'{type(exc).__name__}: {exc}')
        output.with_suffix('.error.json').write_text(json.dumps(error,indent=2)+'\n')
        return dict(path=path,obs_end=expected_end,vector=None,status='error',error=error['error'],cached=False)


def build_cache(directory,cache,threads=1,end=None,limit=None):
    records=frame_paths(directory,end)
    if limit is not None:
        records=records[:limit]
    if not records:
        raise ValueError('no raw SUVI Fe195 frames')
    items=[(stamp,path,str(cache)) for stamp,path in records]
    results=[]
    if threads==1:
        iterator=map(cached_frame,items)
        for i,result in enumerate(iterator,1):
            results.append(result)
            if i%100==0:
                print(json.dumps({'frames_processed':i,'frames_total':len(items)}),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=threads) as pool:
            for i,result in enumerate(pool.map(cached_frame,items,chunksize=1),1):
                results.append(result)
                if i%100==0:
                    print(json.dumps({'frames_processed':i,'frames_total':len(items)}),flush=True)
    good=[r for r in results if r['status']=='ok']
    report=dict(total=len(results),ok=len(good),excluded_eclipse2=sum(r['status']=='excluded_eclipse2' for r in results),
                failed=sum(r['status']=='error' for r in results),cached=sum(r['cached'] for r in results),
                source=str(Path(directory).resolve()),recipe='git6b85acc0b:research/e2100_tda_pi_suvi-20261031/tda.py',
                parameters=dict(A=A,GRID=GRID,SIGMA=SIGMA,REBIN=REBIN,weight='persistence'))
    Path(cache).mkdir(parents=True,exist_ok=True)
    (Path(cache)/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    if not good:
        raise ValueError(f'no usable frames: {report}')
    frame=pd.DataFrame(np.vstack([r['vector'] for r in good]),columns=PI_COLS)
    frame.insert(0,'obs_end',pd.DatetimeIndex([r['obs_end'] for r in good]))
    frame=frame.sort_values('obs_end').drop_duplicates('obs_end',keep='last').reset_index(drop=True)
    print(json.dumps(report),flush=True)
    return frame


def frame_lookup(frame,origins):
    """Latest actual observation end <= origin; old valid frames ffill indefinitely."""
    frame=frame.sort_values('obs_end')
    ends=pd.DatetimeIndex(frame.obs_end)
    origins=pd.DatetimeIndex(origins)
    positions=ends.searchsorted(origins,side='right')-1
    observed=positions>=0
    result=np.full((len(origins),392),np.nan)
    result[observed]=frame[PI_COLS].to_numpy()[positions[observed]]
    age=np.full(len(origins),np.nan)
    age[observed]=(origins[observed]-ends[positions[observed]]).total_seconds()/3600.
    return result,observed,age


def speed_timeline(data):
    rows=pd.read_parquet(data,columns=['origin','v_lag0'],filters=[('h','==',1)])
    rows['origin']=pd.to_datetime(rows.origin,utc=True)
    if rows.groupby('origin').v_lag0.nunique(dropna=False).gt(1).any():
        raise ValueError('train/query input speeds disagree')
    series=rows.drop_duplicates('origin').sort_values('origin').set_index('origin').v_lag0
    return series.reindex(pd.date_range(series.index.min(),series.index.max(),freq='h',tz='UTC')).ffill()


def load_grid(data,period,training):
    filters=[(f'fit_{period}','==',True)] if training else [('set','==','query'),('period','==',period)]
    rows=pd.read_parquet(data,columns=['origin','h','y'],filters=filters).sort_values(['origin','h'])
    if len(rows)%H:
        raise ValueError('incomplete horizon grid')
    origins=pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H],utc=True))
    if origins.has_duplicates or not np.array_equal(rows.h.to_numpy().reshape(-1,H),np.broadcast_to(np.arange(1,H+1),(len(origins),H))):
        raise ValueError('invalid horizon grid')
    if training and not (origins+pd.Timedelta(hours=144)<=pd.Timestamp(CUTOFFS[period],tz='UTC')).all():
        raise ValueError('maturity cutoff violated')
    return origins,rows.y.to_numpy(np.float64).reshape(-1,H)


def ridge_fit_predict(x,y,query,origins,out):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import joblib
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    val_start=origins.max()-pd.Timedelta(days=60)
    val=origins>val_start
    # Training target availability cannot cross into the inner validation origins.
    fit=origins+pd.Timedelta(hours=144)<val_start
    if not fit.any() or not val.any():
        raise ValueError('insufficient prior history for alpha selection')
    selector_scaler=StandardScaler().fit(x[fit])
    a=selector_scaler.transform(x[fit]);b=selector_scaler.transform(x[val])
    scaler=StandardScaler().fit(x)
    whole,que=scaler.transform(x),scaler.transform(query)
    pred=np.empty((len(query),H));chosen=[];trace=[];inference=0.
    for h in range(H):
        train_y,valid_y=y[fit,h],y[val,h]
        observed,vo=np.isfinite(train_y),np.isfinite(valid_y)
        if not observed.any() or not vo.any():
            raise ValueError(f'h{h+1}: missing alpha-selection labels after causal target ffill')
        scores=[]
        for alpha in ALPHAS:
            model=Ridge(alpha=alpha).fit(a[observed],train_y[observed])
            loss=float(np.mean((model.predict(b[vo])-valid_y[vo])**2))
            scores.append(loss)
            trace.append(dict(horizon=h+1,alpha=alpha,validation_mse=loss))
        alpha=ALPHAS[int(np.argmin(scores))]
        observed=np.isfinite(y[:,h])
        model=Ridge(alpha=alpha).fit(whole[observed],y[observed,h])
        joblib.dump(model,out/f'h{h+1:02}.joblib')
        start=time.perf_counter();pred[:,h]=model.predict(que);inference+=time.perf_counter()-start
        chosen.append(alpha)
    joblib.dump(scaler,out/'scaler.joblib')
    pd.DataFrame(trace).to_csv(out/'selection_loss.csv',index=False)
    receipt=dict(alpha_grid=ALPHAS,chosen=chosen,selection_train_end=str(origins[fit].max()),
                 selection_valid_start=str(origins[val].min()),selection_valid_end=str(origins[val].max()),purge_hours=144)
    (out/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return pred,chosen,inference/max(len(query),1)


def run_period(data,out,period,frame,smoke=False):
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    origins,y=load_grid(data,period,True)
    query_origins,_=load_grid(data,period,False)
    if len(query_origins)!=QUERY_COUNTS[period]:
        raise ValueError('unexpected query count')
    speeds=speed_timeline(data)
    pi,ok,age=frame_lookup(frame,origins)
    ace,names=ace_features(speeds,origins)
    x=np.hstack([pi,ace]);keep=ok&np.isfinite(x).all(axis=1)&np.isfinite(y).any(axis=1)
    removed=int((~keep).sum());origins,x,y=origins[keep],x[keep],y[keep]
    if smoke:
        ix=np.unique(np.linspace(0,len(origins)-1,min(128,len(origins))).astype(int))
        origins,x,y=origins[ix],x[ix],y[ix]
        query_origins=query_origins[:2]
    pi,ok,age=frame_lookup(frame,query_origins)
    ace,_=ace_features(speeds,query_origins)
    query=np.hstack([pi,ace])
    if not ok.all() or not np.isfinite(query).all():
        raise ValueError('query lacks prior SUVI/speed input after causal ffill')
    pred,chosen,timing=ridge_fit_predict(x,y,query,origins,out/f'{period}_models')
    pred=np.clip(pred,150.,1500.)
    if not np.isfinite(pred).all():
        raise ValueError('nonfinite forecasts')
    np.savez_compressed(out/f'{period}_predictions.npz',pred=pred,origins=query_origins.asi8)
    report=dict(experiment='e2100_tda_pi_suvi72',period=period,smoke=smoke,source=str(Path(data).resolve()),
                feature_names=PI_COLS+names,feature_count=x.shape[1],train_origins=len(origins),excluded_leading_inputs=removed,
                train_start=str(origins.min()),train_end=str(origins.max()),cutoff=CUTOFFS[period],maturity_hours=144,
                target_rule='prepared training-y causal ffill; remaining leading missing targets excluded per head; query truth unused',
                frame_rule='latest admissible actual observation end <= origin; feature ffill without age limit',
                speed_rule='new df v_lag0 at input origin; no future or query target values',
                original_feature_source='git6b85acc0b:research/e2100_tda_pi_suvi-20261031/tda.py',
                clip=[150.,1500.],alpha_grid=ALPHAS,chosen_alphas=chosen,
                inference_seconds_per_fold=timing,end_to_end_seconds=time.perf_counter()-start)
    import importlib.metadata as metadata
    report['versions']={name:metadata.version(name) for name in ['numpy','pandas','scikit-learn','gudhi','h5netcdf']}
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False)
    report['git_commit']=revision.stdout.strip() if revision.returncode==0 else None
    (out/f'{period}_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',default='e2100_tda_pi_suvi72',choices=['e2100_tda_pi_suvi72'])
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--raw-data',type=Path,help='Original df provenance; fitting uses prepared --data only')
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--raw-suvi-dir',default=Path('data-v2/raw/SUVI/nc/g18'),type=Path)
    parser.add_argument('--feature-cache',type=Path)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--period',choices=list(CUTOFFS))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--features-only',action='store_true')
    parser.add_argument('--merge',action='store_true')
    args=parser.parse_args()
    if args.threads<1:parser.error('threads must be positive')
    if args.merge:
        print(json.dumps({'prediction_rows':merge(args.out)}));return
    cache=args.feature_cache or args.out/'image_features'
    max_origin=pd.read_parquet(args.data,columns=['origin'],filters=[('h','==',1)]).origin.max()
    frame=build_cache(args.raw_suvi_dir,cache,args.threads,pd.Timestamp(max_origin),limit=2 if args.smoke else None)
    if args.features_only:
        return
    if args.smoke:
        print('Smoke extracts only two images; predictions remain smoke-only.',flush=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        for period in [args.period] if args.period else (['F1'] if args.smoke else list(CUTOFFS)):
            print(json.dumps(run_period(args.data,args.out,period,frame,args.smoke)),flush=True)
    if not args.period and not args.smoke:
        print(json.dumps({'prediction_rows':merge(args.out)}))


if __name__=='__main__':
    main()
