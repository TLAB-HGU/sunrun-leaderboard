"""Fresh E1466 PFSS/WSA/HUXt physics from causal original 6-hour GONG maps.

Original mathematical kernels: git d55d9f6d6 extract_pfss.py and archived
E1466 tdwsahuxt_state.py. No old HUXt forecasts or learned predictions are read.
Map size, FITS header, source path, publication receipt and explicit configuration
identify caches. No file hashes are computed or verified.
Run under gong_env/.venv/bin/python. Full generation is an explicit CLI stage.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from email.utils import parsedate_to_datetime
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import time
import urllib.request
import uuid
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key,"1")
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT/'leaderboard/experiments/solar_enablephys_boostmhd72_20261015'
INDEX=ROOT/'leaderboard/experiments/solar_enabledata_gongreceiptaudit_20261015/byte_receipt_index.parquet'
COLS=['huxt_max','huxt_argmax','huxt_h1','huxt_h24','huxt_h48','huxt_h72']
RSS,NR,LON_STEP_DEG=2.5,15,10
PHOTO_LAT_DEG=(-10.,0.,10.)
FS_MIN,FS_MAX,THETA_MAX=1.,300.,30.
WSA_V0,WSA_DV,WSA_FS_EXP,WSA_TB_SCALE=267.5,410.,-.4,2.
V_MIN,V_MAX=240.,800.
HOUR_NS=3_600_000_000_000
NS=1_000_000_000
R_MIN_RSUN,R_MAX_RSUN,FUTURE_HOURS=21.5,215.,72
_WORKER={}
PROFILE_CONFIG=dict(source='gitd55d9f6d6 extract_pfss.map_profiles',raw_shape=[180,360],coarse_shape=[18,36],rss=RSS,nr=NR,
                    longitude_step=LON_STEP_DEG,photosphere_latitudes=list(PHOTO_LAT_DEG),atol=1e-4,rtol=1e-4,
                    wsa=[WSA_V0,WSA_DV,WSA_FS_EXP,WSA_TB_SCALE,V_MIN,V_MAX])


def _endpoint_ok(line) -> bool:
    if len(line) <= 1:
        return False
    r = np.asarray(line._r, dtype=float)
    return (min(abs(r[0] - 1.0), abs(r[0] - RSS)) < 0.05
            and min(abs(r[-1] - 1.0), abs(r[-1] - RSS)) < 0.05)

def _great_circle_deg(a_lon, a_lat, b_lon, b_lat) -> np.ndarray:
    a_lon = np.deg2rad(float(a_lon))
    a_lat = np.deg2rad(float(a_lat))
    b_lon = np.deg2rad(np.asarray(b_lon, dtype=float))
    b_lat = np.deg2rad(np.asarray(b_lat, dtype=float))
    cos_d = np.sin(a_lat) * np.sin(b_lat) + np.cos(a_lat) * np.cos(b_lat) * np.cos(a_lon - b_lon)
    return np.rad2deg(np.arccos(np.clip(cos_d, -1.0, 1.0)))

def wsa_speed(fs: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    fs = np.clip(np.asarray(fs, dtype=float), FS_MIN, FS_MAX)
    theta_b = np.clip(np.asarray(theta_b, dtype=float), 0.0, THETA_MAX)
    speed = WSA_V0 + WSA_DV * np.power(fs, WSA_FS_EXP) * np.power(
        1.0 - 0.8 * np.exp(-np.square(theta_b / WSA_TB_SCALE)), 3.0)
    return np.clip(speed, V_MIN, V_MAX)

def map_profiles(record: dict) -> dict:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import sunpy.map
    import pfsspy
    from pfsspy.tracing import PythonTracer

    path = Path(record["path"])
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{record['file']}: byte length differs from receipt")

    source_map = sunpy.map.Map(str(path))
    raw = np.asarray(source_map.data, dtype=float)
    if raw.size != 180 * 360:
        raise ValueError(f"{record['file']}: expected 180x360 GONG map, got {raw.shape}")
    coarse = raw.reshape(18, 10, 36, 10).mean((1, 3))
    header = pfsspy.utils.carr_cea_wcs_header(
        record["date_obs_utc"], coarse.T.shape,
        map_center_longitude=float(record["crval1"]) * u.deg)
    gong_map = sunpy.map.Map(coarse - coarse.mean(), header)
    solution = pfsspy.pfss(pfsspy.Input(gong_map, NR, RSS))
    tracer = PythonTracer(atol=1e-4, rtol=1e-4)

    longitude = np.arange(0.0, 360.0, LON_STEP_DEG)
    photo_lon = np.tile(longitude, len(PHOTO_LAT_DEG))
    photo_lat = np.repeat(np.asarray(PHOTO_LAT_DEG), len(longitude))
    photo_seeds = SkyCoord(photo_lon * u.deg, photo_lat * u.deg,
                           1.001 * u.R_sun, frame=solution.coordinate_frame)
    photo_lines = tracer.trace(photo_seeds, solution)
    if not all(_endpoint_ok(line) for line in photo_lines):
        raise ValueError(f"{record['file']}: photospheric trace endpoint failed")
    photo_open = np.asarray([line.is_open for line in photo_lines], dtype=bool)
    closed = photo_seeds[~photo_open]

    source_seeds = SkyCoord(longitude * u.deg, np.zeros(len(longitude)) * u.deg,
                            (RSS - 0.001) * u.R_sun, frame=solution.coordinate_frame)
    source_lines = tracer.trace(source_seeds, solution)
    if not all(_endpoint_ok(line) for line in source_lines):
        raise ValueError(f"{record['file']}: source-surface trace endpoint failed")

    fs = np.full(len(longitude), np.nan, dtype=np.float64)
    theta = np.full(len(longitude), np.nan, dtype=np.float64)
    for i, line in enumerate(source_lines):
        if not line.is_open or not np.isfinite(line.expansion_factor):
            continue
        foot = line.solar_footpoint
        expansion = float(np.clip(line.expansion_factor, FS_MIN, FS_MAX))
        if len(closed):
            closed_lon = closed.lon.to_value(u.deg)
            closed_lat = closed.lat.to_value(u.deg)
            boundary = float(_great_circle_deg(foot.lon.to_value(u.deg),
                                               foot.lat.to_value(u.deg),
                                               closed_lon, closed_lat).min())
        else:
            boundary = THETA_MAX
        fs[i] = expansion
        theta[i] = float(np.clip(boundary, 0.0, THETA_MAX))
    valid = np.isfinite(fs) & np.isfinite(theta)
    if int(valid.sum()) < 6:
        raise ValueError(f"{record['file']}: only {int(valid.sum())}/36 usable open-source profiles")
    speed = np.full(len(longitude), np.nan, dtype=np.float64)
    speed[valid] = wsa_speed(fs[valid], theta[valid])
    return {"longitude_deg": longitude, "fs": fs, "theta_b_deg": theta,
            "v_wsa_kms": speed, "open_source_count": int(valid.sum()),
            "open_photosphere_count": int(photo_open.sum())}

def _new_model(init_ns: int, boundary: np.ndarray, simtime_h: float):
    import astropy.units as u
    from astropy.time import Time
    from sunpy.coordinates import sun
    import huxt.huxt as H

    init = pd.Timestamp(int(init_ns), tz="UTC")
    time_origin = Time(init.to_pydatetime())
    cr = float(sun.carrington_rotation_number(time_origin))
    cr_int = int(math.floor(cr))
    cr_lon = 360.0 * (1.0 - (cr - cr_int))
    model = H.HUXt(
        v_boundary=np.asarray(boundary, dtype=np.float64) * H.km_per_s,
        cr_num=cr_int, cr_lon_init=cr_lon * u.deg,
        r_min=R_MIN_RSUN * u.solRad, r_max=R_MAX_RSUN * u.solRad,
        simtime=(simtime_h * u.hour).to(u.day), dt_scale=1,
        frame="synodic", track_cmes=False,
    )
    init_stamp = pd.Timestamp(model.time_init.datetime)
    if init_stamp.tzinfo is None:
        init_stamp = init_stamp.tz_localize("UTC")
    else:
        init_stamp = init_stamp.tz_convert("UTC")
    actual_init_ns = init_stamp.value
    return H, model, actual_init_ns

def _standard_buffer(model, speed: np.ndarray) -> tuple[int, float]:
    import astropy.units as u
    import huxt.huxt as H
    min_speed = float(np.min(speed))
    if not math.isfinite(min_speed) or min_speed <= 0:
        raise ValueError("causal WSA boundary minimum speed must be positive")
    travel = 1.5 * (model.rrel[-1] / (min_speed * H.km_per_s)).to_value(u.s)
    dt_s = float(model.dt.to_value(u.s))
    steps = int(np.fix(travel / dt_s))
    if steps < 1:
        raise ValueError("HUXt characteristic spin-up time is shorter than one native step")
    return steps, dt_s

def _rotated_profiles(model, speeds: np.ndarray) -> np.ndarray:
    import huxt.huxt as H
    lons = np.asarray(model.v_boundary_lons.value, dtype=np.float64)
    shifted = H.zerototwopi((model.v_boundary_lons - model.cr_lon_init).value)
    order = np.argsort(shifted)
    result = np.empty_like(speeds, dtype=np.float64)
    for i, profile in enumerate(speeds):
        result[i] = np.interp(lons, shifted[order], profile[order], period=model.twopi)
    return result

def _boundary_matrix(model, model_times_s: np.ndarray, selected_profile: np.ndarray,
                     rotated_profiles: np.ndarray, simtime_s: float) -> np.ndarray:
    """Apply HUXt's ts_from_vlong longitude/time mapping to a changing profile index."""
    import astropy.units as u
    import huxt.huxt as H

    times = np.asarray(model_times_s, dtype=np.float64)
    selected = np.asarray(selected_profile, dtype=np.int32)
    if len(times) != len(selected):
        raise ValueError("time-dependent boundary profile indices do not align with HUXt model time")
    dt_s = float(model.dt.to_value(u.s))
    buffer_s = float(times[0] * -1.0)
    rotation_s = float(model.rotation_period.to_value("s"))
    simlon = float(model.twopi * simtime_s / rotation_s)
    bufferlon = float(model.twopi * buffer_s / rotation_s)
    dlondt = float(model.twopi * dt_s / rotation_s)
    lons_out = np.asarray(model.lon.value, dtype=np.float64)
    lon_start = lons_out - simlon - dlondt
    lon_stop = lons_out + bufferlon
    ntime = len(times)
    lon_step = (lon_stop - lon_start) / ntime
    boundary_lons = np.asarray(model.v_boundary_lons.value, dtype=np.float64)
    output = np.empty((ntime, len(lons_out)), dtype=np.float32)
    row_numbers = np.arange(ntime, dtype=np.int64)
    input_point = ntime - 1 - row_numbers  # HUXt ts_from_vlong applies flipud.
    for first in range(0, ntime, 512):
        last = min(first + 512, ntime)
        rows = row_numbers[first:last]
        points = input_point[first:last, None]
        longitudes = H.zerototwopi(lon_start[None, :] + points * lon_step[None, :])
        for profile_id in np.unique(selected[first:last]):
            if profile_id < 0 or profile_id >= len(rotated_profiles):
                raise ValueError("no causal SW-012 PFSS/WSA profile covers a HUXt boundary timestep")
            local = np.flatnonzero(selected[first:last] == profile_id)
            sample = longitudes[local]
            output[first + local] = np.interp(
                sample.ravel(), boundary_lons, rotated_profiles[int(profile_id)], period=model.twopi
            ).reshape(len(local), len(lons_out))
    if not np.isfinite(output).all() or np.any(output <= 0):
        raise ValueError("time-dependent HUXt boundary contains invalid speeds")
    return output

def _boundary_at_time(model, relative_time_s: float, profile: np.ndarray) -> np.ndarray:
    """Evaluate the HUXt longitude/time mapping at one causal substep boundary."""
    import astropy.units as u
    import huxt.huxt as H

    times = np.asarray(model.model_time.to_value(u.s), dtype=np.float64)
    if relative_time_s < times[0] or relative_time_s > times[-1]:
        raise ValueError("causal substep lies outside the standard HUXt model time support")
    dt_s = float(model.dt.to_value(u.s))
    rotation_s = float(model.rotation_period.to_value(u.s))
    buffer_s = float(-times[0])
    simtime_s = float(model.simtime.to_value(u.s))
    simlon = float(model.twopi * simtime_s / rotation_s)
    bufferlon = float(model.twopi * buffer_s / rotation_s)
    dlondt = float(model.twopi * dt_s / rotation_s)
    lons_out = np.asarray(model.lon.value, dtype=np.float64)
    lon_start = lons_out - simlon - dlondt
    lon_stop = lons_out + bufferlon
    lon_step = (lon_stop - lon_start) / len(times)
    row = (relative_time_s - times[0]) / dt_s
    input_point = len(times) - 1 - row
    longitudes = H.zerototwopi(lon_start + input_point * lon_step)
    rotated = _rotated_profiles(model, np.asarray(profile, dtype=np.float64)[None, :])[0]
    boundary_lons = np.asarray(model.v_boundary_lons.value, dtype=np.float64)
    boundary = np.interp(longitudes, boundary_lons, rotated, period=model.twopi)
    if not np.isfinite(boundary).all() or np.any(boundary <= 0):
        raise ValueError("causal fractional HUXt boundary contains invalid speeds")
    return boundary

def _fractional_step(state: np.ndarray, boundary: np.ndarray, model, duration_s: float) -> None:
    import astropy.units as u
    import huxt.huxt as H

    if duration_s <= 0:
        return
    native_dt = float(model.dt.to_value(u.s))
    if duration_s > native_dt * (1.0 + 1e-8):
        raise ValueError("causal HUXt fractional step exceeds one native step")
    params = model.model_params
    dtdr = float(params[0]) * (duration_s / native_dt)
    alpha, r_accel = float(params[1]), float(params[2])
    rrel = np.asarray(model.rrel.value, dtype=np.float64)
    for lon in range(model.nlon):
        state[lon, 0] = boundary[lon]
        state[lon, 1:] = H._upwind_step_(state[lon, 1:].copy(), state[lon, :-1].copy(),
                                         dtdr, alpha, r_accel, rrel)

def _step_function():
    from numba import njit

    @njit(cache=True)
    def run(boundary, rrel, dtdr, alpha, r_accel, save_ceil, save_floor,
            fractions, output):
        ntime, nlon = boundary.shape
        nr = len(rrel)
        state = np.full((nlon, nr), 400.0, dtype=np.float64)
        previous = state.copy()
        accel_arg = -rrel[:-1] / r_accel
        accel_arg_p = -rrel[1:] / r_accel
        exp_a = np.exp(accel_arg)
        exp_p = np.exp(accel_arg_p)
        save_index = 0
        for t in range(ntime):
            previous[:, :] = state
            for lon in range(nlon):
                state[lon, 0] = boundary[t, lon]
                for radial in range(1, nr):
                    up = previous[lon, radial]
                    dn = boundary[t, lon] if radial == 1 else previous[lon, radial - 1]
                    new = up - dtdr * up * (up - dn)
                    source = dn / (1.0 + alpha * (1.0 - exp_a[radial - 1]))
                    speed_gain = alpha * source * (exp_a[radial - 1] - exp_p[radial - 1])
                    state[lon, radial] = new + dn * dtdr * speed_gain
            while save_index < len(save_ceil) and save_ceil[save_index] == t:
                if save_floor[save_index] == t:
                    output[save_index] = state.T
                else:
                    f = fractions[save_index]
                    output[save_index] = (1.0 - f) * previous.T + f * state.T
                save_index += 1
            if save_index == len(save_ceil):
                break
        return save_index
    return run

def _forecast_worker_init(state_path: str, origins_path: str, floor_path: str, speed_values: np.ndarray,
                          available_ns: np.ndarray, updates: list[dict], radial_count: int,
                          longitude_count: int, cache_key: str) -> None:
    _WORKER.update({
        "state": np.load(state_path, mmap_mode="r"),
        "origins": np.load(origins_path, mmap_mode="r"),
        "floor_ns": np.load(floor_path, mmap_mode="r"),
        "speeds": speed_values, "available_ns": available_ns, "updates": updates,
        "nr": radial_count, "nlon": longitude_count, "cache_key": cache_key,
    })

def _forecast_one(index: int):
    import astropy.units as u
    import huxt.huxt as H
    index = int(index)
    origin_ns = int(_WORKER["origins"][index])
    available = _WORKER["available_ns"]
    selected = int(np.searchsorted(available, origin_ns, side="right") - 1)
    if selected < 0:
        raise ValueError(f"no SW-012 boundary profile is available at origin index {index}")
    profile = _WORKER["speeds"][selected]
    Hmod, model, model_origin_ns = _new_model(origin_ns, profile, FUTURE_HOURS + 1)
    if abs(model_origin_ns - origin_ns) > 2 * NS:
        raise ValueError("HUXt model origin differs from the requested UTC origin by more than 2 seconds")
    steps, dt_s = _standard_buffer(model, profile)
    model.buffertime = steps * model.dt
    model.ts_from_vlong()
    future_indices = np.flatnonzero(np.asarray(model.model_time.to_value(u.s)) > 0.0)
    if not len(future_indices):
        raise RuntimeError("HUXt did not create a post-origin inner-boundary time series")
    state_override = _WORKER.get("state_override", {})
    floor_state = state_override.get(index, _WORKER["state"][index])
    field0 = np.asarray(floor_state, dtype=np.float64).T.copy()  # longitude x radial at causal native floor
    floor_ns = int(_WORKER["floor_ns"][index])
    if floor_ns > origin_ns or origin_ns - floor_ns > dt_s * NS + 2:
        raise ValueError("cached HUXt state floor is not the nearest native time at or before its origin")
    state = field0
    segment_start = floor_ns
    selected_floor = int(np.searchsorted(available, floor_ns, side="right") - 1)
    if selected_floor < 0:
        raise ValueError("no SW-012 boundary profile is available at the HUXt state floor")
    event_ids = np.flatnonzero((available > floor_ns) & (available <= origin_ns))
    for event_id in event_ids:
        event_ns = int(available[event_id])
        segment_profile = int(np.searchsorted(available, segment_start, side="right") - 1)
        segment_boundary = _boundary_at_time(model, (segment_start - origin_ns) / NS,
                                             _WORKER["speeds"][segment_profile])
        _fractional_step(state, segment_boundary, model, (event_ns - segment_start) / NS)
        segment_start = event_ns
    segment_profile = int(np.searchsorted(available, segment_start, side="right") - 1)
    if origin_ns > segment_start:
        segment_boundary = _boundary_at_time(model, (segment_start - origin_ns) / NS,
                                             _WORKER["speeds"][segment_profile])
        _fractional_step(state, segment_boundary, model, (origin_ns - segment_start) / NS)
    state[:, 0] = _boundary_at_time(model, 0.0, profile)
    state_origin = state.copy()
    params = model.model_params
    dtdr, alpha, r_accel = float(params[0]), float(params[1]), float(params[2])
    boundary = np.asarray(model.input_v_ts[future_indices], dtype=np.float64)
    target_s = np.arange(1, FUTURE_HOURS + 1, dtype=np.float64) * 3600.0
    observer = Hmod.Observer("EARTH", model.time_init + target_s * u.s)
    # HUXt's synodic frame samples Earth at Observer.lon (HEEQ longitude), as
    # huxt_analysis.get_observer_timeseries does for model.frame == "synodic".
    earth_lon = np.asarray(observer.lon.to_value(u.rad), dtype=np.float64)
    lon_grid = np.asarray(model.lon.value, dtype=np.float64)
    lon_delta = (lon_grid[None, :] - earth_lon[:, None] + np.pi) % (2 * np.pi) - np.pi
    earth_lon_index = np.argmin(np.abs(lon_delta), axis=1).astype(np.uint8)
    fields = np.empty((FUTURE_HOURS, model.nr, model.nlon), dtype=np.float16)
    state = state_origin
    previous = state.copy()
    previous_t = 0.0
    next_target = 0
    for row, time_index in enumerate(future_indices):
        current_t = float(model.model_time[time_index].to_value(u.s))
        previous = state.copy()
        for lon in range(model.nlon):
            state[lon, 0] = boundary[row, lon]
            state[lon, 1:] = Hmod._upwind_step_(
                state[lon, 1:].copy(), state[lon, :-1].copy(), dtdr, alpha, r_accel, model.rrel.value
            )
        while next_target < FUTURE_HOURS and target_s[next_target] <= current_t:
            fraction = (target_s[next_target] - previous_t) / (current_t - previous_t)
            fields[next_target] = ((1.0 - fraction) * previous.T + fraction * state.T).astype(np.float16)
            next_target += 1
        previous_t = current_t
    if next_target != FUTURE_HOURS or not np.isfinite(fields).all():
        raise RuntimeError(f"incomplete HUXt lead fields for origin {pd.Timestamp(origin_ns, tz='UTC')}")
    return index, fields, earth_lon_index


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n');os.replace(tmp,path)


def versions():
    return {n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','astropy','sunpy','pfsspy','streamtracer','huxt','numba']}


def inventory(end,download_dir):
    """Original maps/ six-hour inventory, never maps-hourly substitutes."""
    rows=pd.read_parquet(INDEX)
    rows=rows[rows.relative_path.str.startswith('maps/') & (rows.fits_observation_time_utc<=end)]
    sources={}
    for rel in ['source_receipt.json','official_input/source_receipt.json']:
        for row in json.loads((SOURCE_ROOT/rel).read_text())['rows']:
            if row.get('pinned_path') and Path(row['pinned_path']).is_file():
                sources[Path(row['file']).name]=row
    result=[]
    for row in rows.sort_values('fits_observation_time_utc').to_dict('records'):
        filename=Path(row['relative_path']).name
        source=sources.get(filename,{})
        path=Path(source.get('pinned_path',Path(download_dir)/row['relative_path']))
        obs=pd.Timestamp(row['fits_observation_time_utc'])
        modified=source.get('http_last_modified_utc') or row.get('selected_http_last_modified_utc')
        modified=None if pd.isna(modified) else pd.Timestamp(modified)
        available=max(obs+pd.Timedelta(hours=2),modified) if modified is not None else obs+pd.Timedelta(hours=2)
        match=re.search(r'mrzqs(\d{6})t',filename)
        month=obs.strftime('%Y%m')
        url=f'https://gong.nso.edu/data/magmap/QR/zqs/{month}/mrzqs{match[1]}/{filename}'
        saved=path.with_suffix(path.suffix+'.receipt.json')
        if saved.exists():
            receipt=json.loads(saved.read_text())
            modified=pd.Timestamp(receipt['last_modified_utc']) if receipt.get('last_modified_utc') else modified
            available=max(obs+pd.Timedelta(hours=2),modified) if modified is not None else obs+pd.Timedelta(hours=2)
        result.append(dict(file=filename,path=str(path.resolve()),bytes=int(source.get('bytes') or row['local_bytes']),
                           observation_utc=obs.isoformat(),last_modified_utc=modified.isoformat() if modified is not None else None,
                           available_utc=available.isoformat(),url=url,source_receipt=str(INDEX),exists=path.is_file()))
    if not result:raise ValueError('no original six-hour GONG source inventory')
    return result


def fetch_missing(records):
    """Explicit download stage; preserve provider Last-Modified when supplied."""
    for row in records:
        path=Path(row['path'])
        if path.is_file():continue
        path.parent.mkdir(parents=True,exist_ok=True)
        req=urllib.request.Request(row['url'],headers={'User-Agent':'sunrun-original-GONG-reproduction/1.0'})
        with urllib.request.urlopen(req,timeout=60) as response:
            body=response.read();modified=response.headers.get('Last-Modified')
        if len(body)!=row['bytes']:raise ValueError(f"source size changed for {row['file']}")
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(body);os.replace(tmp,path)
        if modified:
            now=pd.Timestamp(parsedate_to_datetime(modified))
            recorded=row['last_modified_utc']
            # A newer server rewrite must delay availability, never silently backdate it.
            when=max(now,pd.Timestamp(recorded)) if recorded else now
            row['last_modified_utc']=when.isoformat()
            row['available_utc']=max(pd.Timestamp(row['observation_utc'])+pd.Timedelta(hours=2),when).isoformat()
        row['exists']=True
        write_json(path.with_suffix(path.suffix+'.receipt.json'),row)
        print(json.dumps({'downloaded':row['file'],'bytes':len(body)}),flush=True)
    return records


def source_record(row):
    from astropy.io import fits
    path=Path(row['path'])
    if not path.is_file() or path.stat().st_size!=row['bytes']:raise ValueError('missing/changed original map '+row['file'])
    with fits.open(path,memmap=False) as hdus:
        header=hdus[0].header
        if hdus[0].shape!=(180,360):raise ValueError('unexpected GONG map shape')
        date=header.get('DATE-OBS')
        timestamp=header.get('DATE')
        crval1=float(header['CRVAL1'])
    if not date:raise ValueError('missing FITS DATE-OBS')
    if timestamp:
        header_time=pd.Timestamp(timestamp)
        if header_time.tz is None:header_time=header_time.tz_localize('UTC')
        if header_time!=pd.Timestamp(row['observation_utc']):raise ValueError('FITS observation timestamp disagrees with receipt')
    # E1466 prepare_sources passes the complete observation instant to the PFSS header.
    return dict(row,date_obs_utc=row['observation_utc'].replace('+00:00',''),crval1=crval1,mtime_ns=path.stat().st_mtime_ns)


def profile_identity(row):
    return dict(file=row['file'],path=row['path'],bytes=row['bytes'],mtime_ns=row['mtime_ns'],
                observation_utc=row['observation_utc'],available_utc=row['available_utc'],
                date_obs_utc=row['date_obs_utc'],crval1=row['crval1'],config=PROFILE_CONFIG,versions=versions())


def _profile_child(row, channel):
    try:
        channel.send((map_profiles(row),None))
    except Exception as exc:
        channel.send((None,f"{type(exc).__name__}: {exc}"))
    finally:
        channel.close()


def calculate_profile(row,timeout=120):
    context=mp.get_context('fork')
    receive,send=context.Pipe(duplex=False)
    process=context.Process(target=_profile_child,args=(row,send))
    process.start();send.close()
    try:
        if receive.poll(timeout):
            profile,error=receive.recv()
            process.join(5)
            if error:raise RuntimeError(error)
            return profile,False
        process.terminate();process.join(5)
        if process.is_alive():process.kill();process.join()
        # Preserve the original 120s timeout rule: a missing map never invents a boundary.
        return dict(longitude_deg=np.arange(0.,360.,10.),fs=np.full(36,np.nan),theta_b_deg=np.full(36,np.nan),
                    v_wsa_kms=np.full(36,np.nan),open_source_count=0,open_photosphere_count=0),True
    finally:
        receive.close()
        if process.is_alive():process.terminate();process.join()
        process.close()


def legacy_inventory():
    manifest=json.loads((SOURCE_ROOT/'profile_manifest.json').read_text())
    sources={Path(r['file']).name:r for r in json.loads((SOURCE_ROOT/'source_receipt.json').read_text())['rows']}
    return {Path(r['file']).name:dict(r,source=sources.get(Path(r['file']).name)) for r in manifest['maps']}


def legacy_profile(row,legacy):
    """Require identical raw path, byte count, timing receipt and physics config."""
    source=legacy.get('source') or {}
    if (not source.get('pinned_path') or Path(source['pinned_path']).resolve()!=Path(row['path']).resolve()
            or int(source['bytes'])!=row['bytes']
            or pd.Timestamp(source['observation_time_utc'])!=pd.Timestamp(row['observation_utc'])
            or pd.Timestamp(source['available_at_utc'])!=pd.Timestamp(row['available_utc'])):
        raise ValueError('legacy PFSS source/availability differs')
    if legacy.get('timed_out'):raise ValueError('legacy timeout is not a verified numerical profile')
    with np.load(legacy['cache_path'],allow_pickle=False) as z:
        cfg=json.loads(str(z['identity']))
        expected=dict(rss=RSS,nr=NR,longitude_grid_deg=list(range(0,360,10)),photosphere_latitude_seeds_deg=list(PHOTO_LAT_DEG),
                      wsa=[WSA_V0,WSA_DV,WSA_FS_EXP,WSA_TB_SCALE,V_MIN,V_MAX],
                      boundary_distance='nearest traced closed photospheric seed, great-circle degrees')
        if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('legacy PFSS configuration differs')
        return {k:z[k] for k in ['longitude_deg','fs','theta_b_deg','v_wsa_kms','open_source_count','open_photosphere_count']}


def verify_legacy(records,cache):
    legacy=legacy_inventory();proof=[]
    for row in records:
        if row['file'] not in legacy or not row['exists']:continue
        if len(proof)==2:break
        checked=source_record(row)
        expected=legacy_profile(checked,legacy[row['file']])
        fresh=profile_one((row,str(Path(cache)/'verification-fresh')))
        with np.load(fresh['cache_path'],allow_pickle=False) as z:
            for key in ['longitude_deg','fs','theta_b_deg','v_wsa_kms']:
                if not np.array_equal(expected[key],z[key],equal_nan=True):raise ValueError('legacy/fresh numerical mismatch '+key)
        proof.append(dict(file=row['file'],source=profile_identity(checked),numeric_equality='exact; longitude/fs/theta/speed'))
    if len(proof)<2:raise ValueError('two actual raw-map recomputations are required')
    receipt=dict(config=PROFILE_CONFIG,versions=versions(),maps=proof,passed=True)
    write_json(Path(cache)/'legacy-verification.json',receipt)
    return receipt


def profile_one(task):
    row,cache,*optional=task
    row=source_record(row)
    identity=profile_identity(row)
    out=Path(cache)/(row['file']+'.npz');out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        with np.load(out,allow_pickle=False) as z:
            if json.loads(str(z['identity']))==identity:
                return dict(row,cache_path=str(out),cached=True)
    begin=time.perf_counter()
    reused=False
    try:
        if optional and optional[0] is not None:
            profile=legacy_profile(row,optional[0]);timed_out=False;reused=True
        else:
            profile,timed_out=calculate_profile(row)
    except Exception as exc:
        write_json(out.with_suffix('.error.json'),dict(identity=identity,error=f'{type(exc).__name__}: {exc}'))
        raise
    tmp=out.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,identity=json.dumps(identity,sort_keys=True),timed_out=timed_out,**profile)
    os.replace(tmp,out)
    return dict(row,cache_path=str(out),cached=False,reused_verified_legacy=reused,timed_out=timed_out,seconds=time.perf_counter()-begin)


def build_profiles(records,cache,workers=1,verification=None):
    legacy={}
    if verification:
        proof=json.loads(Path(verification).read_text())
        if not proof.get('passed') or len(proof.get('maps',[]))<2 or proof['config']!=PROFILE_CONFIG or proof['versions']!=versions():
            raise ValueError('legacy verification does not match this environment/configuration')
        for item in proof['maps']:
            current=source_record(next(r for r in records if r['file']==item['file']))
            if profile_identity(current)!=item['source']:raise ValueError('verified raw map changed')
        legacy=legacy_inventory()
    tasks=[]
    for row in records:
        old=legacy.get(row['file'])
        if old:
            try:legacy_profile(source_record(row),old)
            except (OSError,ValueError,KeyError):old=None
        tasks.append((row,str(cache),old))
    result=[]
    if workers==1:
        for task in tasks:
            result.append(profile_one(task));print(json.dumps({'profile':result[-1]['file'],'completed':len(result),'total':len(tasks)}),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(profile_one,tasks,chunksize=1):
                result.append(row)
                if len(result)%10==0:print(json.dumps({'profiles':len(result),'total':len(tasks)}),flush=True)
    write_json(Path(cache)/'manifest.json',dict(rows=result,config=PROFILE_CONFIG,versions=versions(),old_cache_reused=bool(verification)))
    return result


def load_profile_manifest(records,cache):
    manifest=json.loads((Path(cache)/'manifest.json').read_text())
    if manifest['config']!=PROFILE_CONFIG or manifest['versions']!=versions():
        raise ValueError('profile cache environment/configuration changed')
    rows=manifest['rows'];current={r['file']:r for r in records}
    if set(current)!={r['file'] for r in rows}:raise ValueError('profile cache source inventory changed')
    for row in rows:
        record=current[row['file']]
        for key in ['path','bytes','observation_utc','available_utc']:
            if row[key]!=record[key]:raise ValueError('profile source identity changed: '+row['file'])
        stat=Path(record['path']).stat()
        if stat.st_size!=row['bytes'] or stat.st_mtime_ns!=row['mtime_ns']:
            raise ValueError('raw profile source changed: '+row['file'])
        if not Path(row['cache_path']).is_file():raise ValueError('profile file is missing: '+row['file'])
    return rows


def causal_profiles(rows):
    last=np.full(36,np.nan);latest=np.full(36,np.iinfo(np.int64).min,dtype=np.int64)
    updates=[];speeds=[]
    for row in sorted(rows,key=lambda r:(r['available_utc'],r['observation_utc'],r['file'])):
        obs=pd.Timestamp(row['observation_utc']).value
        with np.load(row['cache_path'],allow_pickle=False) as z:values=z['v_wsa_kms']
        good=np.isfinite(values)&(values>0)&(obs>=latest)
        last[good]=values[good];latest[good]=obs
        if good.any() and np.isfinite(last).all():
            updates.append(dict(file=row['file'],observation_utc=row['observation_utc'],available_utc=row['available_utc'],observation_ns_by_bin=latest.tolist()))
            speeds.append(last.copy())
    if not speeds:raise ValueError('no complete causal PFSS longitude boundary')
    return updates,np.array([pd.Timestamp(r['available_utc']).value for r in updates],np.int64),np.asarray(speeds)


def summarize(traj):
    traj=np.asarray(traj)
    if traj.shape[-1]!=72 or not np.isfinite(traj).all():raise ValueError('invalid 72h Earth trajectory')
    return np.stack([traj.max(axis=-1),traj.argmax(axis=-1)+1,traj[...,0],traj[...,23],traj[...,47],traj[...,71]],axis=-1)


def build_history(requested,available_ns,speeds,cache):
    """Original E1466 global causal finite-volume history, freshly integrated."""
    import astropy.units as u
    import huxt.huxt as H
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    requested=np.unique(np.asarray(requested,np.int64))
    first=speeds[0];base=int(available_ns[0])+HOUR_NS
    _,model,actual=_new_model(base,first,FUTURE_HOURS+1)
    steps,dt_s=_standard_buffer(model,first);buffer_s=steps*dt_s
    increment=int(round(dt_s*NS))
    jump=max(0,math.ceil((int(available_ns[0])-(actual-int(round(buffer_s*NS))))/increment))
    # Same integer native-step lattice and minimal supported initial time as the
    # original linear while loop, without reconstructing thousands of models.
    while True:
        nominal=base+jump*increment
        _,model,actual=_new_model(nominal,first,FUTURE_HOURS+1)
        steps,new_dt=_standard_buffer(model,first)
        if new_dt!=dt_s:raise ValueError('HUXt native timestep changed during alignment')
        buffer_s=steps*dt_s
        if actual-int(round(buffer_s*NS))<int(available_ns[0]):
            jump+=1;continue
        if jump:
            _,previous,previous_actual=_new_model(base+(jump-1)*increment,first,FUTURE_HOURS+1)
            previous_steps,previous_dt=_standard_buffer(previous,first)
            if previous_actual-int(round(previous_steps*previous_dt*NS))>=int(available_ns[0]):
                jump-=1;continue
        break
    model.buffertime=steps*model.dt
    start=actual-int(round(buffer_s*NS));duration_s=(int(requested[-1])-actual)/NS
    if duration_s<=0:raise ValueError('requested origins precede causal spin-up')
    future_steps=int(np.floor(duration_s/dt_s))
    relative=np.arange(-steps,future_steps+1,dtype=np.float64)*dt_s
    absolute=start+np.rint(np.arange(len(relative))*dt_s*NS).astype(np.int64)
    ids=np.searchsorted(available_ns,absolute,side='right')-1
    if (ids<0).any():raise ValueError('missing causal spin-up boundary')
    rotated=_rotated_profiles(model,speeds)
    boundary=_boundary_matrix(model,relative,ids.astype(np.int32),rotated,duration_s)
    supported=requested[requested>=actual]
    floors=np.floor((supported-start)/(dt_s*NS)).astype(np.int64)
    floor_ns=start+np.rint(floors*dt_s*NS).astype(np.int64)
    history_min=np.minimum.accumulate(np.min(speeds,axis=1))[np.searchsorted(available_ns,floor_ns,side='right')-1]
    required=np.fix((1.5*(model.rrel[-1]/(history_min*H.km_per_s))).to_value(u.s)/dt_s).astype(np.int64)
    good=(floor_ns-start)>=np.rint(required*dt_s*NS).astype(np.int64)
    supported,floors,floor_ns=supported[good],floors[good],floor_ns[good]
    if not len(supported):raise ValueError('no supported origins after characteristic spin-up')
    identity=dict(config=PROFILE_CONFIG,available_ns=available_ns.tolist(),speeds=speeds.tolist(),origins=supported.tolist(),
                  versions=versions(),r_min=R_MIN_RSUN,r_max=R_MAX_RSUN,dt_s=dt_s,start_ns=start)
    receipt=cache/'history.json';state_path=cache/'state_at_origin.npy'
    if receipt.exists() and state_path.exists() and json.loads(receipt.read_text())==identity:
        stored=np.load(state_path,mmap_mode='r')
        if stored.shape==(len(supported),model.nr,model.nlon) and np.isfinite(stored).all():
            return supported,floor_ns,state_path,model
    # Atomic promotion: interrupted integrations cannot masquerade as completed history.
    tmp=cache/'state_in_progress.npy'
    output=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(supported),model.nr,model.nlon))
    params=model.model_params
    count=_step_function()(boundary,np.asarray(model.rrel.value,np.float64),float(params[0]),float(params[1]),float(params[2]),
                           floors,floors,np.zeros(len(floors)),output)
    if count!=len(supported):raise RuntimeError('incomplete HUXt history integration')
    target=cache/'state_float16.tmp.npy'
    final=np.lib.format.open_memmap(target,mode='w+',dtype=np.float16,shape=output.shape)
    for i in range(0,len(output),128):final[i:i+128]=output[i:i+128]
    final.flush();del final,output;os.replace(target,state_path);tmp.unlink()
    np.save(cache/'origins_ns.npy',supported);np.save(cache/'state_floor_ns.npy',floor_ns)
    write_json(receipt,identity)
    return supported,floor_ns,state_path,model


def forecast_summary(index):
    index,fields,earth=_forecast_one(index)
    # Preserve original float16 field quantization before float32 Earth extraction.
    ring=fields[:,-1,:]
    trajectory=np.take_along_axis(ring,earth[:,None].astype(np.int64),axis=1)[:,0].astype(np.float32)
    return index,trajectory,summarize(trajectory)


def build_forecasts(requested,updates,available,speeds,out,workers=1):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    supported,floors,state,model=build_history(requested,available,speeds,out/'history')
    init=(str(state),str(out/'history/origins_ns.npy'),str(out/'history/state_floor_ns.npy'),speeds,available,updates,model.nr,model.nlon,'fresh-reproduction')
    # Each origin is independently durable; full lead fields need not be retained.
    cache=out/'trajectories';cache.mkdir(exist_ok=True)
    identity=json.loads((out/'history/history.json').read_text())
    identity_path=cache/'identity.json'
    prior=json.loads(identity_path.read_text()) if identity_path.exists() else {}
    generation=prior['generation'] if prior.get('identity')==identity else uuid.uuid4().hex
    write_json(identity_path,dict(identity=identity,generation=generation))
    # Every partial result carries this source/config generation for safe resume.
    values=np.full((len(supported),6),np.nan);todo=[]
    for i,origin in enumerate(supported):
        path=cache/f'{origin}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                # History identity is checked before reuse; per-origin cache stores current source index.
                selected=int(np.searchsorted(available,origin,side='right')-1)
                if 'generation' in z and str(z['generation'])==generation and int(z['origin'])==origin and int(z['selected_available_ns'])==available[selected]:
                    values[i]=summarize(z['trajectory']);continue
        todo.append(i)
    def save(result):
        i,traj,six=result;origin=int(supported[i]);selected=int(np.searchsorted(available,origin,side='right')-1)
        path=cache/f'{origin}.npz';tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,origin=origin,trajectory=traj,selected_available_ns=available[selected],generation=generation)
        os.replace(tmp,path);values[i]=six
    if workers==1:
        _forecast_worker_init(*init)
        for i in todo:save(forecast_summary(i))
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=_forecast_worker_init,initargs=init) as pool:
            for done,result in enumerate(pool.map(forecast_summary,todo,chunksize=1),1):
                save(result)
                if done%100==0:print(json.dumps({'physics_forecasts_completed':done,'total':len(todo)}),flush=True)
    if not np.isfinite(values).all():raise RuntimeError('nonfinite HUXt summaries')
    frame=pd.DataFrame(values,columns=COLS);frame.insert(0,'origin',pd.DatetimeIndex(supported,tz='UTC'))
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE/'input/huxt_features.parquet')
    parser.add_argument('--cache',type=Path,default=HERE/'gong_cache')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--origin-start',help='Optional requested-origin lower bound; source history remains complete')
    parser.add_argument('--legacy-verification',type=Path)
    parser.add_argument('--stage',choices=['preflight','download','verify-legacy','profiles','forecast','all','smoke'],default='preflight')
    args=parser.parse_args()
    if not 1<=args.threads<=4:parser.error('physics workers must be 1..4')
    os.environ['XDG_DATA_HOME']=str(HERE/'gong_env/runtime')
    from astropy.utils import iers
    iers.conf.auto_download=False
    dates=pd.read_parquet(args.data,columns=['origin'],filters=[('h','==',1)])
    origins=pd.DatetimeIndex(pd.to_datetime(dates.origin,utc=True)).unique().sort_values()
    if args.origin_start:
        start=pd.Timestamp(args.origin_start)
        start=start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')
        origins=origins[origins>=start]
    records=inventory(origins.max(),args.cache/'raw')
    report=dict(stage=args.stage,original_6h_maps=len(records),existing=sum(r['exists'] for r in records),missing=sum(not r['exists'] for r in records),
                requested_origins=len(origins),origin_start=str(origins.min()),origin_end=str(origins.max()),versions=versions(),output_columns=['origin',*COLS])
    print(json.dumps(report),flush=True)
    write_json(args.cache/'preflight.json',report)
    if args.stage=='preflight':return
    if args.stage=='verify-legacy':
        print(json.dumps(verify_legacy(records,args.cache)),flush=True);return
    if args.stage in ['download','all']:
        records=fetch_missing(records);write_json(args.cache/'sources.json',dict(rows=records))
        if args.stage=='download':return
    if args.stage=='smoke':records=[r for r in records if r['exists']][:1]
    if any(not Path(r['path']).is_file() for r in records):raise ValueError('download stage must restore missing original maps first')
    rows=build_profiles(records,args.cache/'profiles',args.threads,args.legacy_verification) if args.stage in ['profiles','all','smoke'] else load_profile_manifest(records,args.cache/'profiles')
    if args.stage=='profiles':return
    updates,available,speeds=causal_profiles(rows)
    if args.stage=='smoke':
        origin=pd.Timestamp(available[0],tz='UTC').ceil('h')+pd.Timedelta(days=20)
        origins=pd.DatetimeIndex([origin])
    frame=build_forecasts(origins.asi8,updates,available,speeds,args.cache/'physics',args.threads)
    if args.stage!='smoke':
        required=origins[origins>=pd.Timestamp('2024-06-01',tz='UTC')]
        if not required.isin(frame.origin).all():raise ValueError('physics lacks required roll6/query coverage')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.out.with_suffix('.parquet.tmp');frame.to_parquet(tmp,index=False);os.replace(tmp,args.out)
    report.update(output=str(args.out.resolve()),rows=len(frame),first_origin=str(frame.origin.min()),last_origin=str(frame.origin.max()),
                  missing_early_spinup_origins=len(origins)-len(frame),source_rule='original6h maps only; max(obs+2h,HTTP Last-Modified)',
                  old_huxt_predictions_used=False,original_sources=['gitd55d9f6d6 extract_pfss.py','archived E1466 tdwsahuxt_state.py'],
                  environment_note='original HUXt package receipt absent; restored official HUXt5.0.3 API with numerical parity tests')
    write_json(args.out.with_suffix('.metadata.json'),report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
