"""Causal SW-012 time-dependent PFSS/WSA boundary history and HUXt fields."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("TDWSAHUXT_SOURCE_DIR", HERE / "cache/source")).resolve()
START_GRID = pd.Timestamp("2022-10-01T00:00:00Z")
END_GRID = pd.Timestamp("2026-09-26T23:00:00Z")
OFFICIAL_START = pd.Timestamp("2026-05-31T23:00:00Z")
OFFICIAL_END = END_GRID
HOUR_NS = 3_600_000_000_000
NS = 1_000_000_000
R_MIN_RSUN = 21.5
R_MAX_RSUN = 215.0
FUTURE_HOURS = 72
_WORKER = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


def _load_profiles():
    source_path = CACHE / "source_manifest.json"
    profile_path = CACHE / "profile_manifest.json"
    source = json.loads(source_path.read_text())
    profile = json.loads(profile_path.read_text())
    if profile.get("source_manifest_sha256") != sha256(source_path):
        raise ValueError("SW-012 PFSS/WSA profiles do not match the item-local source snapshot")
    profile_by_file = {row["file"]: row for row in profile["maps"]}
    updates, speeds = [], []
    last = np.full(36, np.nan, dtype=np.float64)
    last_observation_ns = np.full(36, np.iinfo(np.int64).min, dtype=np.int64)
    for source_row in sorted(source["rows"], key=lambda r: (r["available_utc"], r["observation_utc"], r["file"])):
        observation = pd.Timestamp(source_row["observation_utc"])
        if observation.tzinfo is None:
            observation = observation.tz_localize("UTC")
        else:
            observation = observation.tz_convert("UTC")
        last_modified = source_row.get("last_modified_utc")
        if last_modified:
            modified = pd.Timestamp(last_modified)
            if modified.tzinfo is None:
                modified = modified.tz_localize("UTC")
            else:
                modified = modified.tz_convert("UTC")
        else:
            modified = observation
        expected_available = max(observation + pd.Timedelta(hours=2), modified)
        recorded_available = pd.Timestamp(source_row["available_utc"])
        if recorded_available.tzinfo is None:
            recorded_available = recorded_available.tz_localize("UTC")
        else:
            recorded_available = recorded_available.tz_convert("UTC")
        if recorded_available != expected_available:
            raise ValueError(f"SW-012 availability receipt does not match obs+2h/Last-Modified: {source_row['file']}")
        raw_path = Path(source_row["pinned_path"])
        if not raw_path.is_file() or raw_path.stat().st_size != int(source_row["bytes"]):
            raise ValueError(f"pinned SW-012 source map is missing or has the wrong size: {source_row['file']}")
        if sha256(raw_path) != source_row["sha256"]:
            raise ValueError(f"pinned SW-012 source map SHA mismatch: {source_row['file']}")
        profile_row = profile_by_file.get(source_row["file"])
        has_current_profile = bool(profile_row and not profile_row.get("timed_out") and profile_row.get("cache_path"))
        if profile_row and not profile_row.get("timed_out") and profile_row.get("cache_path"):
            profile_file = Path(profile_row["cache_path"])
            if (profile_row.get("body_sha256") != source_row["sha256"]
                    or not profile_row.get("profile_sha256")
                    or not profile_file.is_file()
                    or sha256(profile_file) != profile_row["profile_sha256"]):
                raise ValueError(f"SW-012 PFSS/WSA profile provenance/hash mismatch: {source_row['file']}")
            with np.load(profile_file, allow_pickle=False) as data:
                current = np.asarray(data["v_wsa_kms"], dtype=np.float64)
            if current.shape != (36,):
                raise ValueError(f"unexpected SW-012 PFSS/WSA profile shape: {source_row['file']}")
            good = np.isfinite(current) & (current > 0)
            # Publication time controls when a map may be used; observation
            # time controls whether it is newer than the value already held
            # for each longitude. Late publication of an old synoptic map
            # must never roll the boundary back to an older observation.
            observation_ns = int(observation.value)
            newer = good & (observation_ns >= last_observation_ns)
            last[newer] = current[newer]
            last_observation_ns[newer] = observation_ns
        else:
            newer = np.zeros_like(last, dtype=bool)
        if has_current_profile and newer.any() and np.isfinite(last).all():
            updates.append({
                "file": source_row["file"], "observation_utc": source_row["observation_utc"],
                "available_utc": source_row["available_utc"], "available_ns": pd.Timestamp(source_row["available_utc"]).value,
                "sha256": source_row["sha256"], "profile_cache_key": profile_row.get("cache_key") if profile_row else None,
                "observation_ns_by_bin": last_observation_ns.tolist(),
            })
            speeds.append(last.copy())
    if not updates:
        raise RuntimeError("no complete causal PFSS/WSA boundary profile is available")
    available_ns = np.asarray([r["available_ns"] for r in updates], dtype=np.int64)
    order = np.lexsort((np.asarray([r["file"] for r in updates]), available_ns))
    updates = [updates[int(i)] for i in order]
    available_ns = available_ns[order]
    speeds = np.asarray(speeds, dtype=np.float64)[order]
    if not np.isfinite(speeds).all() or np.any(speeds <= 0):
        raise ValueError("completed PFSS/WSA history must be finite and positive")
    return source, profile, updates, available_ns, speeds


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


def _history_cache_key(source_sha: str, profile_sha: str, origin_sha: str,
                       huxt_sha: str, ephemeris_sha: str) -> str:
    payload = {
        "source_manifest_sha256": source_sha, "profile_manifest_sha256": profile_sha,
        "requested_origin_grid_sha256": origin_sha, "huxt_module_sha256": huxt_sha,
        "earth_ephemeris_sha256": ephemeris_sha,
        "huxt_version": importlib.metadata.version("huxt"),
        "state_code_sha256": sha256(Path(__file__)), "r_min_rsun": R_MIN_RSUN,
        "r_max_rsun": R_MAX_RSUN, "history_rule": "max(obs+2h, Last-Modified) piecewise-constant; no post-origin boundary",
        "forecast_h": FUTURE_HOURS, "encoder_field": "all HUXt radial x synodic-longitude values at each hourly lead",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def _origin_history_probe(origin_ns: int, updates: list[dict], available_ns: np.ndarray,
                          speeds: np.ndarray, profile_mutation: tuple[int, float] | None = None):
    """Single-origin full HUXt replay used by future-source poison and positive control."""
    import astropy.units as u
    import huxt.huxt as H
    selected = int(np.searchsorted(available_ns, origin_ns, side="right") - 1)
    if selected < 0:
        raise ValueError("poison probe origin has no causal SW-012 boundary profile")
    changed = np.array(speeds, copy=True)
    if profile_mutation is not None:
        profile_id, amount = profile_mutation
        changed[profile_id] = changed[profile_id] + amount
    causal_profiles = changed[:selected + 1]
    _, model, model_origin_ns = _new_model(origin_ns, changed[selected], FUTURE_HOURS + 1)
    steps, dt_s = _standard_buffer(model, causal_profiles[-1])
    model.buffertime = steps * model.dt
    model.ts_from_vlong()
    rel = np.asarray(model.model_time.to_value(u.s), dtype=np.float64)
    abs_ns = model_origin_ns + np.rint(rel * NS).astype(np.int64)
    lookup = available_ns[:selected + 1]
    picked = np.searchsorted(lookup, abs_ns, side="right") - 1
    picked[abs_ns >= model_origin_ns] = selected  # future boundary is held at the last causal map
    if np.any(picked < 0):
        raise ValueError("GONG causal history does not cover the HUXt characteristic support window")
    rotated = _rotated_profiles(model, changed[:selected + 1])
    boundary = _boundary_matrix(model, rel, picked.astype(np.int32), rotated,
                                float(model.simtime.to_value(u.s)))
    # No source mutation after origin can affect this matrix: every future row is held at selected.
    model.input_v_ts = boundary.copy()
    model.input_v_ts_flag = True
    model.solve([])
    origin_row = int(np.argmin(np.abs(np.asarray(model.time_out.to_value(u.s)))))
    state = np.asarray(model.v_grid[origin_row].to_value(H.km_per_s), dtype=np.float64)
    forecast = np.asarray(model.v_grid, dtype=np.float64)
    return boundary, state, forecast, selected


def _poison_probe(out_dir: Path, cache_root: Path, updates, available_ns, speeds,
                  source_manifest_sha: str, model, boundary, model_time_s,
                  selected_at_time, duration_s: float, supported_origins,
                  floor_index, state_path: Path, origins_path: Path,
                  floor_path: Path, field_path: Path, integrator) -> dict:
    """Poison the same state integrator and forecast worker used by production."""
    official_origins = pd.date_range(OFFICIAL_START, OFFICIAL_END, freq="h").as_unit("ns").asi8
    official_pos = np.searchsorted(supported_origins, official_origins)
    official_index = int(official_pos[len(official_pos) // 2])
    official_origin_ns = int(supported_origins[official_index])

    dense = (supported_origins >= pd.Timestamp("2025-01-01T00:00:00Z").value) & (
        supported_origins < pd.Timestamp("2026-05-29T00:00:00Z").value
    )
    dense_candidates = []
    for index in np.flatnonzero(dense):
        origin = int(supported_origins[index])
        selected = int(np.searchsorted(available_ns, origin, side="right") - 1)
        if selected <= 0 or selected >= len(available_ns) - 1:
            continue
        age_s = origin - int(available_ns[selected])
        prior_gap_s = int(available_ns[selected] - available_ns[selected - 1])
        if 0 <= age_s <= 2 * 3600 * NS and 0 < prior_gap_s <= 24 * 3600 * NS:
            dense_candidates.append(index)
    if not dense_candidates:
        raise RuntimeError("no dense causal SW-012 boundary interval is available for a past-map positive control")
    dense_index = min(dense_candidates, key=lambda i: abs(
        int(supported_origins[i]) - pd.Timestamp("2025-07-01T00:00:00Z").value
    ))
    dense_origin_ns = int(supported_origins[dense_index])

    states = np.load(state_path, mmap_mode="r")
    origins = np.load(origins_path, mmap_mode="r")
    floors = np.load(floor_path, mmap_mode="r")
    params = model.model_params
    rrel = np.asarray(model.rrel.value, dtype=np.float64)

    def integrate_to_floor(source_speeds: np.ndarray, index: int) -> np.ndarray:
        rotated = _rotated_profiles(model, source_speeds)
        source_boundary = _boundary_matrix(
            model, model_time_s, selected_at_time.astype(np.int32), rotated, duration_s
        )
        target = np.asarray([floor_index[index]], dtype=np.int64)
        result = np.empty((1, model.nr, model.nlon), dtype=np.float32)
        saved = integrator(source_boundary, rrel, float(params[0]), float(params[1]), float(params[2]),
                           target, target.copy(), np.zeros(1, dtype=np.float64), result)
        if saved != 1:
            raise RuntimeError(f"production HUXt state replay saved {saved}/1 probe origins")
        return result[0]

    official_cached_state = np.asarray(states[official_index], dtype=np.float32)
    official_replayed_state = integrate_to_floor(speeds, official_index)
    if not np.array_equal(official_replayed_state.astype(np.float16), official_cached_state.astype(np.float16)):
        raise RuntimeError("production HUXt integrator does not reproduce the cached official origin state")
    official_future_speeds = np.array(speeds, copy=True)
    official_future_speeds[available_ns > official_origin_ns] += 1.0e6
    official_future_state = integrate_to_floor(official_future_speeds, official_index)
    official_future_state_delta = float(np.max(np.abs(official_replayed_state - official_future_state)))

    _forecast_worker_init(str(state_path), str(origins_path), str(floor_path), speeds,
                          available_ns, updates, model.nr, model.nlon, "poison-probe")
    _, official_field, _ = _forecast_one(official_index)
    with h5py.File(field_path, "r") as handle:
        cached_official_field = np.asarray(handle["fields"][official_index], dtype=np.float16)
    official_cache_delta = float(np.max(np.abs(official_field - cached_official_field)))
    _WORKER["speeds"] = official_future_speeds
    _, official_future_field, _ = _forecast_one(official_index)
    official_future_field_delta = float(np.max(np.abs(official_field - official_future_field)))

    dense_source_id = int(np.searchsorted(available_ns, dense_origin_ns, side="right") - 1)
    dense_prior_id = dense_source_id - 1
    if dense_prior_id < 0:
        raise RuntimeError("dense source positive control has no earlier admissible map")
    dense_cached_state = np.asarray(states[dense_index], dtype=np.float32)
    dense_replayed_state = integrate_to_floor(speeds, dense_index)
    if not np.array_equal(dense_replayed_state.astype(np.float16), dense_cached_state.astype(np.float16)):
        raise RuntimeError("production HUXt integrator does not reproduce the cached positive-control state")
    dense_future_speeds = np.array(speeds, copy=True)
    dense_future_speeds[available_ns > dense_origin_ns] += 1.0e6
    dense_future_state = integrate_to_floor(dense_future_speeds, dense_index)
    dense_future_state_delta = float(np.max(np.abs(dense_replayed_state - dense_future_state)))
    dense_past_speeds = np.array(speeds, copy=True)
    dense_past_speeds[dense_prior_id] += 10.0
    dense_past_state = integrate_to_floor(dense_past_speeds, dense_index)
    dense_past_state_delta = float(np.max(np.abs(dense_replayed_state - dense_past_state)))

    _WORKER["speeds"] = speeds
    _, dense_field, _ = _forecast_one(dense_index)
    with h5py.File(field_path, "r") as handle:
        cached_dense_field = np.asarray(handle["fields"][dense_index], dtype=np.float16)
    dense_cache_delta = float(np.max(np.abs(dense_field - cached_dense_field)))
    _WORKER["speeds"] = dense_future_speeds
    _, dense_future_field, _ = _forecast_one(dense_index)
    dense_future_field_delta = float(np.max(np.abs(dense_field - dense_future_field)))
    _WORKER["speeds"] = dense_past_speeds
    _WORKER["state_override"] = {dense_index: dense_past_state}
    _, dense_past_field, _ = _forecast_one(dense_index)
    _WORKER.pop("state_override", None)
    _WORKER["speeds"] = speeds

    dense_past_field_delta = float(np.max(np.abs(dense_field - dense_past_field)))
    passed = bool(
        official_cache_delta == 0.0 and official_future_state_delta == 0.0
        and official_future_field_delta == 0.0 and dense_cache_delta == 0.0
        and dense_future_state_delta == 0.0 and dense_future_field_delta == 0.0
        and dense_past_state_delta > 0.0 and dense_past_field_delta > 0.0
    )
    if not passed:
        raise RuntimeError("production-path SW-012 future poison or admissible past-map positive control failed")

    fields_path = cache_root / "source_poison_fields.npz"
    np.savez_compressed(
        fields_path, official_origin_ns=np.asarray(official_origin_ns, dtype=np.int64),
        official_base=official_field, official_future=official_future_field,
        dense_origin_ns=np.asarray(dense_origin_ns, dtype=np.int64), dense_base=dense_field,
        dense_future=dense_future_field, dense_past=dense_past_field,
    )
    receipt = {
        "passed": passed, "source_id": "SW-012", "source_manifest_sha256": source_manifest_sha,
        "production_path": "_step_function causal history integration -> _forecast_one causal fractional origin step and HUXt forecast",
        "official_probe_origin_utc": str(pd.Timestamp(official_origin_ns, tz="UTC")),
        "official_future_map_count": int(np.sum(available_ns > official_origin_ns)),
        "official_cached_state_replay_delta_kms": float(np.max(np.abs(official_replayed_state - official_cached_state))),
        "official_cached_field_replay_delta_kms": official_cache_delta,
        "official_future_source_mutation_kms": 1.0e6,
        "official_future_source_state_delta_kms": official_future_state_delta,
        "official_future_source_field_delta_kms": official_future_field_delta,
        "dense_positive_control_origin_utc": str(pd.Timestamp(dense_origin_ns, tz="UTC")),
        "dense_selected_map": updates[dense_source_id]["file"],
        "dense_prior_admissible_map": updates[dense_prior_id]["file"],
        "dense_prior_map_observation_utc": updates[dense_prior_id]["observation_utc"],
        "dense_prior_map_perturbation_kms": 10.0,
        "dense_future_source_state_delta_kms": dense_future_state_delta,
        "dense_future_source_field_delta_kms": dense_future_field_delta,
        "dense_past_source_state_delta_kms": dense_past_state_delta,
        "dense_past_source_field_delta_kms": dense_past_field_delta,
        "encoder_prediction_poison_bundle": str(fields_path),
        "official_targets_read": False,
    }
    digest = hashlib.sha256()
    with fields_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    receipt["encoder_prediction_poison_bundle_sha256"] = digest.hexdigest()
    _dump(cache_root / "source_poison.json", receipt)
    _dump(out_dir / "tdwsahuxt_gong_source_poison.json", receipt)
    (out_dir / "tdwsahuxt_source_poison_fields.npz").write_bytes(fields_path.read_bytes())
    return receipt


def build(out_dir: Path, requested_origins: np.ndarray | None = None) -> dict:
    started = time.monotonic()
    import astropy.units as u
    import huxt.huxt as H

    source_path = CACHE / "source_manifest.json"
    profile_path = CACHE / "profile_manifest.json"
    source, profile, updates, available_ns, speeds = _load_profiles()
    full_origins = pd.date_range(START_GRID, END_GRID, freq="h").as_unit("ns").asi8
    if requested_origins is None:
        requested_origins = full_origins
    requested_origins = np.unique(np.asarray(requested_origins, dtype=np.int64))
    if (not len(requested_origins) or np.any(np.diff(requested_origins) <= 0)
            or requested_origins[0] < full_origins[0] or requested_origins[-1] > full_origins[-1]):
        raise ValueError("requested HUXt origin grid must be nonempty, unique, and inside local SW-012 coverage")
    official_origins = pd.date_range(OFFICIAL_START, OFFICIAL_END, freq="h").as_unit("ns").asi8
    if not np.isin(official_origins, requested_origins).all():
        raise ValueError("the requested HUXt field grid must include all 2,833 official origins")
    origins_sha = hashlib.sha256(requested_origins.tobytes()).hexdigest()
    first = speeds[0]
    nominal_init_ns = int(available_ns[0]) + HOUR_NS
    Hmod, model, actual_init_ns = _new_model(nominal_init_ns, first, FUTURE_HOURS + 1)
    buffer_steps, dt_s = _standard_buffer(model, first)
    buffer_s = buffer_steps * dt_s
    # Align the HUXt origin so its native spin-up begins after the first map is published.
    while actual_init_ns - int(round(buffer_s * NS)) < int(available_ns[0]):
        nominal_init_ns += int(round(dt_s * NS))
        Hmod, model, actual_init_ns = _new_model(nominal_init_ns, first, FUTURE_HOURS + 1)
        buffer_steps, dt_s = _standard_buffer(model, first)
        buffer_s = buffer_steps * dt_s
    model.buffertime = buffer_steps * model.dt
    start_ns = actual_init_ns - int(round(buffer_s * NS))
    end_ns = int(full_origins[-1])
    duration_s = (end_ns - actual_init_ns) / NS
    if duration_s <= 0:
        raise ValueError("HUXt global state timeline ends before its first supported origin")
    n_future_steps = int(np.floor(duration_s / dt_s))
    model_time_s = (np.arange(-buffer_steps, n_future_steps + 1, dtype=np.float64) * dt_s)
    absolute_ns = start_ns + np.rint(np.arange(len(model_time_s), dtype=np.float64) * dt_s * NS).astype(np.int64)
    selected_at_time = np.searchsorted(available_ns, absolute_ns, side="right") - 1
    if np.any(selected_at_time < 0):
        raise ValueError("SW-012 local history does not cover the standard HUXt spin-up support")
    rotated = _rotated_profiles(model, speeds)
    boundary = _boundary_matrix(model, model_time_s, selected_at_time.astype(np.int32), rotated, duration_s)
    supported_origins = requested_origins[requested_origins >= actual_init_ns]
    if not len(supported_origins):
        raise ValueError("no model origins have enough causal SW-012 HUXt support")
    candidate_floor_index = np.floor((supported_origins - start_ns) / (dt_s * NS)).astype(np.int64)
    candidate_floor_ns = start_ns + np.rint(candidate_floor_index * dt_s * NS).astype(np.int64)
    selected_at_floor = np.searchsorted(available_ns, candidate_floor_ns, side="right") - 1
    if np.any(selected_at_floor < 0):
        raise ValueError("SW-012 local history does not cover a requested origin's causal state floor")
    history_minimum = np.minimum.accumulate(np.min(speeds, axis=1))[selected_at_floor]
    characteristic_s = 1.5 * (model.rrel[-1] / (history_minimum * H.km_per_s)).to_value(u.s)
    required_steps = np.fix(characteristic_s / dt_s).astype(np.int64)
    enough_support = (candidate_floor_ns - start_ns) >= np.rint(required_steps * dt_s * NS).astype(np.int64)
    supported_origins = supported_origins[enough_support]
    floor_index = candidate_floor_index[enough_support]
    state_floor_ns = candidate_floor_ns[enough_support]
    fractions = np.zeros(len(supported_origins), dtype=np.float64)
    ceil_index = floor_index.copy()
    if ceil_index[-1] >= len(model_time_s):
        raise ValueError("HUXt history simulation does not reach the final requested origin")

    source_sha = sha256(source_path)
    profile_sha = sha256(profile_path)
    huxt_module = Path(H.__file__)
    huxt_sha = sha256(huxt_module)
    ephemeris_path = Path(H._setup_dirs_()["ephemeris"])
    ephemeris_sha = sha256(ephemeris_path)
    cache_key = _history_cache_key(source_sha, profile_sha, origins_sha, huxt_sha, ephemeris_sha)
    cache_root = out_dir / "tdwsahuxt-state" / f"td-state-{cache_key}"
    cache_root.mkdir(parents=True, exist_ok=True)
    state_path = cache_root / "state_at_origin.npy"
    origins_path = cache_root / "origins_ns.npy"
    floor_path = cache_root / "state_floor_ns.npy"
    # Keep the large state and HDF5 artifacts in this EID's durable run folder
    # so a runtime cleanup of the source tree cannot discard completed work.
    field_path = cache_root / "lead_fields.h5"
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("passed") and manifest.get("cache_key") == cache_key:
            if (sha256(origins_path) != manifest["origins_sha256"]
                    or sha256(floor_path) != manifest["state_floor_sha256"]
                    or sha256(state_path) != manifest["states_sha256"]
                    or sha256(field_path) != manifest["lead_fields_sha256"]):
                raise ValueError("completed HUXt time-dependent field cache changed after its receipt")
            out_dir.mkdir(parents=True, exist_ok=True)
            _dump(out_dir / "tdwsahuxt_source_coverage.json", manifest["coverage"])
            _dump(out_dir / "tdwsahuxt_cache_manifest.json", manifest)
            poison_cache = cache_root / "source_poison.json"
            if sha256(poison_cache) != manifest["source_poison_sha256"]:
                raise ValueError("cached GONG source poison receipt changed after completion")
            fields_cache = cache_root / "source_poison_fields.npz"
            if sha256(fields_cache) != manifest["source_poison_fields_sha256"]:
                raise ValueError("cached production-path GONG poison fields changed after completion")
            (out_dir / "tdwsahuxt_gong_source_poison.json").write_bytes(poison_cache.read_bytes())
            (out_dir / "tdwsahuxt_source_poison_fields.npz").write_bytes(fields_cache.read_bytes())
            print("TD-WSAHUXT72 state cache hit " + json.dumps({"cache_key": cache_key, "origins": manifest["n_origins"]}), flush=True)
            return manifest

    nr, nlon = model.nr, model.nlon
    integrator = _step_function()
    reuse_state = False
    if state_path.is_file() and origins_path.is_file() and floor_path.is_file():
        try:
            prior_origins = np.load(origins_path, allow_pickle=False)
            prior_floor = np.load(floor_path, allow_pickle=False)
            prior_state = np.load(state_path, mmap_mode="r", allow_pickle=False)
            reuse_state = (np.array_equal(prior_origins, supported_origins)
                           and np.array_equal(prior_floor, state_floor_ns)
                           and prior_state.shape == (len(supported_origins), nr, nlon)
                           and prior_state.dtype == np.dtype(np.float16))
            if reuse_state:
                for start in range(0, len(supported_origins), 128):
                    block = np.asarray(prior_state[start:start + 128])
                    if not np.isfinite(block).all() or np.any(block <= 0):
                        reuse_state = False
                        break
            del prior_state
        except (OSError, ValueError):
            reuse_state = False

    if reuse_state:
        print("TD-WSAHUXT72 reused verified finite causal origin-state cache " + json.dumps({
            "origins": int(len(supported_origins)), "state_sha256": sha256(state_path),
            "state_floor_sha256": sha256(floor_path),
        }), flush=True)
    else:
        np.save(origins_path, supported_origins)
        np.save(floor_path, state_floor_ns)
        state_map = np.lib.format.open_memmap(state_path, mode="w+", dtype=np.float16,
                                              shape=(len(supported_origins), nr, nlon))
        output_rows = np.lib.format.open_memmap(cache_root / "states-in-progress.npy", mode="w+",
                                                 dtype=np.float32, shape=state_map.shape)
        params = model.model_params
        print("TD-WSAHUXT72 integrating causal boundary history " + json.dumps({
            "native_steps": int(len(model_time_s)), "requested_origins": int(len(supported_origins)),
            "native_dt_seconds": dt_s, "threads": 4,
        }), flush=True)
        saved = integrator(boundary, np.asarray(model.rrel.value, dtype=np.float64),
                           float(params[0]), float(params[1]), float(params[2]),
                           ceil_index.astype(np.int64), floor_index.astype(np.int64), fractions, output_rows)
        if saved != len(supported_origins):
            raise RuntimeError(f"HUXt history state integrator saved {saved}/{len(supported_origins)} origins")
        state_map[:] = output_rows[:]
        state_map.flush()
        del output_rows, state_map
        (cache_root / "states-in-progress.npy").unlink(missing_ok=True)
        print("TD-WSAHUXT72 saved causal origin states " + json.dumps({"origins": len(supported_origins)}), flush=True)

    # Validate the custom boundary mapper against the official HUXt static mapping.
    probe_profile = int(np.searchsorted(available_ns, int(pd.Timestamp("2026-07-01T00:00:00Z").value), side="right") - 1)
    _, static_model, _ = _new_model(int(pd.Timestamp("2026-07-01T00:00:00Z").value), speeds[probe_profile], FUTURE_HOURS + 1)
    static_model.ts_from_vlong()
    static_rel = np.asarray(static_model.model_time.to_value(u.s), dtype=np.float64)
    static_ids = np.full(len(static_rel), probe_profile, dtype=np.int32)
    static_rotated = _rotated_profiles(static_model, speeds[[probe_profile]])
    static_boundary = _boundary_matrix(static_model, static_rel, static_ids * 0, static_rotated,
                                       float(static_model.simtime.to_value(u.s)))
    static_delta = float(np.max(np.abs(static_boundary - np.asarray(static_model.input_v_ts, dtype=np.float32))))
    if static_delta > 1e-5:
        raise RuntimeError(f"custom time-dependent boundary mapper differs from HUXt static boundary by {static_delta}")

    official_pos = np.searchsorted(supported_origins, official_origins)
    official_ok = bool(np.all(official_pos < len(supported_origins))
                       and np.array_equal(supported_origins[official_pos], official_origins))
    if len(official_origins) != 2833 or not official_ok:
        raise RuntimeError("causal time-dependent HUXt history does not cover all 2,833 official origins")

    expected_shape = (len(supported_origins), FUTURE_HOURS, nr, nlon)
    if field_path.exists():
        with h5py.File(field_path, "r") as existing:
            reusable = ("fields" in existing and "done" in existing and "origins_ns" in existing
                        and "earth_lon_index" in existing
                        and tuple(existing["fields"].shape) == expected_shape
                        and np.array_equal(existing["origins_ns"][:], supported_origins))
        if not reusable:
            raise ValueError("incomplete HUXt future-field cache has a mismatched shape or origin index")
    else:
        with h5py.File(field_path, "w") as handle:
            handle.create_dataset("origins_ns", data=supported_origins, compression="gzip", compression_opts=1)
            handle.create_dataset(
                "fields", shape=expected_shape, dtype="<f2", chunks=(1, FUTURE_HOURS, nr, nlon),
                compression="gzip", compression_opts=1, shuffle=True, fillvalue=np.nan,
            )
            handle.create_dataset("earth_lon_index", shape=(len(supported_origins), FUTURE_HOURS),
                                  dtype="u1", fillvalue=255)
            handle.create_dataset("done", shape=(len(supported_origins),), dtype="u1", fillvalue=0)
    with h5py.File(field_path, "r+") as handle:
        fields, earth_indices, done = handle["fields"], handle["earth_lon_index"], handle["done"]
        todo = np.flatnonzero(np.asarray(done[:] == 0))
        initargs = (str(state_path), str(origins_path), str(floor_path), speeds, available_ns, updates, nr, nlon, cache_key)
        completed = int(len(supported_origins) - len(todo))
        with ProcessPoolExecutor(max_workers=4, initializer=_forecast_worker_init, initargs=initargs) as pool:
            todo_iter = iter(int(i) for i in todo)
            pending = {}
            for _ in range(min(16, len(todo))):
                index = next(todo_iter)
                pending[pool.submit(_forecast_one, index)] = index
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = pending.pop(future)
                    result_index, rows, earth_lon_index = future.result()
                    if result_index != index or rows.shape != (FUTURE_HOURS, nr, nlon) or not np.isfinite(rows).all():
                        raise RuntimeError(f"invalid HUXt future field returned for origin index {index}")
                    if earth_lon_index.shape != (FUTURE_HOURS,) or np.any(earth_lon_index >= nlon):
                        raise RuntimeError(f"invalid Earth longitude grid for HUXt origin index {index}")
                    fields[index] = rows
                    earth_indices[index] = earth_lon_index
                    done[index] = 1
                    completed += 1
                    try:
                        next_index = next(todo_iter)
                    except StopIteration:
                        continue
                    pending[pool.submit(_forecast_one, next_index)] = next_index
                if completed % 128 == 0 or completed == len(supported_origins):
                    handle.flush()
                    print(f"TD-WSAHUXT72 future field forecasts {completed}/{len(supported_origins)}", flush=True)
        handle.flush()
        if not bool(np.all(done[:])):
            raise RuntimeError("HUXt time-dependent future field cache is incomplete or nonfinite")

    official_source_ids = np.searchsorted(available_ns, official_origins, side="right") - 1
    if np.any(official_source_ids < 0):
        raise RuntimeError("causal SW-012 boundary history does not reach the official forecast origins")
    observation_ns_by_bin = np.asarray([row["observation_ns_by_bin"] for row in updates], dtype=np.int64)
    official_obs_ns = observation_ns_by_bin[official_source_ids]
    official_observation_age_h = (official_origins[:, None] - official_obs_ns) / HOUR_NS
    official_observation_map_count = sum(
        OFFICIAL_START <= pd.Timestamp(row["observation_utc"]) <= OFFICIAL_END
        for row in source["rows"]
    )
    all_official_observations = official_obs_ns.reshape(-1)
    coverage = {
        "source_id": "SW-012", "source_rule": source.get("availability_rule"),
        "source_maps": int(source.get("source_map_count", len(source["rows"]))),
        "finite_pfss_wsa_profiles": int(len(updates)),
        "source_map_range_utc": source.get("source_map_range_utc"),
        "official_window_available_map_count_manifest": int(source.get("official_window_map_count", 0)),
        "official_window_observation_map_count": int(official_observation_map_count),
        "official_latest_observation_utc": str(pd.Timestamp(int(all_official_observations.max()), tz="UTC")),
        "official_distinct_observations_used_across_longitude_bins": int(len(np.unique(all_official_observations))),
        "official_observation_age_hours_p50_p90_max": [
            float(value) for value in np.quantile(official_observation_age_h, [0.5, 0.9, 1.0])
        ],
        "supported_origins": int(len(supported_origins)),
        "requested_origins": int(len(requested_origins)),
        "official_origins": int(len(official_origins)), "official_coverage": 1.0,
        "official_coverage_semantics": "causal state fields exist for all origins; SW-012 has no observations inside the official window, so latest admissible per-longitude observations from May are held forward",
        "first_supported_origin_utc": str(pd.Timestamp(supported_origins[0], tz="UTC")),
        "last_supported_origin_utc": str(pd.Timestamp(supported_origins[-1], tz="UTC")),
        "buffer_rule": "only origins whose causal native state floor is at least 1.5 x the HUXt radial characteristic travel time from the first verified map are supported; travel time uses the minimum PFSS/WSA speed available by that floor",
        "r_min_rsun": R_MIN_RSUN, "r_max_rsun": R_MAX_RSUN, "radial_cells": int(nr),
        "synodic_longitude_cells": int(nlon), "native_dt_seconds": dt_s,
        "initial_map_available_utc": updates[0]["available_utc"],
        "history_boundary_selection": "per longitude, latest finite PFSS/WSA value whose observation time is newest among profiles with available_at <= boundary time; late publication of an older map cannot roll the state back",
        "origin_state_capture": "last HUXt native step at or before origin, followed by a causal fractional upwind step split at every SW-012 profile availability time through the origin",
        "future_boundary": "latest profile with available_at <= origin held at all post-origin boundary times",
    }
    manifest = {
        "passed": True, "cache_key": cache_key, "source_id": "SW-012",
        "source_manifest_sha256": source_sha, "profile_manifest_sha256": profile_sha,
        "origin_grid_sha256": origins_sha, "huxt_module_sha256": huxt_sha,
        "earth_ephemeris_path": str(ephemeris_path), "earth_ephemeris_sha256": ephemeris_sha,
        "huxt_version": importlib.metadata.version("huxt"),
        "state_code_sha256": sha256(Path(__file__)), "states_sha256": sha256(cache_root / "state_at_origin.npy"),
        "origins_sha256": sha256(origins_path), "state_floor_sha256": sha256(floor_path),
        "lead_fields_sha256": sha256(field_path),
        "n_origins": int(len(supported_origins)), "official_coverage": official_ok,
        "static_boundary_max_delta_kms": static_delta,
        "field_shape": [int(len(supported_origins)), FUTURE_HOURS, int(nr), int(nlon)],
        "field_dtype": "float16 HUXt speed in km/s", "field_cache": str(field_path),
        "earth_longitude_index": "lead-specific nearest HUXt synodic longitude grid point to HUXt.Observer(EARTH).lon, matching huxt_analysis.get_observer_timeseries",
        "state_cache": str(cache_root / "state_at_origin.npy"), "origin_index": str(origins_path),
        "state_floor_index": str(floor_path), "state_floor_sha256": sha256(floor_path),
        "coverage": coverage, "elapsed_seconds": time.monotonic() - started,
    }
    poison = _poison_probe(
        out_dir, cache_root, updates, available_ns, speeds, source_sha,
        model, boundary, model_time_s, selected_at_time, duration_s,
        supported_origins, floor_index, state_path, origins_path, floor_path,
        field_path, integrator,
    )
    poison_path = cache_root / "source_poison.json"
    manifest["source_poison_sha256"] = sha256(poison_path)
    manifest["source_poison_fields_path"] = str(cache_root / "source_poison_fields.npz")
    manifest["source_poison_fields_sha256"] = sha256(cache_root / "source_poison_fields.npz")
    manifest["source_poison_passed"] = bool(poison["passed"])
    _dump(manifest_path, manifest)
    _dump(out_dir / "tdwsahuxt_cache_manifest.json", manifest)
    _dump(out_dir / "tdwsahuxt_source_coverage.json", coverage)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tdwsahuxt_gong_source_poison.json").write_bytes(poison_path.read_bytes())
    print("TD-WSAHUXT72 causal HUXt fields complete " + json.dumps({
        "cache_key": cache_key, "origins": len(supported_origins), "official_coverage": official_ok,
        "field_shape": manifest["field_shape"], "source_poison": poison["passed"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }), flush=True)
    return manifest


def read_cache(manifest_path: Path) -> tuple[dict, h5py.File, np.ndarray, np.ndarray]:
    source_path = CACHE / "source_manifest.json"
    profile_path = CACHE / "profile_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("TD-WSAHUXT72 state cache is not built")
    manifest = json.loads(manifest_path.read_text())
    cache_root = Path(manifest["field_cache"]).parent
    if (not manifest.get("passed") or not manifest.get("source_poison_passed")
            or manifest.get("source_manifest_sha256") != sha256(source_path)
            or manifest.get("profile_manifest_sha256") != sha256(profile_path)):
        raise ValueError("TD-WSAHUXT72 state cache receipt is incomplete or mismatched")
    if sha256(cache_root / "state_at_origin.npy") != manifest["states_sha256"]:
        raise ValueError("TD-WSAHUXT72 origin-state cache SHA mismatch")
    if sha256(Path(manifest["state_floor_index"])) != manifest["state_floor_sha256"]:
        raise ValueError("TD-WSAHUXT72 origin-state floor index SHA mismatch")
    field_path = Path(manifest["field_cache"])
    if sha256(field_path) != manifest["lead_fields_sha256"]:
        raise ValueError("TD-WSAHUXT72 lead-field cache SHA mismatch")
    if sha256(Path(manifest["earth_ephemeris_path"])) != manifest["earth_ephemeris_sha256"]:
        raise ValueError("TD-WSAHUXT72 HUXt Earth ephemeris SHA mismatch")
    if sha256(Path(manifest["source_poison_fields_path"])) != manifest["source_poison_fields_sha256"]:
        raise ValueError("TD-WSAHUXT72 production poison field SHA mismatch")
    _, _, _, _, speeds = _load_profiles()
    origins = np.load(manifest["origin_index"], allow_pickle=False)
    fields = h5py.File(field_path, "r")
    if not np.array_equal(fields["origins_ns"][:], origins):
        fields.close()
        raise ValueError("TD-WSAHUXT72 field cache origin index differs from its receipt")
    return manifest, fields, speeds, origins


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--origins", type=Path)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    origin_grid = np.load(args.origins, allow_pickle=False) if args.origins else None
    build(args.out, origin_grid)
