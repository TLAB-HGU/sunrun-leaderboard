#!/usr/bin/env python3
"""Render frozen submitted forecasts and explicitly labeled reference comparators; never fit a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

EXPERIMENTS = {
    "E1466": {
        "model": "e1466_td_wsa_huxt72",
        "title": "E1466 TD-WSA HUXt72 (3-seed mean)",
        "expected_mse": 3246.254993999773,
        "seeds": [0, 1, 2],
        "aggregation": "equal prediction mean",
        "ace_latency_hours": 2,
    },
    "E1604": {
        "model": "e1604_hsspeak25",
        "title": "E1604 HSSPEAK25 (3-seed mean)",
        "model_config_sha256": "4a0e7bb648ab92c9bb75c76443d5576f5bd4576481edc8161c44f8c0321e7f5b",
        "expected_mse": 3222.882998440603,
        "seeds": [0, 1, 2],
        "aggregation": "equal prediction mean",
    },
    "E1593": {
        "model": "hsspeak72",
        "title": "E1593 HSSPEAK72 (3-seed mean)",
        "model_config_sha256": "eef5f301b253b41369b1b18821e685dc95b5cc9d80d766f5fa06d78d8f5cf965",
        "expected_mse": 3225.3843694024536,
        "seeds": [0, 1, 2],
        "aggregation": "equal prediction mean",
    },
    "E0471": {
        "model": "xgb_d76_lead_moe_enlil",
        "model_config_sha256": "6d169ff2b12085f3639b812884cef17324e1b096fb644410e18bd074103d3b54",
        "expected_mse": 3489.9620220435354,
    },
    "E0472": {
        "model": "xgb_d76_lead_moe_resid_dv_enlil",
        "model_config_sha256": "b8ae672fdb6ff9a6dc85bc5c3f2d8b2acbc9207c98126d39ad1913f24581962c",
        "expected_mse": 3248.6812422215053,
    },
    "E1149": {
        "model": "E1149_L72_COND",
        "title": "E1149 L72-COND (ACE+2h)",
        "recipe_code_sha256": "d8271c4c9bbb02734abaf1399393183143cfe318e7551e35f6119e0086105eb9",
        "expected_mse": 3383.173808352058,
        "ace_latency_hours": 2,
    },
    "E1149_ACE0H": {
        "model": "E1149_NO_DELAY",
        "title": "E1149 L72-COND (ACE+0h)",
        "parent_recipe_code_sha256": "d8271c4c9bbb02734abaf1399393183143cfe318e7551e35f6119e0086105eb9",
        "reference_only": True,
        "ace_latency_hours": 0,
    },
    "E2000FS_DIRMO24": {
        "model": "e2000fs_dirmo_24",
        "title": "E2000FS DirMO-24 (seed 42)",
        "model_config_sha256": "6fc76b1836f3a22669108823c6bf6833082b4d9a74bedf80edd8199f66183037",
        "expected_mse": 3480.2701508981454,  # overall F1 + F2
        "seeds": [42],
        "aggregation": "single model",
        "periods": ("F1", "F2"),
    },
}
ORIGIN = "origin_last_input_utc"
KEY = [ORIGIN, "horizon_hours"]
EXPECTED_MONTHS = {"2026-06": 720, "2026-07": 744, "2026-08": 744, "2026-09": 625}
F1_MONTHS = {"2026-03": 744, "2026-04": 720, "2026-05": 673}


def expected_months(name):
    """Official fold counts per UTC month (origin + 1 h); F2 only unless the experiment was scored on F1 + F2."""
    return {**F1_MONTHS, **EXPECTED_MONTHS} if "F1" in EXPERIMENTS[name].get("periods", ()) else EXPECTED_MONTHS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def history_hours(args):
    latency = getattr(args, "ace_latency_hours", 0)
    require(latency in (0, 2), "ACE latency must be 0 or 2 hours")
    expected = EXPERIMENTS[args.experiment].get("ace_latency_hours", 0)
    require(latency == expected, f"{args.experiment} requires --ace-latency-hours {expected}")
    return np.arange(-167 - latency, 1 - latency)


def load_data(args):
    require(not EXPERIMENTS[args.experiment].get("reference_only", False),
            "Unsubmitted comparator requires --reference-month")
    input_hours = history_hours(args)
    expected_mse = EXPERIMENTS[args.experiment]["expected_mse"]
    month_counts = expected_months(args.experiment)
    n_folds = sum(month_counts.values())
    folds = pd.read_parquet(args.folds)
    pred = pd.read_parquet(args.predictions)
    truth = pd.read_parquet(args.truth)
    for frame in (folds, pred, truth):
        frame[ORIGIN] = pd.to_datetime(frame[ORIGIN], utc=True)
    origins = pd.DatetimeIndex(folds[ORIGIN]).sort_values()
    require(len(origins) == n_folds and origins.is_unique, f"Expected {n_folds:,} unique folds")
    expected = pd.MultiIndex.from_product([origins, range(1, 73)], names=KEY)
    for name, frame in (("predictions", pred), ("truth", truth)):
        keys = pd.MultiIndex.from_frame(frame[KEY])
        require(keys.is_unique and len(keys) == len(expected)
                and expected.difference(keys).empty and keys.difference(expected).empty,
                f"{name}: expected exactly one row for every fold and horizon 1..72")
    paired = truth[KEY + ["target_kms", "was_missing"]].merge(
        pred[KEY + ["pred_kms"]], on=KEY, validate="one_to_one"
    ).sort_values(KEY)
    require(np.isfinite(paired[["target_kms", "pred_kms"]].to_numpy()).all(),
            "Non-finite forecast or truth")
    require(paired.was_missing.isin([0, 1]).all(), "Invalid truth missing flags")
    mse = float(np.mean((paired.target_kms - paired.pred_kms) ** 2))
    require(abs(mse - expected_mse) <= 1e-7, f"Frozen leaderboard MSE mismatch: {mse}")
    months = (origins + pd.Timedelta(hours=1)).strftime("%Y-%m")
    counts = pd.Series(months).value_counts().sort_index().to_dict()
    require(counts == month_counts, f"Unexpected month counts: {counts}")

    history = pd.read_parquet(args.history)
    history.index = pd.to_datetime(history.pop("timestamp_utc"), utc=True)
    require(history.index.is_unique, "Duplicate history timestamps")
    history = history.sort_index()
    # Use only existing hourly values: no interpolation or extra forward filling.
    slots = origins.tz_localize(None).to_numpy()[:, None] + pd.to_timedelta(input_hours, unit="h").to_numpy()
    selected = history.reindex(pd.DatetimeIndex(slots.ravel(), tz="UTC"))
    require(np.isfinite(selected.filled_speed_kms.to_numpy()).all(),
            "History is missing required input hours")
    require(selected.was_missing.isin([0, 1]).all(), "Invalid history missing flags")
    inputs = selected.filled_speed_kms.to_numpy().reshape(-1, 168)
    input_missing = selected.was_missing.to_numpy(dtype=bool).reshape(-1, 168)
    actual = paired.target_kms.to_numpy().reshape(-1, 72)
    predicted = paired.pred_kms.to_numpy().reshape(-1, 72)
    actual_missing = paired.was_missing.to_numpy(dtype=bool).reshape(-1, 72)
    values = np.concatenate([inputs.ravel(), actual.ravel(), predicted.ravel()])
    lower = float(np.floor((values.min() - 25) / 50) * 50)
    upper = float(np.ceil((values.max() + 25) / 50) * 50)
    return origins, months, inputs, input_missing, actual, predicted, actual_missing, mse, (lower, upper)


def load_reference_data(args):
    """Validate monthly replay without applying official score gates."""
    input_hours = history_hours(args)
    month = args.reference_month
    require(len(month) == 7 and pd.Period(month, freq="M").strftime("%Y-%m") == month,
            "reference-month must be YYYY-MM")
    pred = pd.read_parquet(args.predictions)
    pred[ORIGIN] = pd.to_datetime(pred[ORIGIN], utc=True)
    origins = pd.DatetimeIndex(pred[ORIGIN].unique()).sort_values()
    require(len(origins) > 0 and not origins.hasnans, "No valid reference origins")
    require((origins == origins.floor("h")).all(), "Reference origins must be hourly")
    months = (origins + pd.Timedelta(hours=1)).strftime("%Y-%m")
    require((months == month).all(), "Reference origins fall outside requested month")
    expected = pd.MultiIndex.from_product([origins, range(1, 73)], names=KEY)
    keys = pd.MultiIndex.from_frame(pred[KEY])
    require(keys.is_unique and len(keys) == len(expected)
            and expected.difference(keys).empty and keys.difference(expected).empty,
            "predictions: expected exactly one row for every origin and horizon 1..72")
    predicted = pred.sort_values(KEY).pred_kms.to_numpy().reshape(-1, 72)
    require(np.isfinite(predicted).all(), "Non-finite forecast")
    history = pd.read_parquet(args.history)
    history.index = pd.to_datetime(history.pop("timestamp_utc"), utc=True)
    require(history.index.is_unique, "Duplicate history timestamps")
    slots = (origins.tz_localize(None).to_numpy()[:, None]
             + pd.to_timedelta(np.concatenate([input_hours, np.arange(1, 73)]), unit="h").to_numpy())
    selected = history.reindex(pd.DatetimeIndex(slots.ravel(), tz="UTC"))
    require(np.isfinite(selected.filled_speed_kms.to_numpy()).all(),
            "History is missing required input or future actual hours")
    require(selected.was_missing.isin([0, 1]).all(), "Invalid history missing flags")
    values = selected.filled_speed_kms.to_numpy().reshape(-1, 240)
    missing = selected.was_missing.to_numpy(dtype=bool).reshape(-1, 240)
    inputs, actual = values[:, :168], values[:, 168:]
    all_values = np.concatenate([values.ravel(), predicted.ravel()])
    lower = 200. if all_values.min() >= 200 else float(np.floor((all_values.min() - 25) / 50) * 50)
    upper = 800. if all_values.max() <= 800 else float(np.ceil((all_values.max() + 25) / 50) * 50)
    mse = float(np.mean((actual - predicted) ** 2))
    return origins, months, inputs, missing[:, :168], actual, predicted, missing[:, 168:], mse, (lower, upper)


def fixed_palette():
    # Shared RGB cube plus greys keeps color identities consistent across frames.
    colors = [(r, g, b) for r in range(0, 256, 51)
              for g in range(0, 256, 51) for b in range(0, 256, 51)]
    colors += [(v, v, v) for v in np.linspace(0, 255, 40, dtype=int)]
    palette = Image.new("P", (1, 1))
    palette.putpalette([int(v) for color in colors for v in color])
    return palette


def render(args, data):
    experiment = EXPERIMENTS[args.experiment]
    history_x, future_x = history_hours(args), np.arange(1, 73)
    latency = int(-history_x[-1])
    title = experiment.get("title", experiment["model"])
    origins, months, inputs, input_missing, actual, predicted, actual_missing, mse, ylim = data
    reference_month = getattr(args, "reference_month", None)
    render_months = [reference_month] if reference_month else list(expected_months(args.experiment))
    manifest_name = f"manifest_{reference_month}.json" if reference_month else "manifest.json"
    provenance = None
    if reference_month and getattr(args, "provenance", None):
        require(getattr(args, "model_file", None) is not None, "--provenance requires --model-file")
        provenance = json.loads(args.provenance.read_text())
        require(provenance.get("model_sha256") == sha256(args.model_file), "Provenance model hash mismatch")
        require(provenance.get("predictions_sha256") == sha256(args.predictions), "Provenance predictions hash mismatch")
    output = args.output_dir
    if reference_month:
        for name in [manifest_name, f"validation_{reference_month}.gif"]:
            require(not (output / name).exists(), f"Refusing to overwrite existing reference output: {name}")
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(args.width / 100, args.height / 100), dpi=100)
    fig.subplots_adjust(left=.085, right=.98, bottom=.17, top=.80)
    ax.set(xlim=(int(history_x[0]), 72), ylim=ylim, xlabel="Hours relative to forecast origin (UTC)",
           ylabel="Solar wind speed (km/s)")
    ax.set_xticks([-144, -120, -96, -72, -48, -24, 0, 24, 48, 72])
    ax.axvspan(history_x[0], history_x[-1], color="#eaf2fa", zorder=0)
    ax.axvspan(0, 72, color="#fff3df", zorder=0)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax.grid(alpha=.20)
    input_line, = ax.plot(history_x, inputs[0], color="#003366", linewidth=1.4, label="Model available input (168 h)" if latency else "Input observation (168 h)")
    truth_line, = ax.plot(future_x, actual[0], color="black", linewidth=1.4, label="Future actual (72 h)")
    pred_line, = ax.plot(future_x, predicted[0], color="#ff9900", linewidth=1.7, label="Model prediction")
    missing_line, = ax.plot([], [], linestyle="none", marker="o", markersize=3.8,
                           markerfacecolor="white", markeredgecolor="#666666", markeredgewidth=.8,
                           label="Filled / missing observation")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.16), ncol=2, fontsize=8, frameon=False)
    heading = fig.suptitle("", fontsize=12, y=.97)
    scale_label = "reference frames" if reference_month else "folds"
    fig.text(.085, .035, f"ACE observation time: {history_x[0]} to {history_x[-1]} h  |  Forecast: +1 to +72 h  |  Fixed scale across {scale_label}", fontsize=8)
    palette = fixed_palette()
    results = []
    try:
        for month in render_months:
            indices = np.flatnonzero(months == month)
            if args.limit is not None:
                indices = indices[:args.limit]
            frames = []
            try:
                for index in indices:
                    input_line.set_ydata(inputs[index])
                    truth_line.set_ydata(actual[index])
                    pred_line.set_ydata(predicted[index])
                    missing_line.set_data(
                        np.concatenate([history_x[input_missing[index]], future_x[actual_missing[index]]]),
                        np.concatenate([inputs[index][input_missing[index]], actual[index][actual_missing[index]]]))
                    rmse = float(np.sqrt(np.mean((actual[index] - predicted[index]) ** 2)))
                    frame_label = "Reference (non-official)" if reference_month else "Fold"
                    if experiment.get("reference_only"):
                        frame_label = "Unsubmitted comparator"
                    heading.set_text(f"{title}  |  {frame_label} {index + 1:,} / {len(origins):,}\n"
                                     f"Origin {origins[index]:%Y-%m-%d %H:%M UTC}  |  72 h RMSE {rmse:.2f} km/s")
                    fig.canvas.draw()
                    rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
                    frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
                    rgb.close()
                    if len(frames) % 100 == 0:
                        print(f"{month}: rendered {len(frames)}/{len(indices)}", flush=True)
                path = output / f"validation_{month}.gif"
                frames[0].save(path, save_all=True, append_images=frames[1:], duration=args.frame_ms,
                               loop=0, optimize=True, disposal=2)
            finally:
                for frame in frames:
                    frame.close()
            with Image.open(path) as gif:
                require(gif.n_frames == len(indices), f"GIF frame count mismatch: {path}")
                require(gif.size == (args.width, args.height), "GIF dimensions mismatch")
                require(gif.info.get("loop") == 0, "GIF must loop indefinitely")
                duration = 0
                for i in range(gif.n_frames):
                    gif.seek(i)
                    gif.load()
                    require(gif.info.get("duration") == args.frame_ms, "GIF frame duration mismatch")
                    duration += gif.info["duration"]
            results.append({"month": month, "file": path.name, "frames": len(indices),
                            "first_fold": int(indices[0] + 1), "last_fold": int(indices[-1] + 1),
                            "first_origin_utc": origins[indices[0]].isoformat(),
                            "last_origin_utc": origins[indices[-1]].isoformat(),
                            "duration_ms": duration, "bytes": path.stat().st_size, "sha256": sha256(path)})
            print(f"{path.name}: verified {len(indices)} frames, {path.stat().st_size:,} bytes", flush=True)
    finally:
        plt.close(fig)
    config = {"width": args.width, "height": args.height, "frame_ms": args.frame_ms,
              "input_hours": 168, "forecast_hours": 72, "x_hours": [int(history_x[0]), 72],
              "ace_latency_hours": latency, "input_x_hours": [int(history_x[0]), int(history_x[-1])],
              "y_limits_kms": list(ylim), "limit_per_month": args.limit,
              "month_grouping": "UTC month of origin + 1 hour", "palette": "RGB cube + greys; no dithering"}
    source_names = ["predictions", "history"] if reference_month else ["predictions", "folds", "truth", "history"]
    if reference_month and getattr(args, "model_file", None):
        source_names.append("model_file")
    if provenance is not None:
        source_names.append("provenance")
    manifest = {
        "model": experiment["model"],
        **{key: experiment[key] for key in ("model_config_sha256", "recipe_code_sha256", "parent_recipe_code_sha256") if key in experiment},
        "experiment": args.experiment,
        **({"seeds": experiment["seeds"], "aggregation": experiment["aggregation"]}
           if "seeds" in experiment else {"seed": 0}),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(__file__), "config": config,
        "rendering_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "sources": {name: {"file": str(getattr(args, name).resolve()), "sha256": sha256(getattr(args, name))}
                    for name in source_names},
        "verification": {"folds": len(origins), "pairs": len(origins) * 72,
                         "mse": mse, "expected_mse": experiment.get("expected_mse"), "mse_tolerance": 1e-7,
                         "expected_month_counts": expected_months(args.experiment),
                         "rendered_frames": sum(item["frames"] for item in results),
                         "complete": args.limit is None, "all_gifs_decoded": True,
                         "prediction_truth_join": "one-to-one; complete origin x horizons 1..72",
                         "input_window": f"168 existing hourly samples ending at origin minus {latency} hours; no added filling"},
        "notes": ["Display shows seven days, not the model's complete 26-28 day feature history.",
                  "Hollow markers identify filled observations according to source was_missing flags.",
                  "All validation truth, including filled observations, contributes to displayed RMSE."],
        "outputs": results,
    }
    if args.experiment.startswith("E1149"):
        manifest["notes"][0] = "Display shows 168 hours of model-available ACE observations, not the complete feature and analog history."
    if experiment.get("reference_only"):
        manifest["comparison_status"] = "Unsubmitted no-delay comparator; parent recipe identity is not a training-config hash"
    if reference_month:
        manifest["evaluation_scope"] = ("non-official reference replay" if args.experiment.startswith("E1149")
                                        else "non-official reference replay; v2 transition validation pending")
        manifest["reference_month"] = reference_month
        if provenance is not None:
            manifest["provenance"] = provenance
        for key in ("expected_mse", "mse_tolerance", "expected_month_counts"):
            del manifest["verification"][key]
        manifest["verification"]["origins"] = manifest["verification"].pop("folds")
        manifest["verification"]["actual_window"] = "72 existing hourly samples after origin; no added filling"
        manifest["notes"][-1] = "All reference actuals, including filled observations, contribute to displayed RMSE; excluded from official leaderboard score."
        for result in results:
            result["first_frame"] = result.pop("first_fold")
            result["last_frame"] = result.pop("last_fold")
    (output / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="E0471")
    for name in ("predictions", "history", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ace-latency-hours", type=int, choices=(0, 2), default=0)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--reference-month", help="Non-official replay month YYYY-MM (origin + 1 h)")
    parser.add_argument("--provenance", type=Path, help="Reference inference receipt with model and prediction hashes")
    parser.add_argument("--model-file", type=Path, help="Frozen model file to hash for reference provenance")
    parser.add_argument("--frame-ms", type=int, default=100)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--limit", type=int, help="Smoke test: render only this many folds per month")
    args = parser.parse_args()
    require(args.frame_ms > 0 and args.frame_ms % 10 == 0, "GIF frame-ms must be a positive multiple of 10")
    require(args.width >= 600 and args.height >= 350, "Use width >=600 and height >=350 for readable plots")
    require(args.limit is None or args.limit > 0, "limit must be positive")
    if args.reference_month:
        data = load_reference_data(args)
    else:
        require(args.provenance is None, "--provenance is only available with --reference-month")
        require(args.folds is not None and args.truth is not None, "Official mode requires --folds and --truth")
        data = load_data(args)
    render(args, data)


if __name__ == "__main__":
    main()
