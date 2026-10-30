#!/usr/bin/env python3
"""Render frozen E0471 or E0472 seed-0 leaderboard forecasts; never fit a model."""
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
}
ORIGIN = "origin_last_input_utc"
KEY = [ORIGIN, "horizon_hours"]
EXPECTED_MONTHS = {"2026-06": 720, "2026-07": 744, "2026-08": 744, "2026-09": 625}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(args):
    expected_mse = EXPERIMENTS[args.experiment]["expected_mse"]
    folds = pd.read_parquet(args.folds)
    pred = pd.read_parquet(args.predictions)
    truth = pd.read_parquet(args.truth)
    for frame in (folds, pred, truth):
        frame[ORIGIN] = pd.to_datetime(frame[ORIGIN], utc=True)
    origins = pd.DatetimeIndex(folds[ORIGIN]).sort_values()
    require(len(origins) == 2833 and origins.is_unique, "Expected 2,833 unique folds")
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
    require(counts == EXPECTED_MONTHS, f"Unexpected month counts: {counts}")

    history = pd.read_parquet(args.history)
    history.index = pd.to_datetime(history.pop("timestamp_utc"), utc=True)
    require(history.index.is_unique, "Duplicate history timestamps")
    history = history.sort_index()
    # Use only existing hourly values: no interpolation or extra forward filling.
    slots = origins.tz_localize(None).to_numpy()[:, None] + pd.to_timedelta(np.arange(-167, 1), unit="h").to_numpy()
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
    origins, months, inputs, input_missing, actual, predicted, actual_missing, mse, ylim = data
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(args.width / 100, args.height / 100), dpi=100)
    fig.subplots_adjust(left=.085, right=.98, bottom=.17, top=.80)
    ax.set(xlim=(-167, 72), ylim=ylim, xlabel="Hours relative to forecast origin (UTC)",
           ylabel="Solar wind speed (km/s)")
    ax.set_xticks([-144, -120, -96, -72, -48, -24, 0, 24, 48, 72])
    ax.axvspan(-167, 0, color="#eaf2fa", zorder=0)
    ax.axvspan(0, 72, color="#fff3df", zorder=0)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax.grid(alpha=.20)
    history_x, future_x = np.arange(-167, 1), np.arange(1, 73)
    input_line, = ax.plot(history_x, inputs[0], color="#003366", linewidth=1.4, label="Input observation (168 h)")
    truth_line, = ax.plot(future_x, actual[0], color="black", linewidth=1.4, label="Future actual (72 h)")
    pred_line, = ax.plot(future_x, predicted[0], color="#ff9900", linewidth=1.7, label="Model prediction")
    missing_line, = ax.plot([], [], linestyle="none", marker="o", markersize=3.8,
                           markerfacecolor="white", markeredgecolor="#666666", markeredgewidth=.8,
                           label="Filled / missing observation")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.16), ncol=2, fontsize=8, frameon=False)
    heading = fig.suptitle("", fontsize=12, y=.97)
    fig.text(.085, .035, "Input: -167 to 0 h  |  Forecast: +1 to +72 h  |  Fixed scale across all folds", fontsize=8)
    palette = fixed_palette()
    results = []
    try:
        for month in EXPECTED_MONTHS:
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
                    heading.set_text(f"{experiment['model']}  |  Fold {index + 1:,} / {len(origins):,}\n"
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
              "input_hours": 168, "forecast_hours": 72, "x_hours": [-167, 72],
              "y_limits_kms": list(ylim), "limit_per_month": args.limit,
              "month_grouping": "UTC month of origin + 1 hour", "palette": "RGB cube + greys; no dithering"}
    manifest = {
        "model": experiment["model"], "model_config_sha256": experiment["model_config_sha256"],
        "experiment": args.experiment, "seed": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(__file__), "config": config,
        "rendering_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "sources": {name: {"file": str(getattr(args, name).resolve()), "sha256": sha256(getattr(args, name))}
                    for name in ("predictions", "folds", "truth", "history")},
        "verification": {"folds": len(origins), "pairs": len(origins) * 72,
                         "mse": mse, "expected_mse": experiment["expected_mse"], "mse_tolerance": 1e-7,
                         "expected_month_counts": EXPECTED_MONTHS,
                         "rendered_frames": sum(item["frames"] for item in results),
                         "complete": args.limit is None, "all_gifs_decoded": True,
                         "prediction_truth_join": "one-to-one; complete origin x horizons 1..72",
                         "input_window": "168 existing hourly samples ending at origin; no added filling"},
        "notes": ["Display shows seven days, not the model's complete 26-28 day feature history.",
                  "Hollow markers identify filled observations according to source was_missing flags.",
                  "All validation truth, including filled observations, contributes to displayed RMSE."],
        "outputs": results,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="E0471")
    for name in ("predictions", "folds", "truth", "history", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frame-ms", type=int, default=100)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--limit", type=int, help="Smoke test: render only this many folds per month")
    args = parser.parse_args()
    require(args.frame_ms > 0 and args.frame_ms % 10 == 0, "GIF frame-ms must be a positive multiple of 10")
    require(args.width >= 600 and args.height >= 350, "Use width >=600 and height >=350 for readable plots")
    require(args.limit is None or args.limit > 0, "limit must be positive")
    render(args, load_data(args))


if __name__ == "__main__":
    main()
