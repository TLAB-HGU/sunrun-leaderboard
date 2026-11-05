"""Render the seven monthly forecast GIFs of the submitted budget-spectral forecast.

The frames show the forecast that was actually submitted to the leaderboard: `results/forecast.npz` of
`research/e2000fs_budget_spectral` (the spectral arm), joined one-to-one with
the official truth. Each frame is one of the 4,970 official origins: 168 h of observed ACE speed ending at the
origin, then the 72 h truth and the 72 h forecast. Hollow markers are values that the input store had to fill.

The script refuses to write a GIF when the prediction/truth join, the recomputed MSE or the monthly frame counts
disagree with the leaderboard receipt.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_budget_spectral/code/render_gifs.py

Use --months 2026-03 --limit 24 for a quick check; all seven months take a few minutes.
"""
import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path("/home/t-lab01/sunrun")
STUDY = ROOT / "research/e2000fs_budget_spectral"
RESULTS = STUDY / "results"
OUT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data-v2/processed/ACE/ace.parquet"
SPEED = ROOT / "data-v2/bundles/frozen/speed.parquet"

INPUT_HOURS, FORECAST_HOURS = 168, 72
WIDTH, HEIGHT, FRAME_MS = 1000, 500, 100
Y_LIMITS = (200.0, 850.0)
COLOR = {"history": "#1a6fb5", "truth": "#111111", "forecast": "#e8620c", "origin": "#8a8a8a"}
EXPECTED_MONTHS = {"2026-03": 744, "2026-04": 720, "2026-05": 673,
                   "2026-06": 720, "2026-07": 744, "2026-08": 744, "2026-09": 625}


def pinned(config, suffix):
    identity = config["data"]["input_identity"]
    return Path(next(p for p in identity if p.endswith(suffix)))


def load(config):
    receipt = json.loads((RESULTS / "leaderboard-result.json").read_text())
    forecast_path = RESULTS / "forecast.npz"
    bundle = np.load(forecast_path)
    origins = pd.DatetimeIndex(bundle["origins_ns"], tz="UTC")
    predictions = bundle["predictions"]

    folds = pd.read_parquet(pinned(config, "/folds.parquet"))
    truth = pd.read_parquet(pinned(config, "/truth.parquet"))
    fold_origins = pd.DatetimeIndex(folds["origin_last_input_utc"])

    assert len(origins) == len(fold_origins) == receipt["n_folds"], "origin count differs from the receipt"
    assert origins.equals(fold_origins), "origins are not the leaderboard folds in order"
    assert predictions.shape == (len(origins), FORECAST_HOURS)
    assert np.isfinite(predictions).all()

    long = pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), FORECAST_HOURS),
        "horizon_hours": np.tile(np.arange(1, FORECAST_HOURS + 1), len(origins)),
        "pred_kms": predictions.ravel()})
    joined = long.merge(truth, on=["origin_last_input_utc", "horizon_hours"], how="inner", validate="one_to_one")
    assert len(joined) == receipt["n_pairs"] == len(origins) * FORECAST_HOURS, "prediction/truth join is incomplete"

    # The inner join keeps the left (fold) order; check it before reshaping, since rendering relies on it.
    expected_order = np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), FORECAST_HOURS)
    assert (joined["origin_last_input_utc"].to_numpy() == expected_order).all(), "join reordered the folds"
    target = joined["target_kms"].to_numpy().reshape(len(origins), FORECAST_HOURS)
    filled = joined["was_missing"].to_numpy().reshape(len(origins), FORECAST_HOURS).astype(bool)

    error = predictions - target
    scores = dict(mse=float(np.mean(error ** 2)),
                  mse_f1=float(np.mean(error[folds["period"].to_numpy() == "F1"] ** 2)),
                  mse_f2=float(np.mean(error[folds["period"].to_numpy() == "F2"] ** 2)))
    for key, value in scores.items():
        assert abs(value - receipt[key]) < 1e-9, f"{key} {value!r} != leaderboard {receipt[key]!r}"
    assert np.abs(error).max() < 1e4 and np.isfinite(target).all()

    series = pd.read_parquet(HISTORY)
    series = series.set_index("timestamp_utc")
    speed = series["filled_speed_kms"]
    speed_gaps = series["was_missing"].to_numpy().astype(bool)
    reference = pd.read_parquet(SPEED)["speed"]
    assert speed.index.equals(reference.index) and np.array_equal(speed.to_numpy(), reference.to_numpy()), \
        "history series differs from the study's model input"

    position = speed.index.get_indexer(origins)
    assert (position >= INPUT_HOURS - 1).all() and (position < len(speed)).all(), "origin outside the history series"
    window = position[:, None] + np.arange(-INPUT_HOURS + 1, 1)[None, :]
    history = speed.to_numpy()[window]
    history_filled = speed_gaps[window]
    assert np.isfinite(history).all(), "history window contains a gap"

    months = (origins + pd.Timedelta(hours=1)).strftime("%Y-%m")
    return dict(origins=origins, predictions=predictions, target=target, filled=filled, history=history,
                history_filled=history_filled, months=months, scores=scores, receipt=receipt, folds=folds)


def figure():
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = fig.add_axes([0.058, 0.115, 0.925, 0.735])
    ax.set_xlim(-INPUT_HOURS + 1, FORECAST_HOURS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xlabel("hours relative to origin (t0, last observed hour)", fontsize=9)
    ax.set_ylabel("solar wind speed (km/s)", fontsize=9)
    ax.set_xticks(np.arange(-168, 73, 24))
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=8.5)
    ax.axvline(0, color=COLOR["origin"], lw=.9, ls=":")
    hist_x = np.arange(-INPUT_HOURS + 1, 1)
    future_x = np.arange(1, FORECAST_HOURS + 1)
    lines = {
        "history": ax.plot(hist_x, np.full(len(hist_x), np.nan), color=COLOR["history"], lw=1.3,
                           label=f"observed, h-{INPUT_HOURS - 1} … h0")[0],
        "history_filled": ax.plot([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=COLOR["history"],
                                  mew=.8, label="observed but filled (missing in the source)")[0],
        "truth": ax.plot(future_x, np.full(len(future_x), np.nan), color=COLOR["truth"], lw=1.3,
                         label=f"truth, h+1 … h+{FORECAST_HOURS}")[0],
        "truth_filled": ax.plot([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=COLOR["truth"], mew=.8,
                                label="truth but filled")[0],
        "forecast": ax.plot(future_x, np.full(len(future_x), np.nan), color=COLOR["forecast"], lw=1.6,
                            label="submitted forecast, h+1 … h+72")[0],
    }
    ax.legend(fontsize=7.6, loc="upper left", ncol=3, framealpha=.86, handlelength=1.7, columnspacing=1.1)
    title = fig.text(0.058, 0.965, "", fontsize=10, va="top")
    subtitle = fig.text(0.058, 0.915, "", fontsize=9, va="top", color="#444444")
    note = fig.text(0.058, 0.022, "e2000fs_budget_spectral · spectral arm · direct 24 h blocks, Top30 features, "
                    "ACE lag 0 · Y axis fixed at 200–850 km/s", fontsize=7.4, color="#666666")
    return fig, lines, (title, subtitle, note)


def frames(data, indices, dpi_scale=1.0):
    """Yield one quantized frame per fold index."""
    fig, lines, texts = figure()
    hist_x = np.arange(-INPUT_HOURS + 1, 1)
    future_x = np.arange(1, FORECAST_HOURS + 1)
    periods = data["folds"]["period"].to_numpy()
    palette = None
    for index in indices:
        history, forecast, truth = data["history"][index], data["predictions"][index], data["target"][index]
        missing_history, missing_truth = data["history_filled"][index], data["filled"][index]
        lines["history"].set_data(hist_x, history)
        lines["history_filled"].set_data(hist_x[missing_history], history[missing_history])
        lines["truth"].set_data(future_x, truth)
        lines["truth_filled"].set_data(future_x[missing_truth], truth[missing_truth])
        lines["forecast"].set_data(future_x, forecast)
        rmse = float(np.sqrt(np.mean((forecast - truth) ** 2)))
        origin = data["origins"][index]
        texts[0].set_text(f"e2000fs_budget_spectral — fold {index + 1:,} / {len(data['origins']):,} · "
                          f"{periods[index]}")
        texts[1].set_text(f"origin (t0) {origin.strftime('%Y-%m-%d %H:%M')} UTC · 72 h RMSE {rmse:.1f} km/s")
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        image = Image.fromarray(buffer)
        if palette is None:
            palette = image.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        yield image.quantize(palette=palette, dither=Image.Dither.NONE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", nargs="*", help="subset of YYYY-MM (default: all seven)")
    parser.add_argument("--limit", type=int, help="render at most this many frames per month (smoke check)")
    parser.add_argument("--suffix", default="", help="suffix for output names (smoke check)")
    args = parser.parse_args()

    config = json.loads((RESULTS / "config.json").read_text())
    data = load(config)
    months = args.months or list(EXPECTED_MONTHS)
    present = pd.Series(1, index=data["months"]).groupby(level=0).size().to_dict()
    assert present == EXPECTED_MONTHS, f"monthly fold counts {present} != {EXPECTED_MONTHS}"

    outputs = []
    for month in months:
        indices = np.flatnonzero(data["months"] == month)
        expected = len(indices)
        if args.limit:
            indices = indices[:args.limit]
        frames_out = list(frames(data, indices))
        name = f"validation_{month}{args.suffix}.gif"
        buffer = io.BytesIO()
        frames_out[0].save(buffer, format="GIF", save_all=True, append_images=frames_out[1:],
                           duration=FRAME_MS, loop=0, optimize=False, disposal=1)
        payload = buffer.getvalue()
        (OUT / name).write_bytes(payload)

        with Image.open(OUT / name) as check:
            assert check.size == (WIDTH, HEIGHT), f"{name} is {check.size}"
            assert check.n_frames == len(frames_out), f"{name} has {check.n_frames} frames"
            assert check.info.get("duration") == FRAME_MS
            for position, frame in enumerate(frames_out):
                check.seek(position)
                assert np.array_equal(np.asarray(check.convert("RGB")), np.asarray(frame.convert("RGB"))), \
                    f"{name} frame {position} does not round-trip"
            assert check.info.get("loop") == 0
        first, last = int(indices[0]) + 1, int(indices[-1]) + 1
        outputs.append(dict(month=month, file=name, frames=len(indices), first_fold=first, last_fold=last,
                            first_origin_utc=data["origins"][indices[0]].isoformat(),
                            last_origin_utc=data["origins"][indices[-1]].isoformat(),
                            duration_ms=FRAME_MS * len(indices), bytes=len(payload)))
        print(json.dumps(outputs[-1]), flush=True)

    if args.suffix:
        print("smoke output written; manifest untouched")
        return

    receipt = data["receipt"]
    manifest = {
        "model": "e2000fs_budget_spectral",
        "experiment": receipt["experiment"],
        "member": receipt["team_member"],
        "submitted_utc": receipt["submitted_utc"],
        "arm": "spectral (FreDF-inspired radial pseudo-Huber on the unitary FFT of the 24 h error)",
        "aggregation": "single model, single seed 42, three 24-output vector boosters per stage",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": dict(width=WIDTH, height=HEIGHT, frame_ms=FRAME_MS, input_hours=INPUT_HOURS,
                       forecast_hours=FORECAST_HOURS, x_hours=[-INPUT_HOURS + 1, FORECAST_HOURS],
                       input_x_hours=[-INPUT_HOURS + 1, 0], ace_latency_hours=0,
                       y_limits_kms=list(Y_LIMITS), month_grouping="UTC month of origin + 1 hour",
                       palette="256-colour palette fixed from the month's first frame; no dithering"),
        "sources": {
            "predictions": {"file": str(RESULTS / "forecast.npz")},
            "truth": {"file": str(pinned(config, "/truth.parquet"))},
            "folds": {"file": str(pinned(config, "/folds.parquet"))},
            "history": {"file": str(HISTORY)},
        },
        "verification": {
            "folds": int(len(data["origins"])), "pairs": int(data["target"].size),
            "mse": data["scores"]["mse"], "expected_mse": receipt["mse"],
            "mse_f1": data["scores"]["mse_f1"], "expected_mse_f1": receipt["mse_f1"],
            "mse_f2": data["scores"]["mse_f2"], "expected_mse_f2": receipt["mse_f2"],
            "mse_tolerance": 1e-9,
            "expected_month_counts": EXPECTED_MONTHS,
            "rendered_frames": int(sum(o["frames"] for o in outputs)),
            "complete": bool(sum(o["frames"] for o in outputs) == len(data["origins"])),
            "all_gifs_decoded": True,
            "prediction_truth_join": "one-to-one; complete origin x horizons 1..72",
            "input_window": f"{INPUT_HOURS} hourly samples ending at the origin; filled values kept as stored",
        },
        "notes": [
            "The spectral arm is the submitted forecast; the paired control arm (point4 L2) scored 3494.210802 and "
            "is shown only in the loss figures.",
            "Hollow markers identify observations filled by the input store (ACE gaps), not model output.",
            "All truth, including filled observations, contributes to the displayed RMSE.",
        ],
        "outputs": outputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(outputs)} GIFs, {manifest['verification']['rendered_frames']:,} frames")


if __name__ == "__main__":
    main()
