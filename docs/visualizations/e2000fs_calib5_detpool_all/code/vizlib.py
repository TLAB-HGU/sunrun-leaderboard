"""Shared loader, verifier and monthly-GIF renderer for the F1/F2/F3 leaderboard visualisations.

Both folders of the 2026-11-01 publication (`e2000fs_calib5_detpool_all`, `e2000fs_calib_point4_144h`) use this module.
It reads the archived forecast that was submitted for the 2026-10-31 F1/F2/F3 refit, rebuilds the (origin, horizon) rows,
joins them to the leaderboard dataset's `truth.parquet` under the pinned `folds.parquet`, and refuses to draw anything
unless every recomputed score equals the leaderboard record to 1e-9. Nothing is retrained here.
"""
from __future__ import annotations

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
STORE = Path("/home/t-lab01/.cache/huggingface/hub/datasets--tlabtlab--sunrun-lb-store")
# Pinned leaderboard-dataset snapshot: folds/truth/naive of the F1/F2/F3 split (2026-10-31).
SNAPSHOT = STORE / "snapshots/dc409fc89c49864bff2760240e0a521afcf5ed87"
PINNED = {
    "folds": SNAPSHOT / "folds.parquet",
    "truth": SNAPSHOT / "truth.parquet",
    "naive": SNAPSHOT / "naive.parquet",
    "history": ROOT / "data-v2/processed/ACE/ace.parquet",
}
KEY = ["origin_last_input_utc", "horizon_hours"]
H = 72
INPUT_HOURS = 168
WIDTH, HEIGHT, FRAME_MS = 1000, 500, 100
Y_LIMITS = (200.0, 850.0)
COLOR = {"history": "#1a6fb5", "truth": "#111111", "forecast": "#e8620c", "origin": "#8a8a8a"}
MONTHS = {"2026-01": 744, "2026-02": 672, "2026-03": 673, "2026-04": 720, "2026-05": 744,
          "2026-06": 649, "2026-07": 744, "2026-08": 744, "2026-09": 625}
PERIOD_OF_MONTH = {"2026-01": "F1", "2026-02": "F1", "2026-03": "F1", "2026-04": "F2", "2026-05": "F2",
                   "2026-06": "F2", "2026-07": "F3", "2026-08": "F3", "2026-09": "F3"}


def pinned(name: str) -> Path:
    path = PINNED[name]
    assert path.exists(), f"missing pinned input {path}"
    return path


def load_predictions(files, keys) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Stack one or three archived numpy archives into (origins, predictions(N, 72))."""
    origins, blocks = [], []
    for path, (origin_key, pred_key) in zip(files, keys):
        bundle = np.load(path)
        origin = pd.DatetimeIndex(bundle[origin_key], tz="UTC")
        pred = bundle[pred_key].astype(np.float64)
        assert pred.shape == (len(origin), H), (path, pred.shape)
        assert np.isfinite(pred).all(), path
        origins.append(origin)
        blocks.append(pred)
    joined = origins[0].append(origins[1:]) if len(origins) > 1 else origins[0]
    pred = np.vstack(blocks)
    assert joined.is_monotonic_increasing and joined.is_unique, "origins are not a sorted unique index"
    assert pred.shape == (len(joined), H)
    return joined, pred


def load_evaluation(origins: pd.DatetimeIndex, receipt: dict) -> dict:
    """Join predictions to the leaderboard truth and verify every score of the leaderboard record."""
    folds = pd.read_parquet(pinned("folds"))
    truth = pd.read_parquet(pinned("truth"))
    fold_origins = pd.DatetimeIndex(folds["origin_last_input_utc"])
    assert len(origins) == len(fold_origins) == receipt["n_folds"], (len(origins), len(fold_origins))
    assert origins.equals(fold_origins), "origins are not the leaderboard folds in order"
    return {"folds": folds, "truth": truth}


def scores(pred: np.ndarray, target: np.ndarray, periods: np.ndarray) -> dict:
    err = (pred - target) ** 2
    high = target >= 550.0

    def mean_over(mask: np.ndarray) -> float:
        return float(err[mask].mean())

    out = {"mse": mean_over(np.ones_like(err, dtype=bool))}
    for period in ("F1", "F2", "F3"):
        where = np.broadcast_to((periods == period)[:, None], err.shape)
        out[f"mse_{period.lower()}"] = mean_over(where)
        out[f"mse_{period.lower()}_ge550"] = mean_over(where & high)
    out["mse_ge550"] = mean_over(high)
    out["n_pairs"] = int(err.size)
    out["n_pairs_ge550"] = int(high.sum())
    return out


def check_scores(recomputed: dict, receipt: dict) -> None:
    for key, value in recomputed.items():
        if key.startswith("n_"):
            assert value == receipt[key], f"{key} {value} != receipt {receipt[key]}"
        else:
            assert abs(value - receipt[key]) < 1e-9, f"{key} {value!r} != receipt {receipt[key]!r}"


def load(out_dir: Path, files, keys) -> dict:
    """Load the submitted forecast, verify it against the receipt in `out_dir`, and collect the drawing inputs."""
    receipt = json.loads((out_dir / "leaderboard-result.json").read_text())
    origins, pred = load_predictions(files, keys)
    ev = load_evaluation(origins, receipt)
    folds, truth = ev["folds"], ev["truth"]

    long = pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                         "horizon_hours": np.tile(np.arange(1, H + 1), len(origins)),
                         "pred_kms": pred.ravel()})
    joined = long.merge(truth, on=KEY, how="inner", validate="one_to_one")
    assert len(joined) == len(origins) * H, "prediction/truth join is incomplete"
    assert (joined["origin_last_input_utc"].to_numpy()
            == np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H)).all(), "join reordered the folds"
    target = joined["target_kms"].to_numpy().reshape(len(origins), H)
    filled = joined["was_missing"].to_numpy().reshape(len(origins), H).astype(bool)
    periods = folds["period"].to_numpy()

    recomputed = scores(pred, target, periods)
    check_scores(recomputed, receipt)

    history = pd.read_parquet(pinned("history"))
    speed = history.set_index("timestamp_utc")["filled_speed_kms"]
    speed.index = pd.DatetimeIndex(speed.index, tz="UTC")
    missing = history.set_index("timestamp_utc")["was_missing"].to_numpy().astype(bool)
    missing = pd.Series(missing, index=speed.index)
    position = speed.index.get_indexer(origins)
    assert (position >= INPUT_HOURS - 1).all() and (position < len(speed)).all(), "origin outside the history series"
    window = position[:, None] + np.arange(-INPUT_HOURS + 1, 1)[None, :]
    history_window = speed.to_numpy()[window]
    history_filled = missing.to_numpy()[window]
    assert np.isfinite(history_window).all(), "history window contains a gap"

    months = (origins + pd.Timedelta(hours=1)).strftime("%Y-%m")
    counts = pd.Series(1, index=months).groupby(level=0).size().to_dict()
    assert counts == MONTHS, f"monthly fold counts {counts} != {MONTHS}"
    return dict(origins=origins, predictions=pred, target=target, filled=filled, periods=periods,
                history=history_window, history_filled=history_filled, months=np.asarray(months),
                scores=recomputed, receipt=receipt, folds=folds)


def figure(footer: str, dpi: float = 100):
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=dpi)
    ax = fig.add_axes([0.058, 0.115, 0.925, 0.735])
    ax.set_xlim(-INPUT_HOURS + 1, H)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xlabel("hours relative to origin (t0, last observed hour)", fontsize=9)
    ax.set_ylabel("solar wind speed (km/s)", fontsize=9)
    ax.set_xticks(np.arange(-168, 73, 24))
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=8.5)
    ax.axvline(0, color=COLOR["origin"], lw=.9, ls=":")
    hist_x = np.arange(-INPUT_HOURS + 1, 1)
    future_x = np.arange(1, H + 1)
    lines = {
        "history": ax.plot(hist_x, np.full(len(hist_x), np.nan), color=COLOR["history"], lw=1.3,
                           label=f"observed, h-{INPUT_HOURS - 1} … h0")[0],
        "history_filled": ax.plot([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=COLOR["history"],
                                  mew=.8, label="observed but filled (missing in the source)")[0],
        "truth": ax.plot(future_x, np.full(len(future_x), np.nan), color=COLOR["truth"], lw=1.3,
                         label=f"truth, h+1 … h+{H}")[0],
        "truth_filled": ax.plot([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=COLOR["truth"], mew=.8,
                                label="truth but filled")[0],
        "forecast": ax.plot(future_x, np.full(len(future_x), np.nan), color=COLOR["forecast"], lw=1.6,
                            label=f"submitted forecast, h+1 … h+{H}")[0],
    }
    ax.legend(fontsize=7.6, loc="upper left", ncol=3, framealpha=.86, handlelength=1.7, columnspacing=1.1)
    title = fig.text(0.058, 0.965, "", fontsize=10, va="top")
    subtitle = fig.text(0.058, 0.915, "", fontsize=9, va="top", color="#444444")
    note = fig.text(0.058, 0.022, footer, fontsize=7.4, color="#666666")
    return fig, lines, (title, subtitle, note)


def frames(data: dict, indices, experiment: str, footer: str, dpi: float = 100):
    """Yield one palette-quantised frame per fold index."""
    fig, lines, texts = figure(footer, dpi)
    hist_x = np.arange(-INPUT_HOURS + 1, 1)
    future_x = np.arange(1, H + 1)
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
        texts[0].set_text(f"{experiment} — fold {index + 1:,} / {len(data['origins']):,} · "
                          f"{data['periods'][index]}")
        texts[1].set_text(f"origin (t0) {origin.strftime('%Y-%m-%d %H:%M')} UTC · 72 h RMSE {rmse:.1f} km/s")
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        image = Image.fromarray(buffer)
        if palette is None:
            palette = image.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        yield image.quantize(palette=palette, dither=Image.Dither.NONE)
    plt.close(fig)


def render_gifs(out_dir: Path, data: dict, experiment: str, footer: str, months=None, limit=None, suffix="",
                prefix: str = "validation_", stride: int = 1, dpi: float = 100, frame_ms: int = FRAME_MS) -> list:
    """Render the monthly GIFs. `stride`/`dpi`/`frame_ms` produce the compact README previews of the same frames."""
    size = (int(round(WIDTH * dpi / 100)), int(round(HEIGHT * dpi / 100)))
    outputs = []
    for month in (months or list(MONTHS)):
        indices = np.flatnonzero(data["months"] == month)[::stride]
        if limit:
            indices = indices[:limit]
        rendered = list(frames(data, indices, experiment, footer, dpi))
        name = f"{prefix}{month}{suffix}.gif"
        buffer = io.BytesIO()
        rendered[0].save(buffer, format="GIF", save_all=True, append_images=rendered[1:],
                         duration=frame_ms, loop=0, optimize=False, disposal=1)
        payload = buffer.getvalue()
        (out_dir / name).write_bytes(payload)

        with Image.open(out_dir / name) as check:
            assert check.size == size, f"{name} is {check.size}"
            assert check.n_frames == len(rendered), f"{name} has {check.n_frames} frames"
            assert check.info.get("duration") == frame_ms
            for position, frame in enumerate(rendered):
                check.seek(position)
                assert np.array_equal(np.asarray(check.convert("RGB")), np.asarray(frame.convert("RGB"))), \
                    f"{name} frame {position} does not round-trip"
            assert check.info.get("loop") == 0
        outputs.append(dict(month=month, period=PERIOD_OF_MONTH[month], file=name, frames=len(indices),
                            first_fold=int(indices[0]) + 1, last_fold=int(indices[-1]) + 1,
                            first_origin_utc=data["origins"][indices[0]].isoformat(),
                            last_origin_utc=data["origins"][indices[-1]].isoformat(),
                            duration_ms=frame_ms * len(indices), bytes=len(payload), stride=stride, dpi=dpi,
                            frame_ms=frame_ms))
        print(json.dumps({k: outputs[-1][k] for k in ("month", "file", "frames", "bytes")}), flush=True)
    return outputs


def gif_manifest(out_dir: Path, data: dict, experiment: str, arm: str, aggregation: str, files, keys,
                 footer: str, outputs: list) -> dict:
    receipt = data["receipt"]
    return {
        "model": experiment,
        "experiment": receipt["experiment"],
        "member": receipt["team_member"],
        "submitted_utc": receipt["submitted_utc"],
        "arm": arm,
        "aggregation": aggregation,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": dict(width=WIDTH, height=HEIGHT, frame_ms=FRAME_MS, input_hours=INPUT_HOURS,
                       forecast_hours=H, x_hours=[-INPUT_HOURS + 1, H], input_x_hours=[-INPUT_HOURS + 1, 0],
                       y_limits_kms=list(Y_LIMITS), month_grouping="UTC month of origin + 1 hour",
                       palette="256-colour palette fixed from the month's first frame; no dithering"),
        "sources": {
            "predictions": [{"file": str(p)} for p in files],
            "prediction_keys": [list(k) for k in keys],
            "folds": {"file": str(pinned("folds"))},
            "truth": {"file": str(pinned("truth"))},
            "history": {"file": str(PINNED["history"])},
            "receipt": {"file": str(out_dir / "leaderboard-result.json")},
        },
        "footer": footer,
        "verification": {
            **{k: v for k, v in data["scores"].items()},
            "receipt": {k: receipt[k] for k in data["scores"]},
            "tolerance": 1e-9,
            "expected_month_counts": MONTHS,
            "rendered_frames": int(sum(o["frames"] for o in outputs)),
            "complete": bool(sum(o["frames"] for o in outputs) == len(data["origins"])),
            "all_gifs_decoded": True,
            "prediction_truth_join": "one-to-one; complete origin x horizons 1..72",
            "input_window": f"{INPUT_HOURS} hourly samples ending at the origin; filled values kept as stored",
        },
        "notes": [
            "The forecast drawn here is the archived submission itself; nothing is retrained or re-post-processed.",
            "Hollow markers identify observations filled by the input store (ACE gaps), not model output.",
            "All truth, including filled observations, contributes to the displayed RMSE.",
        ],
        "diagnostics": [{"file": receipt.name} for receipt in sorted(out_dir.glob("*_receipt.json"))],
        "outputs": outputs,
    }
