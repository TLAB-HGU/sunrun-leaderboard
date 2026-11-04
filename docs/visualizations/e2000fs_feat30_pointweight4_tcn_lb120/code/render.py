"""Visualise the leaderboard submission of seongeun / e2000fs_feat30_pointweight4_tcn_lb120 (fresh-df ffill rerun).

Reads only stored artifacts; nothing is retrained or re-post-processed:
- the submitted forecast `predictions.parquet` and the TCN fit reports/loss logs of
  research/rerun_df_ffill_20261103/runs/e2000fs_feat30_pointweight4_tcn_lb120,
- the leaderboard record `submission.json` of that run, and
- `competition-observations.parquet` at the dataset revision the record was scored against.

The submitted forecast is re-scored with the leaderboard scorer (leaderboard/scorer/competition.py). Nothing is
drawn unless every Total/F1/F2/F3 S, TIMSE, RMSE, tRMSE and bias equals the stored record to 1e-9.

Outputs (this folder): leaderboard-result.json, lead_metrics.csv, lead_curves.png, lead_residue.csv,
loss_history.csv, loss_history.png, forecast_2026-MM.gif (nine months, every scored origin), manifest.json.

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_feat30_pointweight4_tcn_lb120/code/render.py
    (add --months 2026-01 --limit 8 --suffix _smoke for a quick check)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path("/home/t-lab01/sunrun")
sys.path.insert(0, str(ROOT / "leaderboard/scorer"))
from competition import LEADS, VERSION, align_predictions, prepare_context, score_competition  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
EXPERIMENT = "e2000fs_feat30_pointweight4_tcn_lb120"
RUN = ROOT / "research/rerun_df_ffill_20261103/runs" / EXPERIMENT
DATASET = "tlabtlab/sunrun-lb-store"
PERIODS = ("F1", "F2", "F3")
SCOPES = ("total", "f1", "f2", "f3")
METRICS = ("s", "timse", "rmse", "trmse", "bias")
INPUT_HOURS, H = 168, 72
WIDTH, HEIGHT, FRAME_MS = 1000, 500, 250
Y_LIMITS = (200.0, 850.0)
COLOR = {"history": "#1a6fb5", "truth": "#111111", "forecast": "#e8620c", "origin": "#8a8a8a",
         "total": "#111111", "f1": "#0072b2", "f2": "#d55e00", "f3": "#009e73"}
# Record fields copied into leaderboard-result.json (an allowlist: submission keys and URLs that embed them are left out).
KEEP = ("submitted_utc", "team_member", "experiment", "description", "f1_training_data_end_utc",
        "f2_training_data_end_utc", "f3_training_data_end_utc", "config", "inference_seconds_per_fold",
        "architecture_url", "competition", "competition_dataset_revision")
FOOTER = ("fresh raw-only df + causal ffill rerun · TCN, 120 h input, dilations 1–32 · scored 00/06/12/18 UTC "
          "origins · truth = ACE status 0, unfilled · Y 200–850 km/s")


def load() -> dict:
    record = json.loads((RUN / "submission.json").read_text())
    obs_path = hf_hub_download(DATASET, "competition-observations.parquet", repo_type="dataset",
                               revision=record["competition_dataset_revision"])
    observations = pd.read_parquet(obs_path)
    context = prepare_context(observations)
    pred_frame = pd.read_parquet(RUN / "predictions.parquet")
    recomputed = score_competition(pred_frame, context)
    assert recomputed["version"] == record["competition"]["version"] == VERSION
    for scope in SCOPES:
        for metric in METRICS:
            a, b = recomputed[scope][metric], record["competition"][scope][metric]
            assert abs(a - b) < 1e-9, f"{scope}.{metric}: recomputed {a!r} != record {b!r}"
        assert recomputed[scope]["n_pairs"] == record["competition"][scope]["n_pairs"]
        assert recomputed[scope]["replacement_fraction"] == 0
    predicted = align_predictions(pred_frame, context)
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    speed = observations.set_index(pd.DatetimeIndex(observations["timestamp_utc"]))["speed_kms"].sort_index()
    origins = context["origins"]
    hours = pd.date_range(speed.index[0], speed.index[-1], freq="h")
    speed = speed.reindex(hours)
    position = hours.get_indexer(origins)
    assert (position >= INPUT_HOURS - 1).all(), "observations do not cover 168 h before every origin"
    history = speed.to_numpy()[position[:, None] + np.arange(-INPUT_HOURS + 1, 1)[None, :]]
    return dict(record=record, recomputed=recomputed, origins=origins, period=context["period"],
                predicted=predicted, truth=context["truth"], history=history,
                months=np.asarray((origins + pd.Timedelta(hours=1)).strftime("%Y-%m")))


def write_record(data: dict) -> None:
    record = {k: data["record"][k] for k in KEEP}
    (OUT / "leaderboard-result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def lead_outputs(data: dict) -> dict:
    rows = []
    for scope in SCOPES:
        for row in data["recomputed"][scope]["by_lead"]:
            rows.append({"scope": scope, **row})
    leads = pd.DataFrame(rows)
    leads.to_csv(OUT / "lead_metrics.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 11.2), dpi=100, sharex=True)
    for ax, metric, label in zip(axes, ("s", "timse", "rmse"), ("S (km/s)", "TIMSE ((km/s)²)", "RMSE (km/s)")):
        for lead in range(6, 73, 6):
            ax.axvline(lead, color="#cccccc", lw=.7, zorder=0)
        for scope in SCOPES:
            part = leads[leads.scope == scope]
            ax.plot(part.lead_hours, part[metric], color=COLOR[scope], lw=2.6 if scope == "total" else 1.4,
                    label=scope.capitalize() if scope == "total" else scope.upper())
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
    axes[0].legend(ncol=4, loc="lower left", frameon=False)
    axes[-1].set_xlabel("Lead time (hours) — grey lines mark leads 6, 12, …, 72")
    axes[-1].set_xlim(1, 72)
    axes[-1].set_xticks([1, 12, 24, 36, 48, 60, 72])
    fig.suptitle(f"{EXPERIMENT} — lead curves recomputed from the submitted forecast", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "lead_curves.png")
    plt.close(fig)

    # Leads that are multiples of 6 land on 00/06/12/18 UTC, the same hours as the scored origins.
    leads["lead_mod_6"] = leads.lead_hours % 6
    residue = (leads.groupby(["scope", "lead_mod_6"])[["s", "timse", "rmse", "bias"]].mean()
               .reset_index().round(3))
    residue.to_csv(OUT / "lead_residue.csv", index=False)
    selected = data["period"] == "F3"
    truth = data["truth"][selected]
    peak_row, peak_col = np.unravel_index(int(np.nanargmax(truth)), truth.shape)
    peak_time = data["origins"][selected][peak_row] + pd.Timedelta(hours=int(LEADS[peak_col]))
    return {"f3_truth_max_kms": float(truth[peak_row, peak_col]), "f3_truth_max_utc": peak_time.isoformat(),
            "residue_file": "lead_residue.csv"}


def loss_outputs() -> list[dict]:
    frames, fits = [], []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=100)
    for ax, period in zip(axes, PERIODS):
        meta = json.loads((RUN / f"{period}_metadata.json").read_text())
        sel = pd.read_csv(RUN / f"{period}_selection_loss.csv")
        final = pd.read_csv(RUN / f"{period}_train_loss.csv")
        frames.append(sel.assign(period=period, stage="selection"))
        frames.append(final.assign(period=period, stage="final_refit"))
        best = int(meta["selected_epoch"])
        assert best == int(sel.validation_weighted_mse.idxmin()) + 1 and len(final) == best
        ax.plot(sel.epoch, sel.train_weighted_mse, "o-", color="#1a6fb5", ms=3.5, label="selection fit, train")
        ax.plot(sel.epoch, sel.validation_weighted_mse, "o-", color="#e8620c", ms=3.5,
                label="selection fit, validation")
        ax.plot(final.epoch, final.iloc[:, 1], "s", color="#111111", ms=7, label="final refit (all rows), train")
        ax.axvline(best, color="#888888", ls=":", lw=1)
        ax.set_title(f"{period} · validation {meta['selection_valid_start'][:10]} … {meta['selection_valid_end'][:10]}"
                     f"\nselected epoch {best} of {meta['epoch_budget']} (patience {meta['patience']})", fontsize=9.5)
        ax.set_xlabel("epoch")
        ax.set_xticks(sel.epoch)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
        fits.append({k: meta[k] for k in ("period", "seed", "device", "training_cutoff", "train_start", "train_end",
                                          "train_origins", "selection_train_end", "selection_valid_start",
                                          "selection_valid_end", "selected_epoch", "epoch_budget", "patience",
                                          "batch_size", "width", "dilations", "kernel", "dropout", "optimizer",
                                          "learning_rate", "history_rule", "target_rule", "versions", "git_commit")})
    axes[0].set_ylabel("point4-weighted MSE ((km/s)²)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{EXPERIMENT} — TCN training loss (selection fit with early stopping, then a refit of the "
                 "selected epoch count on all eligible rows)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "loss_history.png")
    plt.close(fig)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "loss_history.csv", index=False)
    return fits


def frames(data: dict, indices):
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = fig.add_axes([0.058, 0.115, 0.925, 0.735])
    ax.set_xlim(-INPUT_HOURS + 1, H)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xlabel("hours relative to origin (t0, last observed hour)", fontsize=9)
    ax.set_ylabel("solar wind speed (km/s)", fontsize=9)
    ax.set_xticks(np.arange(-168, 73, 24))
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=8.5)
    ax.axvline(0, color=COLOR["origin"], lw=.9, ls=":")
    hist_x, future_x = np.arange(-INPUT_HOURS + 1, 1), np.arange(1, H + 1)
    line_h = ax.plot(hist_x, np.full(len(hist_x), np.nan), color=COLOR["history"], lw=1.3, marker=".", ms=2.6,
                     label="observed (ACE status 0; gaps left blank)")[0]
    line_t = ax.plot(future_x, np.full(H, np.nan), color=COLOR["truth"], lw=1.3, marker=".", ms=2.6,
                     label="truth h+1 … h+72")[0]
    line_f = ax.plot(future_x, np.full(H, np.nan), color=COLOR["forecast"], lw=1.6,
                     label="submitted forecast h+1 … h+72")[0]
    ax.legend(fontsize=7.6, loc="upper left", ncol=3, framealpha=.86, handlelength=1.7, columnspacing=1.1)
    title = fig.text(0.058, 0.965, "", fontsize=10, va="top")
    subtitle = fig.text(0.058, 0.915, "", fontsize=9, va="top", color="#444444")
    fig.text(0.058, 0.022, FOOTER, fontsize=7.0, color="#666666")
    palette = None
    for index in indices:
        truth, forecast = data["truth"][index], data["predicted"][index]
        line_h.set_ydata(data["history"][index])
        line_t.set_ydata(truth)
        line_f.set_ydata(forecast)
        valid = np.isfinite(truth)
        rmse = float(np.sqrt(np.mean((forecast[valid] - truth[valid]) ** 2))) if valid.any() else float("nan")
        title.set_text(f"{EXPERIMENT} — scored origin {index + 1:,} / {len(data['origins']):,} · "
                       f"{data['period'][index]}")
        subtitle.set_text(f"origin (t0) {data['origins'][index].strftime('%Y-%m-%d %H:%M')} UTC · "
                          f"72 h RMSE {rmse:.1f} km/s over {int(valid.sum())} valid truth hours")
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        if palette is None:
            palette = image.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        yield image.quantize(palette=palette, dither=Image.Dither.NONE)
    plt.close(fig)


def render_gifs(data: dict, months=None, limit=None, suffix="") -> list[dict]:
    outputs = []
    for month in months or sorted(set(data["months"])):
        indices = np.flatnonzero(data["months"] == month)
        if limit:
            indices = indices[:limit]
        rendered = list(frames(data, indices))
        name = f"forecast_{month}{suffix}.gif"
        buffer = io.BytesIO()
        rendered[0].save(buffer, format="GIF", save_all=True, append_images=rendered[1:], duration=FRAME_MS,
                         loop=0, optimize=False, disposal=1)
        (OUT / name).write_bytes(buffer.getvalue())
        with Image.open(OUT / name) as check:
            assert check.size == (WIDTH, HEIGHT) and check.n_frames == len(rendered)
        outputs.append(dict(month=month, period=str(data["period"][indices[0]]), file=name, frames=len(indices),
                            first_origin_utc=data["origins"][indices[0]].isoformat(),
                            last_origin_utc=data["origins"][indices[-1]].isoformat(),
                            bytes=len(buffer.getvalue()), frame_ms=FRAME_MS))
        print(json.dumps({k: outputs[-1][k] for k in ("month", "file", "frames", "bytes")}), flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    data = load()
    if args.suffix:
        render_gifs(data, args.months, args.limit, args.suffix)
        print("smoke output written; other outputs untouched")
        return
    write_record(data)
    diagnostic = lead_outputs(data)
    fits = loss_outputs()
    outputs = render_gifs(data, args.months, args.limit)
    rec = data["record"]
    manifest = {
        "experiment": EXPERIMENT, "member": rec["team_member"], "submitted_utc": rec["submitted_utc"],
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "metric_version": VERSION,
        "sources": {"run": str(RUN), "predictions": str(RUN / "predictions.parquet"),
                    "record": str(RUN / "submission.json"),
                    "observations": {"dataset": DATASET, "file": "competition-observations.parquet",
                                     "revision": rec["competition_dataset_revision"]}},
        "verification": {
            "tolerance": 1e-9,
            "scores": {scope: {m: data["recomputed"][scope][m] for m in METRICS + ("n_pairs", "n_origins")}
                       for scope in SCOPES},
            "all_equal_to_record": True,
            "scored_origins": int(len(data["origins"])),
            "rendered_frames": int(sum(o["frames"] for o in outputs)),
            "complete": bool(sum(o["frames"] for o in outputs) == len(data["origins"])),
        },
        "fits": fits, "lead_diagnostic": diagnostic,
        "config": dict(width=WIDTH, height=HEIGHT, frame_ms=FRAME_MS, input_hours=INPUT_HOURS, forecast_hours=H,
                       y_limits_kms=list(Y_LIMITS), month_grouping="UTC month of origin + 1 hour",
                       frames="every scored origin (00/06/12/18 UTC) of F1/F2/F3"),
        "footer": FOOTER, "outputs": outputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(outputs)} GIFs, {manifest['verification']['rendered_frames']:,} frames; "
          f"complete={manifest['verification']['complete']}")


if __name__ == "__main__":
    main()
