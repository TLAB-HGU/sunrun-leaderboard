"""Visualise the local q10/q50/q90 replacement ablation `offset_roll_quantile` (q50_pinball, seeds 42/43/44).

Not a leaderboard submission: the runs live in research/offset_roll_quantile_20261105 and were scored locally with the
leaderboard scorer (leaderboard/scorer/competition.py). Reads only stored artifacts; nothing is retrained or
re-post-processed:
- runs/q50_pinball_s{seed}/predictions.parquet (pred = q50), quantiles.parquet (q10/q50/q90), submission.json,
  distribution.json, F{1,2,3}_metadata.json,
- reference/baseline_s{seed}/no_offset_predictions.parquet (control A - offset) and paired_no_offset_score.json,
- analysis/results.json (paired bootstrap of the preregistered comparison),
- competition-observations.parquet (observation revision 920588c2a7..., ACE status 0, unfilled).

Every q50 forecast is re-scored and nothing is drawn unless Total/F1/F2/F3 S, TIMSE, RMSE, tRMSE, bias and n_pairs
equal the run's submission.json to 1e-9; the control is checked against its stored paired score the same way.
The GIFs show seed 42 only (fixed before the results: the seed of the leaderboard SOTA arm, not the best seed).

Outputs (this folder): run-result.json, seed_comparison.csv/.png, lead_metrics.csv, lead_curves.png,
quantile_diagnostics.png, forecast_2026-MM.gif (nine months, every scored origin), manifest.json, README.md.

    CUDA_VISIBLE_DEVICES='' /home/t-lab01/sunrun/.venv/bin/python \
        docs/visualizations/offset_roll_quantile_q50_pinball/code/render.py
    (add --months 2026-01 --limit 8 --suffix _smoke for a quick check)
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path("/home/t-lab01/sunrun")
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "leaderboard/scorer"))
sys.path.insert(0, str(OUT.parent / "code"))
from competition import VERSION, align_predictions, prepare_context, score_competition  # noqa: E402
from submission_viz import FRAME_MS, HEIGHT, INPUT_HOURS, WIDTH, Y_LIMITS  # noqa: E402
from submission_viz import COLOR as BASE_COLOR  # noqa: E402

STUDY = ROOT / "research/offset_roll_quantile_20261105"
SEEDS, GIF_SEED, H = (42, 43, 44), 42, 72
SCOPES = ("total", "f1", "f2", "f3")
METRICS = ("s", "timse", "rmse", "trmse", "bias")
LEVELS = ("q10", "q50", "q90")
OBS_REVISION = "920588c2a716130e19b145acbbd91ace4758faf2"
COLOR = dict(BASE_COLOR, band="#f4a261", control="#5e3c99")
SEED_COLOR = {42: "#e8620c", 43: "#0072b2", 44: "#009e73"}


def check(recomputed: dict, stored: dict, label: str) -> None:
    assert recomputed["version"] == stored["version"] == VERSION, f"{label}: scorer version"
    for scope in SCOPES:
        for metric in METRICS:
            a, b = recomputed[scope][metric], stored[scope][metric]
            assert abs(a - b) < 1e-9, f"{label} {scope}.{metric}: recomputed {a!r} != stored {b!r}"
        assert recomputed[scope]["n_pairs"] == stored[scope]["n_pairs"], f"{label} {scope}.n_pairs"


def column_grid(frame: pd.DataFrame, column: str, context: dict) -> np.ndarray:
    keys = frame[["origin_last_input_utc", "horizon_hours"]].assign(pred_kms=frame[column].to_numpy())
    return align_predictions(keys, context)


def load() -> dict:
    observations = pd.read_parquet(STUDY / "competition-observations.parquet")
    context = prepare_context(observations)
    truth = context["truth"]
    runs, controls = {}, {}
    for seed in SEEDS:
        run = STUDY / f"runs/q50_pinball_s{seed}"
        assert int((run / "train.exit").read_text().strip()) == 0, f"s{seed}: train.exit != 0"
        record = json.loads((run / "submission.json").read_text())
        assert record.get("published") is False and record.get("observation_revision") == OBS_REVISION
        pred = pd.read_parquet(run / "predictions.parquet")
        scores = score_competition(pred, context)
        check(scores, record["competition"], f"q50 s{seed}")
        assert scores["total"]["n_replaced"] == 0
        quantiles = pd.read_parquet(run / "quantiles.parquet")
        q = np.stack([column_grid(quantiles, f"{level}_kms", context) for level in LEVELS])
        assert np.array_equal(q[1], align_predictions(pred, context)), f"s{seed}: pred_kms != q50"
        assert (q[0] <= q[1]).all() and (q[1] <= q[2]).all(), f"s{seed}: quantiles cross"
        metadata = [json.loads((run / f"{p}_metadata.json").read_text()) for p in ("F1", "F2", "F3")]
        runs[seed] = dict(record=record, scores=scores, q=q, metadata=metadata,
                          distribution=json.loads((run / "distribution.json").read_text()))
        reference = STUDY / f"reference/baseline_s{seed}"
        control = pd.read_parquet(reference / "no_offset_predictions.parquet")
        control_scores = score_competition(control, context)
        check(control_scores, json.loads((reference / "paired_no_offset_score.json").read_text())["competition"],
              f"A - offset s{seed}")
        controls[seed] = dict(scores=control_scores, grid=align_predictions(control, context))
    speed = observations.set_index(pd.DatetimeIndex(observations["timestamp_utc"]))["speed_kms"].sort_index()
    hours = pd.date_range(speed.index[0], speed.index[-1], freq="h")
    speed = speed.reindex(hours)
    origins = context["origins"]
    position = hours.get_indexer(origins)
    assert (position >= INPUT_HOURS - 1).all(), "observations do not cover 168 h before every origin"
    history = speed.to_numpy()[position[:, None] + np.arange(-INPUT_HOURS + 1, 1)[None, :]]
    analysis = json.loads((STUDY / "analysis/results.json").read_text())
    assert analysis["observation_revision"] == OBS_REVISION
    for seed in SEEDS:
        assert abs(analysis["official"][f"q50_s{seed}"]["total"]["s"] - runs[seed]["scores"]["total"]["s"]) < 1e-9
    return dict(context=context, truth=truth, origins=origins, period=context["period"], runs=runs,
                controls=controls, history=history, analysis=analysis,
                months=np.asarray((origins + pd.Timedelta(hours=1)).strftime("%Y-%m")))


def seed_outputs(data: dict, out: Path) -> pd.DataFrame:
    official = data["analysis"]["official"]
    primary = data["analysis"]["primary_vs_a_minus_offset"]
    versus_a = data["analysis"]["secondary_vs_a"]
    rows = []
    for seed in SEEDS:
        for arm, label in (("q50", "q50_pinball"), ("a_minus_offset", "A - offset"), ("a", "A (offset_roll)")):
            o = official[f"{arm}_s{seed}"]
            rows.append(dict(seed=seed, arm=label, **{f"{scope}_{m}": o[scope][m] for scope in SCOPES
                                                       for m in ("s", "rmse", "trmse", "mae", "bias")}))
    table = pd.DataFrame(rows)
    table.to_csv(out / "seed_comparison.csv", index=False, float_format="%.6f")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=100)
    width = .26
    for i, (arm, color) in enumerate((("q50_pinball", COLOR["forecast"]), ("A - offset", COLOR["control"]),
                                      ("A (offset_roll)", "#888888"))):
        values = table[table.arm == arm].set_index("seed").loc[list(SEEDS), "total_s"]
        bars = axes[0].bar(np.arange(3) + (i - 1) * width, values, width, color=color, label=arm)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2)
    axes[0].set_xticks(range(3), [f"seed {s}" for s in SEEDS])
    low = table.total_s.min()
    axes[0].set_ylim(low - 3, table.total_s.max() + 3)
    axes[0].set_ylabel("Total S (km/s, lower is better)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    axes[0].grid(axis="y", alpha=.25)
    for k, (title, contrast, color) in enumerate((("vs A - offset (primary)", primary, COLOR["control"]),
                                                  ("vs A", versus_a, "#888888"))):
        per_seed = [contrast["per_seed"][str(s)]["total"]["s"] if str(s) in contrast["per_seed"]
                    else contrast["per_seed"][s]["total"]["s"] for s in SEEDS]
        for j, (seed, value) in enumerate(zip(SEEDS, per_seed)):
            axes[1].scatter(k + (j - 1) * .08, value, color=SEED_COLOR[seed], zorder=3, s=28,
                            label=f"seed {seed}" if k == 0 else None)
        lo, hi = contrast["ci95_mean_delta_s"]
        mean = contrast["mean_delta_s"]
        axes[1].errorbar(k + .25, mean, yerr=[[mean - lo], [hi - mean]], fmt="s", color=color, capsize=5, zorder=3)
        axes[1].annotate(f"mean {mean:+.2f}\n95% CI [{lo:+.2f}, {hi:+.2f}]", (k + .32, mean), fontsize=8,
                         va="center")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].axhline(0, color="#111111", lw=.9)
    axes[1].set_xticks([0, 1], ["vs A - offset\n(primary)", "vs A"])
    axes[1].set_xlim(-.4, 1.9)
    axes[1].set_ylabel("ΔS = S(q50) − S(reference)")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].set_title("dots: seeds 42/43/44 · square: 3-seed mean, 168 h block bootstrap CI", fontsize=9)
    fig.suptitle("offset_roll_quantile — q50_pinball vs point-model references (official scorer, F1–F3)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "seed_comparison.png")
    plt.close(fig)
    return table


def lead_outputs(data: dict, out: Path) -> None:
    rows = []
    for seed in SEEDS:
        for arm, scores in (("q50_pinball", data["runs"][seed]["scores"]),
                            ("A - offset", data["controls"][seed]["scores"])):
            for scope in SCOPES:
                rows += [dict(seed=seed, arm=arm, scope=scope, **r) for r in scores[scope]["by_lead"]]
    leads = pd.DataFrame(rows)
    leads.to_csv(out / "lead_metrics.csv", index=False)
    total = leads[leads.scope == "total"]
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 11.2), dpi=100, sharex=True)
    for ax, metric, label in zip(axes, ("s", "rmse", "bias"), ("S (km/s)", "RMSE (km/s)", "bias = mean(pred − truth) (km/s)")):
        for seed in SEEDS:
            for arm, style in (("q50_pinball", "-"), ("A - offset", "--")):
                part = total[(total.seed == seed) & (total.arm == arm)]
                ax.plot(part.lead_hours, part[metric], ls=style, lw=1.4, color=SEED_COLOR[seed],
                        label=f"{arm} s{seed}")
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[2].axhline(0, color="#111111", lw=.8)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(ncol=3, loc="lower right", frameon=False, fontsize=8)
    axes[-1].set_xlabel("Lead time (hours)")
    axes[-1].set_xlim(1, 72)
    axes[-1].set_xticks([1, 12, 24, 36, 48, 60, 72])
    fig.suptitle("Total lead curves — q50_pinball (solid) vs A − offset (dashed), official scorer", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "lead_curves.png")
    plt.close(fig)


def quantile_outputs(data: dict, out: Path) -> dict:
    truth = data["truth"]
    valid = np.isfinite(truth)
    leads = np.arange(1, H + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), dpi=100)
    per_lead = {}
    for seed in SEEDS:
        lo, mid, hi = data["runs"][seed]["q"]
        inside = np.where(valid, (truth >= lo) & (truth <= hi), np.nan)
        below = np.where(valid, truth < lo, np.nan)
        above = np.where(valid, truth > hi, np.nan)
        width = np.where(valid, hi - lo, np.nan)
        per_lead[seed] = dict(coverage=np.nanmean(inside, 0), below_q10=np.nanmean(below, 0),
                              above_q90=np.nanmean(above, 0), width=np.nanmean(width, 0))
        c = SEED_COLOR[seed]
        axes[0, 0].plot(leads, per_lead[seed]["coverage"], color=c, lw=1.4, label=f"s{seed}")
        axes[0, 1].plot(leads, per_lead[seed]["below_q10"], color=c, lw=1.4, ls="--")
        axes[0, 1].plot(leads, per_lead[seed]["above_q90"], color=c, lw=1.4)
        axes[1, 0].plot(leads, per_lead[seed]["width"], color=c, lw=1.4)
    axes[0, 0].axhline(.8, color="#111111", lw=.9, ls=":", label="nominal 80 %")
    axes[0, 0].set_title("[q10, q90] coverage by lead", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].axhline(.1, color="#111111", lw=.9, ls=":")
    axes[0, 1].set_title("truth below q10 (dashed) / above q90 (solid), nominal 10 %", fontsize=10)
    axes[1, 0].set_title("mean q90 − q10 width (km/s)", fontsize=10)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlim(1, 72)
        ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
        ax.set_xlabel("Lead time (hours)")
        ax.grid(alpha=.25)
    groups = ("total", "f1", "f2", "f3", "h1-6", "h7-24", "h25-72", "y≥550")
    for i, seed in enumerate(SEEDS):
        d = data["runs"][seed]["distribution"]
        values = [d[g]["coverage_q10_q90"] for g in groups[:-1]] + [d["total"]["y_ge_550"]["coverage_q10_q90"]]
        axes[1, 1].bar(np.arange(len(groups)) + (i - 1) * .26, values, .26, color=SEED_COLOR[seed], label=f"s{seed}")
    axes[1, 1].axhline(.8, color="#111111", lw=.9, ls=":")
    axes[1, 1].set_xticks(range(len(groups)), [g.upper() if g.startswith("f") else g for g in groups], fontsize=8)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("[q10, q90] coverage by period / lead group / fast wind", fontsize=10)
    axes[1, 1].grid(axis="y", alpha=.25)
    fig.suptitle("q50_pinball — interval diagnostics (truth = ACE status 0, unfilled; descriptive, not a gate)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "quantile_diagnostics.png")
    plt.close(fig)
    return {f"s{s}": {k: [round(float(x), 6) for x in v] for k, v in per_lead[s].items()} for s in SEEDS}


def frames(data: dict, indices, footer: str):
    run, control = data["runs"][GIF_SEED], data["controls"][GIF_SEED]
    lo_all, mid_all, hi_all = run["q"]
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
                     label="observed (ACE status 0; gaps blank)")[0]
    line_t = ax.plot(future_x, np.full(H, np.nan), color=COLOR["truth"], lw=1.3, marker=".", ms=2.6,
                     label="truth h+1 … h+72")[0]
    line_c = ax.plot(future_x, np.full(H, np.nan), color=COLOR["control"], lw=1.2, ls="--",
                     label="A − offset (point model)")[0]
    line_f = ax.plot(future_x, np.full(H, np.nan), color=COLOR["forecast"], lw=1.6, label="q50 (scored)")[0]
    band = ax.fill_between(future_x, np.full(H, np.nan), np.full(H, np.nan), color=COLOR["band"], alpha=.35,
                           lw=0, label="q10–q90")
    ax.legend(fontsize=7.4, loc="upper left", ncol=5, framealpha=.86, handlelength=1.6, columnspacing=1.0)
    title = fig.text(0.058, 0.965, "", fontsize=10, va="top")
    subtitle = fig.text(0.058, 0.915, "", fontsize=9, va="top", color="#444444")
    fig.text(0.058, 0.022, footer, fontsize=7.0, color="#666666")
    palette = None
    for index in indices:
        truth, lo, mid, hi = data["truth"][index], lo_all[index], mid_all[index], hi_all[index]
        band.remove()
        band = ax.fill_between(future_x, lo, hi, color=COLOR["band"], alpha=.35, lw=0, zorder=1)
        line_h.set_ydata(data["history"][index])
        line_t.set_ydata(truth)
        line_c.set_ydata(control["grid"][index])
        line_f.set_ydata(mid)
        valid = np.isfinite(truth)
        if valid.any():
            rmse = float(np.sqrt(np.mean((mid[valid] - truth[valid]) ** 2)))
            inside = float(np.mean((truth[valid] >= lo[valid]) & (truth[valid] <= hi[valid])))
            stats = f"72 h q50 RMSE {rmse:.1f} km/s · {inside:.0%} of {int(valid.sum())} valid truth hours inside q10–q90"
        else:
            stats = "no valid truth hours"
        title.set_text(f"q50_pinball s{GIF_SEED} — scored origin {index + 1:,} / {len(data['origins']):,} · "
                       f"{data['period'][index]}")
        subtitle.set_text(f"origin (t0) {data['origins'][index].strftime('%Y-%m-%d %H:%M')} UTC · {stats}")
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        if palette is None:
            palette = image.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        yield image.quantize(palette=palette, dither=Image.Dither.NONE)
    plt.close(fig)


def render_gifs(data: dict, out: Path, footer: str, months=None, limit=None, suffix="") -> list:
    outputs = []
    for month in months or sorted(set(data["months"])):
        indices = np.flatnonzero(data["months"] == month)
        if limit:
            indices = indices[:limit]
        rendered = list(frames(data, indices, footer))
        name = f"forecast_{month}{suffix}.gif"
        buffer = io.BytesIO()
        rendered[0].save(buffer, format="GIF", save_all=True, append_images=rendered[1:], duration=FRAME_MS,
                         loop=0, optimize=False, disposal=1)
        (out / name).write_bytes(buffer.getvalue())
        with Image.open(out / name) as check_gif:
            assert check_gif.size == (WIDTH, HEIGHT) and check_gif.n_frames == len(rendered)
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
    parser.add_argument("--suffix", default="", help="smoke check: write only GIFs with this suffix")
    args = parser.parse_args()
    data = load()
    footer = (f"q50_pinball s{GIF_SEED} · local replacement ablation, not submitted · scored 00/06/12/18 UTC origins · "
              "truth = ACE status 0, unfilled · Y 200–850 km/s")
    if args.suffix:
        render_gifs(data, OUT, footer, args.months, args.limit, args.suffix)
        print("smoke output written; other outputs untouched")
        return
    record = data["runs"][GIF_SEED]["record"]
    (OUT / "run-result.json").write_text(json.dumps(record, indent=2) + "\n")
    seed_outputs(data, OUT)
    lead_outputs(data, OUT)
    interval = quantile_outputs(data, OUT)
    outputs = render_gifs(data, OUT, footer, args.months, args.limit)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=OUT, text=True, capture_output=True).stdout.strip()
    manifest = {
        "experiment": "offset_roll_quantile / q50_pinball", "seeds": list(SEEDS), "gif_seed": GIF_SEED,
        "submitted": False, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "metric_version": VERSION,
        "render_repository_commit": commit,
        "training_commits": {f"s{s}": sorted({m.get("git_commit") for m in data["runs"][s]["metadata"]}) for s in SEEDS},
        "sources": {"study": str(STUDY), "runs": [f"runs/q50_pinball_s{s}" for s in SEEDS],
                    "controls": [f"reference/baseline_s{s}/no_offset_predictions.parquet" for s in SEEDS],
                    "analysis": "analysis/results.json",
                    "observations": {"file": "competition-observations.parquet", "revision": OBS_REVISION}},
        "verification": {
            "tolerance": 1e-9, "all_equal_to_run_records": True, "controls_equal_to_stored_paired_scores": True,
            "q50_equals_pred_kms": True, "quantiles_ordered": True,
            "scores": {f"s{s}": {sc: {m: data["runs"][s]["scores"][sc][m] for m in METRICS + ("n_pairs", "n_origins")}
                                 for sc in SCOPES} for s in SEEDS},
            "scored_origins": int(len(data["origins"])),
            "rendered_frames": int(sum(o["frames"] for o in outputs)),
            "complete": bool(sum(o["frames"] for o in outputs) == len(data["origins"])),
        },
        "analysis": {k: data["analysis"][k] for k in ("verdict", "submission_candidate", "q50_mean_s", "sota",
                                                      "sota_gap_pct", "sota_minus_2pct_met", "bootstrap")},
        "interval_by_lead": interval,
        "config": dict(width=WIDTH, height=HEIGHT, frame_ms=FRAME_MS, input_hours=INPUT_HOURS, forecast_hours=H,
                       y_limits_kms=list(Y_LIMITS), month_grouping="UTC month of origin + 1 hour",
                       frames="every scored origin (00/06/12/18 UTC) of F1/F2/F3"),
        "footer": footer, "outputs": outputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(outputs)} GIFs, {manifest['verification']['rendered_frames']:,} frames; "
          f"complete={manifest['verification']['complete']}")


if __name__ == "__main__":
    main()
