"""Prediction visualisation of one stored leaderboard submission, for the board's `visualization` column.

    /home/t-lab01/sunrun/.venv/bin/python tools/submission_viz.py \
        --member seongeun --experiment e2000fs_calib3_era --submitted-utc 20261104T013907870Z \
        --out research/e2000fs_calib3_era/visualization --rank 3 --rank-total 41 --board-revision 68490c61e0

Reads only what the leaderboard dataset stores, so it works for every member: the result record and its submitted
prediction parquet (paired inside the dataset, nothing derived from them), and `competition-observations.parquet` at
the dataset revision the record was scored against. The forecast is re-scored with the leaderboard scorer
(leaderboard/scorer/competition.py) and nothing is drawn unless every Total/F1/F2/F3 S, TIMSE, RMSE, tRMSE and bias
equals the stored record to 1e-9. Nothing is retrained or re-post-processed.

Outputs in --out: README.md, leaderboard-result.json, lead_metrics.csv, lead_curves.png, lead_residue.csv,
forecast_2026-MM.gif (nine months, every scored 00/06/12/18 UTC origin), manifest.json.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path("/home/t-lab01/sunrun")
sys.path.insert(0, str(ROOT / "leaderboard/scorer"))
from competition import LEADS, VERSION, align_predictions, prepare_context, score_competition  # noqa: E402

DATASET = "tlabtlab/sunrun-lb-store"
SCOPES = ("total", "f1", "f2", "f3")
METRICS = ("s", "timse", "rmse", "trmse", "bias")
INPUT_HOURS, H = 168, 72
WIDTH, HEIGHT, FRAME_MS = 1000, 500, 250
Y_LIMITS = (200.0, 850.0)
COLOR = {"history": "#1a6fb5", "truth": "#111111", "forecast": "#e8620c", "origin": "#8a8a8a",
         "total": "#111111", "f1": "#0072b2", "f2": "#d55e00", "f3": "#009e73"}
# Record fields copied into leaderboard-result.json: an allowlist, so submission keys and URLs carrying them stay out.
KEEP = ("submitted_utc", "team_member", "experiment", "description", "f1_training_data_end_utc",
        "f2_training_data_end_utc", "f3_training_data_end_utc", "config", "inference_seconds_per_fold",
        "architecture_url", "competition", "competition_dataset_revision")
# Submitters' configs may carry file digests; this folder stores none (workspace rule "No Hashes").
DIGEST_KEY = re.compile(r"sha\d{3}|digest|checksum|hash", re.I)
DIGEST_VALUE = re.compile(r"[0-9a-f]{40,}", re.I)


def scrub(value):
    """Copy of a record with every digest-named key and digest-like string value left out."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()
                if not DIGEST_KEY.search(k) and not (isinstance(v, str) and DIGEST_VALUE.fullmatch(v))}
    if isinstance(value, list):
        return [scrub(v) for v in value if not (isinstance(v, str) and DIGEST_VALUE.fullmatch(v))]
    return value


# The two fastest F3 observations; both fall on 00/06/12/18 UTC, i.e. only at leads that are multiples of 6.
F3_PEAKS = ("2026-09-15T18:00:00Z", "2026-07-23T18:00:00Z")


def stored_submission(member: str, experiment: str, submitted_utc: str) -> tuple[dict, Path]:
    """The result record with this submission time and the prediction parquet stored with it."""
    api = HfApi()
    revision = api.repo_info(DATASET, repo_type="dataset").sha
    files = api.list_repo_files(DATASET, repo_type="dataset", revision=revision)
    prefix = f"{member}/{experiment}/"
    for name in (f for f in files if f.startswith("results/" + prefix) and f.endswith(".json")):
        record = json.loads(Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=revision)).read_text())
        if record.get("submitted_utc") == submitted_utc:
            stem = name[len("results/"):-len(".json")]
            parquet = f"submissions/{stem}.parquet"
            if parquet not in files:
                raise FileNotFoundError(f"no stored prediction parquet beside the {member}/{experiment} record")
            return record, Path(hf_hub_download(DATASET, parquet, repo_type="dataset", revision=revision))
    raise LookupError(f"no stored {member}/{experiment} result submitted at {submitted_utc}")


def mismatch(recomputed: dict, record: dict) -> str | None:
    """First field where the recomputed scores differ from the stored record (1e-9), or None."""
    if not recomputed["version"] == record["competition"]["version"] == VERSION:
        return f"version {recomputed['version']!r} != {record['competition']['version']!r}"
    for scope in SCOPES:
        for metric in METRICS:
            a, b = recomputed[scope][metric], record["competition"][scope][metric]
            if not abs(a - b) < 1e-9:
                return f"{scope}.{metric}: recomputed {a!r} != record {b!r}"
        if recomputed[scope]["n_pairs"] != record["competition"][scope]["n_pairs"]:
            return f"{scope}.n_pairs differs"
    return None


def observation_revisions(record: dict) -> list[str]:
    """The revision the record was scored against; records older than that field (the baseline rows, rescored
    2026-10-31) get every dataset commit, newest first, and load() keeps the first that reproduces the record."""
    if record.get("competition_dataset_revision"):
        return [record["competition_dataset_revision"]]
    return [c.commit_id for c in HfApi().list_repo_commits(DATASET, repo_type="dataset")]


def load(member: str, experiment: str, submitted_utc: str) -> dict:
    record, pred_path = stored_submission(member, experiment, submitted_utc)
    pred_frame = pd.read_parquet(pred_path)
    seen, reason = set(), "no dataset revision tried"
    for revision in observation_revisions(record):
        path = Path(hf_hub_download(DATASET, "competition-observations.parquet", repo_type="dataset",
                                    revision=revision))
        if path.resolve() in seen:  # same file content as a revision already tried
            continue
        seen.add(path.resolve())
        observations = pd.read_parquet(path)
        context = prepare_context(observations)
        recomputed = score_competition(pred_frame, context)
        reason = mismatch(recomputed, record)
        if reason is None:
            break
    else:
        raise AssertionError(reason)
    predicted = align_predictions(pred_frame, context)
    replaced = ~np.isfinite(predicted) | (predicted <= 0)
    predicted = np.where(replaced, context["persistence"][:, None], predicted)  # the scorer's own replacement rule
    speed = observations.set_index(pd.DatetimeIndex(observations["timestamp_utc"]))["speed_kms"].sort_index()
    hours = pd.date_range(speed.index[0], speed.index[-1], freq="h")
    speed = speed.reindex(hours)
    origins = context["origins"]
    position = hours.get_indexer(origins)
    assert (position >= INPUT_HOURS - 1).all(), "observations do not cover 168 h before every origin"
    history = speed.to_numpy()[position[:, None] + np.arange(-INPUT_HOURS + 1, 1)[None, :]]
    # Diagnostic only (not a score): F3 lead TIMSE with the two synoptic-hour peaks left out of the truth.
    trimmed = observations.copy()
    trimmed.loc[pd.to_datetime(trimmed["timestamp_utc"], utc=True).isin(pd.to_datetime(list(F3_PEAKS), utc=True)),
                "speed_kms"] = np.nan
    without_peaks = score_competition(pred_frame, prepare_context(trimmed))["f3"]["by_lead"]
    return dict(record=record, observation_revision=revision, recomputed=recomputed, origins=origins,
                period=context["period"],
                predicted=predicted, replaced=replaced, truth=context["truth"], history=history,
                without_peaks=without_peaks, months=np.asarray((origins + pd.Timedelta(hours=1)).strftime("%Y-%m")))


def write_record(record: dict, out: Path) -> None:
    kept = scrub({k: record[k] for k in KEEP if k in record})
    (out / "leaderboard-result.json").write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n")


def lead_outputs(data: dict, out: Path, experiment: str) -> dict:
    leads = pd.DataFrame([{"scope": scope, **row} for scope in SCOPES for row in data["recomputed"][scope]["by_lead"]])
    leads.to_csv(out / "lead_metrics.csv", index=False)
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
    fig.suptitle(f"{experiment} — lead curves recomputed from the submitted forecast", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "lead_curves.png")
    plt.close(fig)
    leads["lead_mod_6"] = leads.lead_hours % 6
    residue = leads.groupby(["scope", "lead_mod_6"])[["s", "timse", "rmse", "bias"]].mean().reset_index().round(3)
    residue.to_csv(out / "lead_residue.csv", index=False)
    f3 = residue[residue.scope == "f3"].set_index("lead_mod_6").timse
    trimmed = pd.DataFrame(data["without_peaks"])
    trimmed_6 = float(trimmed[trimmed.lead_hours % 6 == 0].timse.mean())
    return {"f3_timse_lead_6k": float(f3[0]), "f3_timse_other_min": float(f3.drop(0).min()),
            "f3_timse_other_max": float(f3.drop(0).max()), "f3_timse_lead_6k_without_peaks": trimmed_6,
            "peaks_left_out": list(F3_PEAKS), "note": "diagnostic only; not a leaderboard score"}


def frames(data: dict, indices, experiment: str, footer: str):
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
    fig.text(0.058, 0.022, footer, fontsize=7.0, color="#666666")
    palette = None
    for index in indices:
        truth, forecast = data["truth"][index], data["predicted"][index]
        line_h.set_ydata(data["history"][index])
        line_t.set_ydata(truth)
        line_f.set_ydata(forecast)
        valid = np.isfinite(truth)
        rmse = float(np.sqrt(np.mean((forecast[valid] - truth[valid]) ** 2))) if valid.any() else float("nan")
        replaced = int(data["replaced"][index].sum())
        title.set_text(f"{experiment} — scored origin {index + 1:,} / {len(data['origins']):,} · {data['period'][index]}")
        subtitle.set_text(f"origin (t0) {data['origins'][index].strftime('%Y-%m-%d %H:%M')} UTC · "
                          f"72 h RMSE {rmse:.1f} km/s over {int(valid.sum())} valid truth hours"
                          + (f" · {replaced} leads persistence-replaced" if replaced else ""))
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        if palette is None:
            palette = image.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        yield image.quantize(palette=palette, dither=Image.Dither.NONE)
    plt.close(fig)


def render_gifs(data: dict, out: Path, experiment: str, footer: str, months=None, limit=None, suffix="") -> list:
    outputs = []
    for month in months or sorted(set(data["months"])):
        indices = np.flatnonzero(data["months"] == month)
        if limit:
            indices = indices[:limit]
        rendered = list(frames(data, indices, experiment, footer))
        name = f"forecast_{month}{suffix}.gif"
        buffer = io.BytesIO()
        rendered[0].save(buffer, format="GIF", save_all=True, append_images=rendered[1:], duration=FRAME_MS,
                         loop=0, optimize=False, disposal=1)
        (out / name).write_bytes(buffer.getvalue())
        with Image.open(out / name) as check:
            assert check.size == (WIDTH, HEIGHT) and check.n_frames == len(rendered)
        outputs.append(dict(month=month, period=str(data["period"][indices[0]]), file=name, frames=len(indices),
                            first_origin_utc=data["origins"][indices[0]].isoformat(),
                            last_origin_utc=data["origins"][indices[-1]].isoformat(),
                            bytes=len(buffer.getvalue()), frame_ms=FRAME_MS))
        print(json.dumps({k: outputs[-1][k] for k in ("month", "file", "frames", "bytes")}), flush=True)
    return outputs


def readme(args, data: dict, diagnostic: dict, outputs: list) -> str:
    rec, sc = data["record"], data["recomputed"]
    n_all = len(data["origins"]) * H
    rows = "\n".join(
        f"| {'**Total**' if s == 'total' else s.upper() + ' (' + p + ')'} | {sc[s]['s']:.2f} | {sc[s]['timse']:,.1f} | "
        f"{sc[s]['rmse']:.2f} | {sc[s]['trmse']:.2f} | {sc[s]['bias']:+.1f} | {sc[s]['n_origins']:,} | "
        f"{sc[s]['n_pairs']:,} | {sc[s]['replacement_fraction']:.2%} |"
        for s, p in zip(SCOPES, ("", "2026-01..03", "2026-04..06", "2026-07..09")))
    gifs = "\n".join(f"| {o['month']} | {o['period']} | {o['frames']} | ![{o['month']}]({o['file']}) |" for o in outputs)
    arch = rec.get("architecture_url")
    arch_line = f"- 모델 정의 코드: [architecture]({arch})\n" if arch else ""
    rank = (f"게시 시점(2026-11-04, Space `{args.board_revision}`) 리더보드 **{args.rank}위 / {args.rank_total}**, "
            if args.rank else "")
    description = rec.get("description", "").replace("\n", " ").strip()
    return f"""# {args.experiment} — lead 곡선 · F1–F3 월별 예측 GIF

작성자 **{args.member}**의 `{args.experiment}`입니다(제출 `{args.submitted_utc}`). {rank}Total S **{sc['total']['s']:.2f}**입니다.

이 페이지는 리더보드 dataset에 저장된 **제출 예측 자체**를 리더보드 채점기(`leaderboard/scorer/competition.py`, `{VERSION}`)로 다시 채점해 그렸습니다. 관측은 기록이 채점된 dataset revision `{data['observation_revision'][:10]}`의 `competition-observations.parquet`입니다. Total/F1/F2/F3의 S·TIMSE·RMSE·tRMSE·bias가 저장 기록과 **1e-9 이내로 같을 때만** 그림을 그립니다([manifest.json](manifest.json)). 재학습이나 후처리는 하지 않았습니다.

| 구간 | S | TIMSE | RMSE | tRMSE | bias | 채점 origin | 유효 pair | persistence 대체 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

단위는 km/s(TIMSE는 (km/s)²)입니다. 채점 origin은 00/06/12/18 UTC이고, 정답은 ACE status 0 관측을 채우지 않고 씁니다. 그래서 {len(data['origins']):,} × 72 = {n_all:,} pair 중 {sc['total']['n_pairs'] / n_all:.1%}만 유효합니다.

## 제출 설명 (제출자 작성, 원문)

> {description}

{arch_line}- 학습 데이터 종료: F1 {rec.get('f1_training_data_end_utc', '—')}, F2 {rec.get('f2_training_data_end_utc', '—')}, F3 {rec.get('f3_training_data_end_utc', '—')}

## Lead 곡선

![lead curves](lead_curves.png)

리더보드 lead curves 열과 같은 값을 이 페이지에서 직접 다시 채점해 그렸습니다([lead_metrics.csv](lead_metrics.csv), lead를 6으로 나눈 나머지별 평균은 [lead_residue.csv](lead_residue.csv)).

F3 TIMSE는 lead 6의 배수(6, 12, …, 72)에서 평균 **{diagnostic['f3_timse_lead_6k']:,.1f}**이고, 그 밖의 lead에서는 {diagnostic['f3_timse_other_min']:,.1f}–{diagnostic['f3_timse_other_max']:,.1f}입니다. 채점 origin이 00/06/12/18 UTC라서, lead 6·k의 목표 시각도 00/06/12/18 UTC에만 놓입니다. F3에서 가장 빠른 관측 두 개(743.4 km/s, 2026-09-15 18:00 UTC · 690.0 km/s, 2026-07-23 18:00 UTC)도 이 시각에 있어서, lead 6·k의 정답 모집단에만 들어갑니다. 두 관측을 정답에서 빼고 다시 계산하면(점수가 아닌 진단값) F3 lead 6·k TIMSE는 **{diagnostic['f3_timse_lead_6k_without_peaks']:,.1f}**입니다.

## 월별 예측 GIF

GIF 한 장의 프레임은 채점 origin 하나입니다. 각 프레임은 origin 직전 168시간의 관측(파랑), 이후 72시간의 정답(검정), 제출 예측(주황)을 Y축 200–850 km/s로 고정해 보여 줍니다. 관측과 정답은 ACE status 0 값만 쓰고, 결측은 **채우지 않고 비워 둡니다**. 제출 예측이 없거나 유효하지 않은 lead는 채점기와 같은 persistence 값으로 그리고, 제목에 그 개수를 표시합니다. 프레임 간격은 250 ms이며, 9개월 합계 {sum(o['frames'] for o in outputs):,} 프레임으로 채점 origin 전체입니다.

| 월 | 기간 | 프레임 | GIF |
| --- | --- | ---: | --- |
{gifs}

## 재현

```bash
/home/t-lab01/sunrun/.venv/bin/python tools/submission_viz.py --member {args.member} \\
    --experiment {args.experiment} --submitted-utc {args.submitted_utc} --out {args.out}
```

CPU만 씁니다. 입력은 리더보드 dataset `{DATASET}`에 저장된 이 제출의 결과 기록·예측 parquet과 `competition-observations.parquet`뿐입니다. [leaderboard-result.json](leaderboard-result.json)은 저장 기록에서 제출 키와 그 키를 담은 URL을 뺀 사본입니다.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--submitted-utc", required=True)
    parser.add_argument("--out", required=True, help="output folder, relative to the repository root or absolute")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--rank-total", type=int)
    parser.add_argument("--board-revision", default="")
    parser.add_argument("--months", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--suffix", default="", help="smoke check: write only GIFs with this suffix")
    parser.add_argument("--record-only", action="store_true", help="rewrite leaderboard-result.json only")
    args = parser.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.record_only:
        record, _ = stored_submission(args.member, args.experiment, args.submitted_utc)
        write_record(record, out)
        return
    data = load(args.member, args.experiment, args.submitted_utc)
    footer = (f"{args.member} · {args.experiment} · submitted {args.submitted_utc} · scored 00/06/12/18 UTC origins · "
              "truth = ACE status 0, unfilled · Y 200–850 km/s")
    if args.suffix:
        render_gifs(data, out, args.experiment, footer, args.months, args.limit, args.suffix)
        print("smoke output written; other outputs untouched")
        return
    write_record(data["record"], out)
    diagnostic = lead_outputs(data, out, args.experiment)
    outputs = render_gifs(data, out, args.experiment, footer, args.months, args.limit)
    rec = data["record"]
    manifest = {
        "experiment": args.experiment, "member": args.member, "submitted_utc": rec["submitted_utc"],
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "metric_version": VERSION,
        "board": {"rank": args.rank, "ranked_rows": args.rank_total, "space_revision": args.board_revision},
        "sources": {"dataset": DATASET, "record_and_predictions": "stored result record and its prediction parquet",
                    "observations": {"file": "competition-observations.parquet",
                                     "revision": data["observation_revision"]}},
        "verification": {
            "tolerance": 1e-9, "all_equal_to_record": True,
            "scores": {s: {m: data["recomputed"][s][m] for m in METRICS + ("n_pairs", "n_origins",
                                                                           "replacement_fraction")} for s in SCOPES},
            "scored_origins": int(len(data["origins"])),
            "rendered_frames": int(sum(o["frames"] for o in outputs)),
            "complete": bool(sum(o["frames"] for o in outputs) == len(data["origins"])),
            "persistence_replaced_leads": int(data["replaced"].sum()),
        },
        "lead_diagnostic": diagnostic,
        "config": dict(width=WIDTH, height=HEIGHT, frame_ms=FRAME_MS, input_hours=INPUT_HOURS, forecast_hours=H,
                       y_limits_kms=list(Y_LIMITS), month_grouping="UTC month of origin + 1 hour",
                       frames="every scored origin (00/06/12/18 UTC) of F1/F2/F3"),
        "footer": footer, "outputs": outputs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(readme(args, data, diagnostic, outputs))
    print(f"wrote {len(outputs)} GIFs, {manifest['verification']['rendered_frames']:,} frames; "
          f"complete={manifest['verification']['complete']}")


if __name__ == "__main__":
    main()
