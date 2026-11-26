# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.23,<1", "altair>=5,<7", "pandas>=2.3,<3", "numpy>=2,<3", "pyarrow>=18", "huggingface-hub>=0.25"]
# ///
"""Open from the repo root with uvx marimo edit --sandbox tools/solar_wind_features.py."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "research"))
    sys.path.insert(0, str(root / "leaderboard" / "scorer"))
    from _module.table import load_grid, load_table
    from competition import load_observations, prepare_context

    experiment = "e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel"
    logs = root / "research" / "result" / experiment / "logs"
    periods = ("F1", "F2", "F3")
    added_features = {
        "ace_log_proton_entropy",
        "ace_nb_cocompression_6h",
        "swpc_hss_coverage49_72",
        "ace_proton_ram_pressure_npa",
        "srs_earthward_eruptive_area",
    }
    return (
        added_features, alt, experiment, hf_hub_download, json, load_grid,
        load_observations, load_table, logs, mo, np, Path, pd, periods, prepare_context,
    )


@app.cell
def _(experiment, mo):
    mo.md(
        f"""
        # 리더보드 1위 피처와 태양풍 관측 속력

        [리더보드](https://huggingface.co/spaces/tlabtlab/sunrun-leaderboard)
        · 2026-11-23 확인 · Total S **92.11**

        `{experiment}`
        """
    )
    return


@app.cell
def _(Path, hf_hub_download, json, load_observations, logs, mo, prepare_context):
    try:
        metadata = json.loads((logs / "F1_s42_metadata.json").read_text())
        receipt = json.loads((logs / "submission-receipt.json").read_text())
        table_path = Path(metadata["input_file"]["path"])
        stat = table_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            metadata["input_file"]["bytes"], metadata["input_file"]["mtime_ns"]
        ):
            raise ValueError(f"학습 입력 파일이 실행 기록과 다릅니다. {table_path}")

        store = {
            "repo_id": "tlabtlab/sunrun-lb-store",
            "filename": "competition-observations.parquet",
            "repo_type": "dataset",
            "revision": receipt["competition_dataset_revision"],
        }
        try:
            observations_path = hf_hub_download(**store, local_files_only=True)
        except FileNotFoundError:
            observations_path = hf_hub_download(**store)
        context = prepare_context(load_observations(observations_path))
    except (OSError, ValueError, KeyError) as error:
        mo.stop(True, mo.callout(f"입력 데이터를 열 수 없습니다. {error}", kind="danger"))
    return context, metadata, table_path


@app.cell
def _(mo):
    period_picker = mo.ui.dropdown(
        {"전체": "all", "F1": "F1", "F2": "F2", "F3": "F3"}, value="전체", label="평가 기간"
    )
    block_picker = mo.ui.dropdown(
        {"1–24h": 0, "25–48h": 1, "49–72h": 2}, value="1–24h", label="리드 블록"
    )
    mo.hstack([period_picker, block_picker], justify="start", gap=2, wrap=True)
    return block_picker, period_picker


@app.cell
def _(block_picker, metadata, mo):
    end = 24 * (block_picker.value + 1)
    feature_names = metadata["expanded_features_by_block"][str(block_picker.value)]
    lead_picker = mo.ui.slider(
        start=end - 23, stop=end, value=end, label="관측 리드 (h)", show_value=True
    )
    feature_picker = mo.ui.dropdown(
        feature_names, value="ace_speed_kms_last", label="입력 피처 (X)", searchable=True
    )
    mo.vstack([lead_picker, feature_picker])
    return feature_picker, lead_picker


@app.cell
def _(added_features, context, load_grid, load_table, metadata, np, pd, periods):
    def feature_values(table_path, name):
        """Return the model design column at all official origins."""
        if name in metadata["constant_features"]:
            feature, feature_lead = name, None
        else:
            feature, suffix = name.rsplit("_h", 1)
            feature_lead = int(suffix)

        constant, varying = ([feature], []) if feature_lead is None else ([], [feature])
        table = load_table(table_path, [feature], lambda *args, **kwargs: None)
        pieces = []
        for period in periods:
            grid, _ = load_grid(table, period, False, constant, varying)
            index = grid["origins"].get_indexer(context["origins"][context["period"] == period])
            if (index < 0).any():
                raise ValueError(f"{period}: 공식 평가 격자의 발행 시각이 학습 입력에 없습니다.")
            values = grid["C"][index, 0] if feature_lead is None else grid["E"][index, feature_lead - 1, 0]
            pieces.append(values)
        column = np.concatenate(pieces)
        if feature in added_features:
            column[column == -999.0] = np.nan
        return column

    def scatter_points(column, period, lead):
        origins = context["origins"]
        frame = pd.DataFrame(
            {
                "period": context["period"],
                "origin": origins,
                "target_time": origins + pd.Timedelta(hours=lead),
                "feature": column,
                "speed_kms": context["truth"][:, lead - 1],
            }
        )
        if period != "all":
            frame = frame.loc[frame["period"].eq(period)]
        valid = np.isfinite(frame["feature"]) & np.isfinite(frame["speed_kms"])
        return frame.loc[valid].copy(), int((~valid).sum())

    return feature_values, scatter_points


@app.cell
def _(feature_picker, feature_values, mo, table_path):
    try:
        column = feature_values(table_path, feature_picker.value)
    except (OSError, ValueError, KeyError) as error:
        mo.stop(True, mo.callout(f"피처를 읽을 수 없습니다. {error}", kind="danger"))
    return (column,)


@app.cell
def _(alt, column, feature_picker, lead_picker, mo, period_picker, scatter_points):
    points, excluded = scatter_points(column, period_picker.value, lead_picker.value)
    counts = mo.md(f"유효한 쌍 **{len(points):,}개** · 결측으로 제외 **{excluded:,}개**")
    mo.stop(
        points.empty,
        mo.vstack([counts, mo.callout("이 조건에서 함께 관측된 피처와 속력 값이 없습니다.", kind="warn")]),
    )

    plot_data = points.assign(
        origin=points["origin"].dt.strftime("%Y-%m-%d %H:%M UTC"),
        target_time=points["target_time"].dt.strftime("%Y-%m-%d %H:%M UTC"),
    )
    chart = (
        alt.Chart(plot_data)
        .mark_circle(size=30, opacity=0.6)
        .encode(
            x=alt.X("feature:Q", title=feature_picker.value, scale=alt.Scale(zero=False)),
            y=alt.Y("speed_kms:Q", title="ACE 관측 태양풍 속력 (km/s)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "period:N", title="평가 기간",
                scale=alt.Scale(domain=["F1", "F2", "F3"], range=["#2672a5", "#bc5b24", "#397f58"]),
                legend=None if period_picker.value != "all" else alt.Legend(),
            ),
            tooltip=[
                alt.Tooltip("period:N", title="평가 기간"),
                alt.Tooltip("origin:N", title="발행 시각"),
                alt.Tooltip("target_time:N", title="관측 시각"),
                alt.Tooltip("feature:Q", title=feature_picker.value, format=".5g"),
                alt.Tooltip("speed_kms:Q", title="관측 속력 (km/s)", format=".1f"),
            ],
        )
        .properties(width="container", height=460)
        .interactive()
    )
    mo.vstack([counts, mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)])
    return


if __name__ == "__main__":
    app.run()
