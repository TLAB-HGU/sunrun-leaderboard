"""Apply the user's training-label ffill policy without changing query truth.

Labels are filled on their physical target_time axis, never by flattened table
row order. Input features have already been causally filled by prepare.py.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent


def fill_training_labels(metadata):
    """Return y with training rows filled from the last observed target timestamp."""
    time_axis = pd.to_datetime(metadata.target_time, utc=True)
    if time_axis.isna().any():
        raise ValueError("target_time must be present")
    timeline = pd.DataFrame({"time": time_axis, "y": metadata.y.to_numpy()})
    if timeline.groupby("time").y.nunique(dropna=False).gt(1).any():
        raise ValueError("The same target timestamp has inconsistent source labels")
    timeline = timeline.drop_duplicates("time").sort_values("time").set_index("time")
    filled = timeline.y.ffill()
    labels = metadata.y.to_numpy(copy=True)
    training = metadata["set"].eq("train").to_numpy()
    labels[training] = filled.reindex(pd.DatetimeIndex(time_axis[training])).to_numpy()
    original = metadata.y.to_numpy()
    if not np.array_equal(labels[~training], original[~training], equal_nan=True):
        raise RuntimeError("Query truth was modified")
    if not np.array_equal(labels[np.isfinite(original)], original[np.isfinite(original)], equal_nan=True):
        raise RuntimeError("Observed labels were modified")
    return labels


def prepare(source, out, source_report, report_path):
    source, out, source_report, report_path = map(Path, (source, out, source_report, report_path))
    if out.exists() or report_path.exists():
        raise FileExistsError("New input artifacts are immutable; choose unused paths")
    started = time.perf_counter()
    reader = pq.ParquetFile(source)
    metadata = reader.read(columns=["set", "target_time", "y"], use_threads=False).to_pandas()
    labels = fill_training_labels(metadata)
    training = metadata["set"].eq("train").to_numpy()
    offset = 0
    temporary = out.with_suffix(".partial.parquet")
    with pq.ParquetWriter(temporary, reader.schema_arrow, compression="zstd") as writer:
        for batch in reader.iter_batches(batch_size=32768, use_threads=False):
            table = pa.Table.from_batches([batch])
            values = labels[offset:offset + batch.num_rows]
            column = pa.array(values, type=reader.schema_arrow.field("y").type, from_pandas=True)
            table = table.set_column(table.schema.get_field_index("y"), table.schema.field("y"), column)
            writer.write_table(table, row_group_size=32768)
            offset += batch.num_rows
    if offset != reader.metadata.num_rows:
        raise RuntimeError("Row count changed")
    original_meta = reader.read(columns=[n for n in reader.schema_arrow.names if n in
                                        ("set", "origin", "h", "target_time", "period", "fit_F1", "fit_F2", "fit_F3")])
    output_reader = pq.ParquetFile(temporary)
    if not original_meta.equals(output_reader.read(columns=original_meta.column_names)):
        raise RuntimeError("Protected keys, periods or training masks changed")
    actual_y = output_reader.read(columns=["y"], use_threads=False)["y"].to_numpy()
    if not np.array_equal(labels, actual_y, equal_nan=True):
        raise RuntimeError("Written labels differ from the verified fill")
    report = json.loads(source_report.read_text())
    report.update(output=str(out.resolve()),
                  policy=report["policy"] + "; training y ffill on sorted target_time; query y unchanged",
                  training_target_ffill={"confirmed_by_user": True, "axis": "target_time",
                                         "training_missing_before": int(metadata.y[training].isna().sum()),
                                         "training_missing_after": int(np.isnan(labels[training]).sum()),
                                         "query_labels_unchanged": True, "observed_labels_unchanged": True,
                                         "official_evaluation_truth": "unchanged, stored independently by leaderboard"},
                  feature_snapshot=str(source.resolve()), label_stage_seconds=time.perf_counter() - started,
                  label_stage_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report["validation"]["metadata_unchanged"] = "all non-target metadata unchanged; only missing training y filled"
    temporary.rename(out)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(out), "rows": offset, **report["training_target_ffill"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "input/df_ffill.parquet")
    parser.add_argument("--out", type=Path, default=HERE / "input/df_ffill_trainy.parquet")
    parser.add_argument("--source-report", type=Path, default=HERE / "input/ffill_report.json")
    parser.add_argument("--report", type=Path, default=HERE / "input/ffill_trainy_report.json")
    args = parser.parse_args()
    prepare(args.source, args.out, args.source_report, args.report)
