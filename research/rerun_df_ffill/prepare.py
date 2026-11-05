"""Create an immutable, causal feature-ffill snapshot of the raw-only input table.

Only feature columns are filled. The target and all nine metadata columns are
copied unchanged. Duplicate train/query keys must have identical input values.
Features are processed in small column groups and staged on disk, then written
in bounded row batches; the full feature matrix is never loaded into memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METADATA = ("set", "origin", "h", "target_time", "period", "fit_F1", "fit_F2", "fit_F3", "y")


def timeline(origin: np.ndarray, horizon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return representative rows, original-row -> unique-key indices, and h boundaries."""
    if len(origin) == 0:
        raise ValueError("The source table is empty")
    if np.isnat(origin).any() or pd.isna(horizon).any():
        raise ValueError("origin and h must not contain missing values")
    order = np.lexsort((origin, horizon))
    sorted_origin, sorted_h = origin[order], horizon[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (sorted_origin[1:] != sorted_origin[:-1]) | (sorted_h[1:] != sorted_h[:-1])
    representatives = order[first]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.cumsum(first, dtype=np.int64) - 1
    unique_h = horizon[representatives]
    boundaries = np.r_[0, np.flatnonzero(unique_h[1:] != unique_h[:-1]) + 1, len(unique_h)]
    return representatives, inverse, boundaries


def fill_feature(values: np.ndarray, representatives: np.ndarray, inverse: np.ndarray,
                 boundaries: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
    """Validate repeated inputs, then forward-fill independently along each h timeline."""
    unique = values[representatives]
    repeated = unique[inverse]
    equal = (values == repeated) | (np.isnan(values) & np.isnan(repeated))
    if not equal.all():
        bad = int(np.flatnonzero(~equal)[0])
        raise ValueError(f"Conflicting duplicate (origin, h) values in {name!r}, source row {bad}")
    del repeated, equal
    before = int(np.isnan(values).sum())
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment = unique[start:stop]
        valid = ~np.isnan(segment)
        previous = np.maximum.accumulate(np.where(valid, np.arange(len(segment)), -1))
        take = previous >= 0
        segment[take] = segment[previous[take]]
    filled = unique[inverse]
    after = int(np.isnan(filled).sum())
    return filled, {"missing_before": before, "missing_after": after,
                    "filled": before - after, "all_missing": before == len(values)}


def _source_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def _metadata_equal(left: pa.Table, right: pa.Table) -> bool:
    if not left.schema.equals(right.schema, check_metadata=True) or left.num_rows != right.num_rows:
        return False
    for name in left.column_names:
        if left[name].equals(right[name]):
            continue
        if pa.types.is_floating(left[name].type):
            if left[name].null_count != right[name].null_count:
                return False
            if not left[name].is_null().equals(right[name].is_null()):
                return False
            if np.array_equal(left[name].to_numpy(), right[name].to_numpy(), equal_nan=True):
                continue
        return False
    return True


def prepare(source: Path, out: Path, report_path: Path, *, features: list[str] | None = None,
            column_chunk: int = 8, row_batch: int = 65536) -> dict:
    """Write a new snapshot and provenance report; refuse to replace existing artifacts."""
    source, out, report_path = map(Path, (source, out, report_path))
    if column_chunk < 1 or row_batch < 1:
        raise ValueError("column_chunk and row_batch must be positive")
    if out.resolve() == source.resolve() or report_path.resolve() in (source.resolve(), out.resolve()):
        raise ValueError("Source, output and report must be different paths")
    if out.exists() or report_path.exists():
        raise FileExistsError("Immutable output or report already exists; use new paths")
    started = time.perf_counter()
    identity = _source_identity(source)
    reader = pq.ParquetFile(source)
    schema = reader.schema_arrow
    absent = set(METADATA).difference(schema.names)
    if absent:
        raise ValueError(f"Missing metadata columns: {sorted(absent)}")
    all_features = [name for name in schema.names if name not in METADATA]
    selected = all_features if features is None else list(features)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one feature, without duplicate names")
    unknown = set(selected).difference(all_features)
    if unknown:
        raise ValueError(f"Unknown or protected feature columns: {sorted(unknown)}")
    for name in selected:
        if not pa.types.is_floating(schema.field(name).type):
            raise ValueError(f"Expected floating-point input feature: {name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staging_bytes = reader.metadata.num_rows * sum(schema.field(n).type.bit_width // 8 for n in selected)
    required = 2 * staging_bytes + identity["size_bytes"] + 64 * 1024**2
    if shutil.disk_usage(out.parent).free < required:
        raise OSError(f"Insufficient disk space: need at least {required} bytes for staging and output")
    keys = reader.read(columns=["origin", "h"], use_threads=False)
    origin = keys["origin"].cast(pa.int64()).to_numpy().view("datetime64[ns]")
    if keys["origin"].null_count or keys["h"].null_count:
        raise ValueError("origin and h must not contain missing values")
    representatives, inverse, boundaries = timeline(origin, keys["h"].to_numpy())
    stats = {}
    metadata_source = reader.read(columns=list(METADATA), use_threads=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest_path = source.parent / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with tempfile.TemporaryDirectory(prefix=".ffill-", dir=out.parent) as temporary:
        scratch = Path(temporary)
        staged = {}
        for start in range(0, len(selected), column_chunk):
            chunk = selected[start:start + column_chunk]
            table = reader.read(columns=chunk, use_threads=False)
            for name in chunk:
                values = table[name].to_numpy()
                filled, stats[name] = fill_feature(values, representatives, inverse, boundaries, name)
                path = scratch / f"feature-{start + chunk.index(name):04d}.npy"
                np.save(path, filled, allow_pickle=False)
                staged[name] = path
                del values, filled
            del table
            pa.default_memory_pool().release_unused()
            print(json.dumps({"phase": "ffill", "features_done": min(start + column_chunk, len(selected)),
                              "features_total": len(selected)}), flush=True)
        untouched = [name for name in schema.names if name not in staged]
        temporary_out = scratch / "snapshot.parquet"
        offset = 0
        with pq.ParquetWriter(temporary_out, schema, compression="zstd") as writer:
            for batch in reader.iter_batches(batch_size=row_batch, columns=untouched, use_threads=False):
                columns = {}
                for name in schema.names:
                    if name in staged:
                        memory = np.load(staged[name], mmap_mode="r", allow_pickle=False)
                        values = np.array(memory[offset:offset + batch.num_rows], copy=True)
                        del memory
                        columns[name] = pa.array(values, type=schema.field(name).type, from_pandas=True)
                    else:
                        columns[name] = batch.column(batch.schema.get_field_index(name))
                table = pa.Table.from_arrays([columns[n] for n in schema.names], schema=schema)
                writer.write_table(table, row_group_size=row_batch)
                offset += batch.num_rows
                del columns, table, batch
        if offset != reader.metadata.num_rows:
            raise RuntimeError("Output row count changed")
        output_reader = pq.ParquetFile(temporary_out)
        if not output_reader.schema_arrow.equals(schema, check_metadata=True):
            raise RuntimeError("Output schema changed")
        if not _metadata_equal(metadata_source, output_reader.read(columns=list(METADATA), use_threads=False)):
            raise RuntimeError("Protected metadata or targets changed")
        if _source_identity(source) != identity:
            raise RuntimeError("Source file changed during preparation")
        report = {
            "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {**identity, "manifest_built_utc": manifest.get("built_utc"),
                       "manifest_path": str(manifest_path.resolve()) if manifest else None},
            "output": str(out.resolve()), "rows": offset, "columns": len(schema),
            "unique_origin_h_keys": len(representatives),
            "duplicate_rows_validated": offset - len(representatives),
            "feature_names": all_features, "filled_feature_names": selected,
            "protected_columns": list(METADATA), "missing_by_feature": stats,
            "policy": "ffill by h over sorted origin, train/query union; identical duplicate inputs required; "
                      "leading NaNs retained; targets and metadata unchanged",
            "validation": {"metadata_unchanged": True, "schema_unchanged": True, "source_unchanged": True},
            "git_commit": commit,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "pyarrow": pa.__version__},
            "column_chunk": column_chunk, "row_batch": row_batch,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        temporary_report = scratch / "report.json"
        temporary_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        # Hard links publish completed files without ever replacing an existing snapshot.
        os.link(temporary_out, out)
        try:
            if report_path.parent.stat().st_dev == scratch.stat().st_dev:
                os.link(temporary_report, report_path)
            else:
                with report_path.open("x") as destination:
                    destination.write(temporary_report.read_text())
        except Exception:
            out.unlink()
            raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data-v2/input/df.parquet")
    parser.add_argument("--out", type=Path, default=HERE / "input/df_ffill.parquet")
    parser.add_argument("--report", type=Path, default=HERE / "input/ffill_report.json")
    parser.add_argument("--features", nargs="+", help="Fill only these features; preserve all other columns")
    parser.add_argument("--column-chunk", type=int, default=8)
    parser.add_argument("--row-batch", type=int, default=65536)
    args = parser.parse_args()
    result = prepare(args.source, args.out, args.report, features=args.features,
                     column_chunk=args.column_chunk, row_batch=args.row_batch)
    print(json.dumps({"status": result["status"], "rows": result["rows"], "output": result["output"],
                      "report": str(args.report), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
