"""Export the original local-file availability predicate to portable metadata."""
import argparse
from pathlib import Path

import pandas as pd


def export_availability(metadata_directory, image_directory, output):
    paths = sorted(Path(metadata_directory).glob("year=*/part.parquet"))
    if not paths:
        raise ValueError("No year=*/part.parquet metadata files")
    metadata = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    available = []
    sizes = []
    for value in metadata["nc_path"].fillna("").astype(str):
        path = Path(value)
        path = path if path.is_absolute() else Path(image_directory) / path
        size = path.stat().st_size if value and path.is_file() else 0
        available.append(size > 0)
        sizes.append(size)
    result = metadata[["slot", "obs_end", "s3_modified", "status"]].copy()
    result["available"] = available
    result["verified_local_bytes"] = sizes
    result.sort_values("slot").reset_index(drop=True).to_parquet(output, index=False)
    return {"rows": len(result), "available": sum(available)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-directory", required=True)
    parser.add_argument("--image-directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export_availability(args.metadata_directory, args.image_directory, args.output))
