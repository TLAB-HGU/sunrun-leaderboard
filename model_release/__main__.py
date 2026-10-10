"""Run with python -m model_release --help."""
import argparse

import pandas as pd

from . import load_model
from .features import REQUIRED_INPUTS


def main():
    parser = argparse.ArgumentParser(description="72-hour forecasts using a frozen seongeun PKL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--origin", action="append", required=True, help="UTC hourly origin; repeat for multiple forecasts")
    for name in REQUIRED_INPUTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--donki-dir")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in (*REQUIRED_INPUTS, "donki_dir")}
    predictions = load_model(args.model).predict(pd.to_datetime(args.origin, utc=True), inputs)
    predictions.to_parquet(args.output, index=False)
    print(f"Wrote {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()
