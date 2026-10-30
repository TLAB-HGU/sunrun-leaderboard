"""Compute the fixed GONG-PFSS OCB table before the three XGBoost seeds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, choices=(3,), required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    from .gong_ocb import OCB_CONFIG, precompute_origin_table

    receipt = precompute_origin_table(n_jobs=args.n_jobs)
    if "EXPOS_OUT" in os.environ:
        out = Path(os.environ["EXPOS_OUT"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "gong_ocb_precompute.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"event": "ocb_precompute_done", "receipt": receipt}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
