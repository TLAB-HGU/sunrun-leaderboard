"""Render the nine monthly forecast GIFs (F1 2026-01..03, F2 04..06, F3 07..09) of the submitted e2000fs_calib_point4_144h forecast.

The frames show the forecast that is on the leaderboard: the three merged `*_predictions.npz` of
`research/e2000fs_f123-20261031/runs/point4_144h_v1`, the 2026-10-31 refit of the point4 144 h control arm (one multi-output
booster per 24 h block and stage). `vizlib.load` refuses to draw when the prediction/truth join, the recomputed F1/F2/F3
scores or the monthly frame counts disagree with the leaderboard record (`leaderboard-result.json`).

    /home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib_point4_144h/code/render_gifs.py

Use --months 2026-01 --limit 24 --suffix _smoke for a quick check; all nine months take a few minutes.
"""
import argparse
import json
from pathlib import Path

import vizlib as V

OUT = Path(__file__).resolve().parents[1]
RUN = V.ROOT / "research/e2000fs_f123-20261031/runs/point4_144h_v1"
FILES = [RUN / f"{period}_predictions.npz" for period in ("F1", "F2", "F3")]
KEYS = [("origins", "pred")] * 3
EXPERIMENT = "e2000fs_calib_point4_144h"
ARM = "point4 144 h control arm: reg:squarederror with 4x weight on targets >= 550 km/s, training mask origin + 144 h <= cutoff"
AGGREGATION = "single model per period; one multi_output_tree booster per 24 h block (3 per stage), 900 rounds, seed 42"
FOOTER = ("e2000fs_calib_point4_144h · point4 (4x high-speed weight), 144 h maturity mask, 24 h multi-output blocks · "
          "F1 01–03 / F2 04–06 / F3 07–09 2026 · Y axis fixed at 200–850 km/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", nargs="*", help="subset of YYYY-MM (default: all nine)")
    parser.add_argument("--limit", type=int, help="render at most this many frames per month (smoke check)")
    parser.add_argument("--suffix", default="", help="suffix for output names (smoke check)")
    parser.add_argument("--previews", action="store_true",
                        help="render the compact preview_*.gif files that the README embeds")
    parser.add_argument("--manifest-only", action="store_true",
                        help="rewrite manifest.json (scores, diagnostics, hashes) for the GIFs already on disk")
    args = parser.parse_args()

    if args.manifest_only:
        previous = json.loads((OUT / "manifest.json").read_text())
        data = V.load(OUT, FILES, KEYS)
        manifest = V.gif_manifest(OUT, data, EXPERIMENT, ARM, AGGREGATION, FILES, KEYS, FOOTER, previous["outputs"])
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"manifest rewritten: {len(previous['outputs'])} GIFs, {manifest['verification']['rendered_frames']:,} frames")
        return

    if args.previews:
        data = V.load(OUT, FILES, KEYS)
        previous = json.loads((OUT / "manifest.json").read_text())
        previews = V.render_gifs(OUT, data, EXPERIMENT, FOOTER, prefix="preview_", stride=4, dpi=72, frame_ms=400)
        manifest = V.gif_manifest(OUT, data, EXPERIMENT, ARM, AGGREGATION, FILES, KEYS, FOOTER, previous["outputs"])
        manifest["previews"] = previews
        manifest["previews_config"] = dict(prefix="preview_", stride=4, dpi=72, frame_ms=400,
                                           note="every 4th fold, 72 dpi, 400 ms per frame: the same drawing as the "
                                                "full GIFs, sized for playback inside the README")
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{len(previews)} previews, {sum(p['bytes'] for p in previews) / 1e6:.1f} MB")
        return

    data = V.load(OUT, FILES, KEYS)
    outputs = V.render_gifs(OUT, data, EXPERIMENT, FOOTER, months=args.months, limit=args.limit, suffix=args.suffix)
    print("scores", json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in data["scores"].items()}))
    if args.suffix:
        print("smoke output written; manifest untouched")
        return
    manifest = V.gif_manifest(OUT, data, EXPERIMENT, ARM, AGGREGATION, FILES, KEYS, FOOTER, outputs)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(outputs)} GIFs, {manifest['verification']['rendered_frames']:,} frames")


if __name__ == "__main__":
    main()
