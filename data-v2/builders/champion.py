"""Rebuild inputs from a fixed stage copy, without reading evaluator data."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import LB, STORE, STAGE

def partition_files(folder):
    files=sorted(Path(folder).glob('year=*/part.parquet'))
    if not files:
        raise FileNotFoundError(f'no parquet partitions under {folder}')
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    sys.path.insert(0, str(LB))
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from experiments.xgboost_suvi_fusion.run import load_ace, make_features
    from experiments.xgboost_suvi_fusion.ch_experiment import extract_features, write_atomic_parquet
    from manifest import write_manifest, write_champion_ready

    # A failed build must never be mistaken for one coherent input generation.
    (STORE/'CHAMPION_READY.json').unlink(missing_ok=True)

    # Stage files are replaced atomically by upstream sync, never edited in-place.
    for name, src in [('ace-snapshot', STAGE/'ace_solar'), ('suvi-meta', STAGE/'suvi_hourly')]:
        target = STORE/name
        target.mkdir(parents=True, exist_ok=True)
        for source in sorted(src.glob('year=*/part.parquet')):
            output = target/source.relative_to(src)
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix('.tmp')
            shutil.copyfile(source, temp)
            os.replace(temp, output)
        write_manifest(target, source=str(src), builder=Path(__file__))
    ch, fusion = STORE/'ch', STORE/'suvi-fusion'
    ch.mkdir(parents=True, exist_ok=True)
    fusion.mkdir(parents=True, exist_ok=True)
    ace = load_ace([str(p) for p in partition_files(STORE/'ace-snapshot')])
    write_atomic_parquet(ace, ch/'ace.parquet')
    meta_file=STORE/'suvi-meta/_metadata.parquet'
    metadata=pd.concat([pd.read_parquet(p) for p in partition_files(STORE/'suvi-meta')],ignore_index=True)
    write_atomic_parquet(metadata,meta_file)
    feature_args = SimpleNamespace(suvi_meta=str(meta_file),
        images_root=str(ROOT.parent/'sun-img/data'),
        feature_cache=str(fusion/'suvi-hourly-v2.parquet'), image_workers=args.workers,
        train_cutoff='2025-07-31T23:00:00Z')
    with threadpool_limits(limits=1):
        frame = make_features(feature_args, ace)
        write_atomic_parquet(frame, fusion/'origin-features-v2.parquet')
        write_atomic_parquet(frame[[c for c in frame if c.startswith('ace_')]], ch/'ace-features.parquet')
        extract_features(SimpleNamespace(output=str(ch), suvi_meta=str(meta_file),
            images_root=str(ROOT.parent/'sun-img/data'), image_workers=args.workers))
    for folder in (fusion, ch):
        write_manifest(folder, source='ACE stage and SUVI Fe195 (fixed stage snapshot)', builder=Path(__file__))
    write_champion_ready(STORE)
    print(json.dumps({'status':'complete','ace_latest':str(ace.timestamp_utc.max()),
                      'origin_latest':str(frame.index.max())}), flush=True)

if __name__ == '__main__':
    main()
