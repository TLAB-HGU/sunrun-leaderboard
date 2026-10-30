"""Bind the already-running legacy build to its actual cache fingerprints.

No READY is published unless ACE values and both SUVI metadata prefixes exactly
reproduce the completed builder input identities. A later stage export alone is
insufficient evidence.
"""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

from paths import LB, STORE, STAGE, ROOT
from manifest import write_champion_ready, write_manifest

def main():
    sys.path.insert(0,str(LB))
    import pandas as pd
    from experiments.xgboost_suvi_fusion import features,coronal_holes
    from experiments.xgboost_suvi_fusion.run import load_ace
    from experiments.xgboost_suvi_fusion.ch_experiment import write_atomic_parquet

    with (STORE/'.refresh.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (STORE/'ch/feature-ready.json').exists():
            raise FileNotFoundError('initial champion feature build is not complete')
        expected=pd.read_parquet(STORE/'ch/ace.parquet')
        actual=load_ace([str(p) for p in sorted((STORE/'ace-snapshot').glob('year=*/part.parquet'))])
        pd.testing.assert_frame_equal(expected,actual,check_exact=True,check_dtype=False)
        metadata=pd.read_parquet(STAGE/'suvi_hourly')
        slot=pd.to_datetime(metadata.slot,utc=True)
        bindings={};max_slot=None
        def forbid_pixel_reads(_):
            raise RuntimeError('metadata prefix does not match the original cache fingerprint; rebuild from fixed inputs')
        def forbid_extraction(*_):
            raise RuntimeError('metadata prefix does not match the original cache fingerprint; rebuild from fixed inputs')
        for name,path,extractor,version in [
            ('suvi',STORE/'suvi-fusion/suvi-hourly-v2.parquet',features.extract_frame_features,None),
            ('ch',STORE/'ch/ch-hourly.parquet',coronal_holes.extract_ch_frame,coronal_holes.VERSION),
        ]:
            cache=pd.read_parquet(path)
            end=pd.to_datetime(cache.index,utc=True).max()
            prefix=metadata.loc[slot<=end].copy()
            # Existing builder returns without reading pixels only if the full
            # metadata + image file identity hash matches its saved receipt.
            checked=features.build_hourly_cache(prefix,ROOT.parent/'sun-img/data',path,
                frame_loader=forbid_pixel_reads,workers=1,extractor=forbid_extraction,extractor_version=version)
            pd.testing.assert_frame_equal(cache,checked,check_exact=True,check_dtype=False,check_names=False,check_freq=False)
            prefix_path=STORE/'suvi-meta'/f'_{name}-builder-input.parquet'
            prefix_path.parent.mkdir(parents=True,exist_ok=True)
            write_atomic_parquet(prefix,prefix_path)
            receipt=json.loads(path.with_suffix(path.suffix+'.json').read_text())
            bindings[name]={'cache':str(path),'metadata_prefix':str(prefix_path),'source_sha256':receipt['source_sha256'],
                            'last_slot':str(end),'rows':len(cache),'fingerprint_verified':True}
            max_slot=end if max_slot is None else max(max_slot,end)
        selected=metadata.loc[slot<=max_slot].copy()
        for year in sorted(pd.to_datetime(selected.slot,utc=True).dt.year.unique()):
            frame=selected.loc[pd.to_datetime(selected.slot,utc=True).dt.year==year].drop(columns=['year'],errors='ignore')
            destination=STORE/'suvi-meta'/f'year={year}/part.parquet'
            destination.parent.mkdir(parents=True,exist_ok=True)
            write_atomic_parquet(frame,destination)
        report={'ace_exact':True,'cache_bindings':bindings,'source_binding':'exact completed cache metadata/file fingerprints'}
        ROOT.joinpath('reports/initial-input-binding.json').write_text(json.dumps(report,indent=2)+'\n')
        for folder in ('ace-snapshot','suvi-meta','ch','suvi-fusion'):
            write_manifest(STORE/folder)
        ready=write_champion_ready(STORE)
        ready['initial_input_binding']=str(ROOT/'reports/initial-input-binding.json')
        (STORE/'CHAMPION_READY.json').write_text(json.dumps(ready,indent=2)+'\n')
        print(json.dumps({'ready':True,'version':ready['version'],'input_binding':report}))

if __name__=='__main__':main()
