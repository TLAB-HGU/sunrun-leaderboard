"""F1 hourly feature reconstruction using the unchanged causal source selectors."""
import json
from pathlib import Path
import re
import pandas as pd
from experiments.e2000 import prepare

def selections(origins, metadata):
    features = json.loads((prepare.V3 / 'features_v3.json').read_text())['features']
    origin_frame = pd.DataFrame({'origin': origins, 'split': 'F1'})
    if metadata is None:
        context = prepare.relocated_builder(prepare.source_root())
    else:
        from experiments.e2000.availability_rebuild import causal_builder
        context = causal_builder(metadata)
    with context as p:
        meta, block = p.ST.load_inputs()
        strip = p.ST.select_slots(origins, meta, block)
        active = strip.loc[~strip.missing & strip.img_stale.eq(0), 'slot_idx'].astype(int)
        if any(block.index[k] != meta.iloc[k].slot for k in active.unique()):
            raise ValueError('strip cache positional source mismatch')
        ch, slots, _, _ = p.P2.filtered_table(origins, [c for c in features if c.startswith('ch_')])
        by_slot = meta[meta.status.eq('ok')].drop_duplicates('slot', keep='last').set_index('slot', drop=False)
        records = []
        for i, origin in enumerate(origins):
            for family in ['strip', 'ch']:
                if family == 'strip':
                    idx = int(strip.iloc[i].slot_idx)
                    row = meta.iloc[idx] if idx >= 0 else None
                    used = idx >= 0  # age/stale features also use the selected object
                    stale = bool(strip.iloc[i].img_stale)
                else:
                    slot = pd.Timestamp(slots[i], tz='UTC')
                    row = by_slot.loc[slot] if pd.notna(slot) and slot in by_slot.index else None
                    used = row is not None  # eligibility also determines all-NaN missingness
                    stale = False
                record = {'origin': origin.isoformat(), 'split': origin_frame.iloc[i].split,
                          'family': family, 'used': used, 'stale': stale, 'missing': row is None,
                          'finite_values_used': bool(ch.iloc[i].notna().any()) if family == 'ch' else bool(used and not stale)}
                if row is not None:
                    key = str(row.s3_key)
                    key = str(Path(key).with_suffix('.nc'))
                    local_path = p.ST.resolve_nc(str(row.nc_path))
                    satellite = re.search(r'_G(\d{2})_', Path(key).name)
                    if not satellite:
                        raise ValueError('satellite missing from key: ' + key)
                    record.update(slot=row.slot.isoformat(), obs_end=row.obs_end.isoformat(),
                                  stage_s3_modified=row.s3_modified.isoformat(), local_path=local_path,
                                  remote_key=key, url=f'https://noaa-goes{satellite.group(1)}.s3.amazonaws.com/{key}')
                records.append(record)
    mapping = pd.DataFrame(records)
    if (mapping.used & mapping.missing).any():
        raise ValueError('used feature has no selected source metadata')
    objects = []
    for url, group in mapping[mapping.used].groupby('url'):
        for name in ['local_path', 'remote_key', 'obs_end']:
            if group[name].nunique() != 1:
                raise ValueError(f'inconsistent {name}: {url}')
        first = group.iloc[0]
        objects.append({name: first[name] for name in ['local_path', 'remote_key', 'url', 'obs_end']} |
                       {'earliest_origin': group.origin.min(), 'origin_family_uses': len(group)})
    return mapping, objects



def rebuild(destination):
    import hashlib
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from experiments.e2000 import source_proof
    from experiments.e2000.availability_rebuild import provisional_metadata, causal_builder
    from experiments.e2000.inputs import atomic_json, reference, widen, sha
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    base = prepare.ROOT / 'research/e0472-multioutput-s42-20261026/relocated-v1'
    receipts = destination / 'objects'
    receipts.mkdir(exist_ok=True)
    for path in (base / 'suvi-head-causal/objects').glob('*.json'):
        if not (receipts / path.name).exists():
            shutil.copy2(path, receipts / path.name)
    origins = pd.date_range('2026-02-28T23:00Z', '2026-05-28T23:00Z', freq='h')
    for iteration in range(1, 21):
        metadata = provisional_metadata(receipts)
        meta_path = destination / f'metadata_{iteration:02d}.parquet'
        metadata.to_parquet(meta_path, index=False)
        mapping, objects = selections(origins, meta_path)
        mapping.to_parquet(destination / 'source_mapping.parquet', index=False)
        unknown = [o for o in objects if not (receipts / (hashlib.sha256(o['url'].encode()).hexdigest()+'.json')).exists()]
        print(f'F1 availability closure {iteration}: {len(objects)} selected, {len(unknown)} need HEAD', flush=True)
        if unknown:
            with ThreadPoolExecutor(max_workers=8) as pool:
                checked = list(pool.map(lambda o: source_proof.verify_object(o, receipts), unknown))
            errors = [r for r in checked if set(r.get('errors', [])) - {'publication_after_origin','observation_after_origin_minus_1h'}]
            if errors:
                atomic_json(destination/'byte_errors.json',errors)
                raise ValueError('F1 selected object byte verification failed')
            continue
        # Rehash every selected local body, recheck actual publication at this origin.
        with ThreadPoolExecutor(max_workers=8) as pool:
            checked = list(pool.map(lambda o: source_proof.verify_object(o, receipts), objects))
        if any(not r['passed'] for r in checked):
            raise ValueError('F1 source selection has unavailable object')
        break
    else:
        raise ValueError('F1 closure did not converge')
    source_identity = prepare.verify_sources(prepare.source_root())
    features = json.loads((prepare.V3/'features_v3.json').read_text())['features']
    with causal_builder(meta_path) as p:
        ace=p.A.filled_hourly(); meta,block=p.ST.load_inputs()
        ch_cols=[c for c in features if c.startswith('ch_')]
        ch,_,_,_=p.P2.filtered_table(origins,ch_cols)
        frames=[]
        for start in range(0,len(origins),256):
            oo=origins[start:start+256]
            ch_row=pd.DataFrame({c:np.repeat(ch.reindex(oo)[c].to_numpy(),72) for c in ch_cols})
            grid=p.v3_base_grid_for(oo,ace,p.HR,meta,block,features,ch_row,lag_hours=2.)
            for name in features:
                if name!='h': grid[name]=grid[name].astype('float32')
            frames.append(grid)
            print(f'F1 rebuild {min(start+256,len(origins))}/{len(origins)}',flush=True)
        dependencies={str(Path(m.__file__).resolve()):sha(m.__file__) for m in (p,p.P2,p.L9,p.HR,p.ia,p.A,p.ST,p.LE,p.L,p.L.SP,p.L.RL)}
    frame=pd.concat(frames,ignore_index=True)
    _,wide=widen(frame,features,expected_origins=origins)
    prior=pd.read_parquet(base/'train_complete_causal.parquet')
    prior=prior[pd.to_datetime(prior.origin,utc=True).isin(origins)]
    match=frame.merge(prior,on=['origin','h'],suffixes=('_new','_old'),validate='one_to_one')
    for c in features:
        if c!='h' and not np.allclose(match[c+'_new'],match[c+'_old'],equal_nan=True,rtol=1e-6,atol=1e-5):
            raise ValueError('F1 overlap changes frozen feature '+c)
    if prepare.verify_sources(prepare.source_root())!=source_identity:
        raise ValueError('raw sources changed during F1 rebuild')
    output=destination/'features_f1.parquet';frame.to_parquet(output,index=False)
    report={'passed':True,'features':reference(output),'metadata':reference(meta_path),'rows':len(frame),
            'origins':len(origins),'wide_features':wide.shape[1],'overlap_rows':len(match),
            'source_mapping':reference(destination/'source_mapping.parquet'),
            'selected_object_receipts':[reference(receipts/(hashlib.sha256(o['url'].encode()).hexdigest()+'.json')) for o in objects],
            'source_identity':source_identity,'code_hashes':dependencies,'preparation_code':reference(__file__),
            'input_policy':'Existing local raw bodies only; S3 HEAD for exact-byte publication metadata, no body downloads'}
    atomic_json(destination/'feature_receipt.json',report)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    print(json.dumps(rebuild(p.parse_args().out),indent=2))
