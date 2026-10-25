"""Canonical CPU direct model. CLI performs preofficial prediction only.

run_stage(..., stage='official') is an internal launch integration: the launcher
must validate its frozen final-confirmation gate before calling it.
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

import gate
from poison import validate_poison
from contracts import (ContractModel, TargetSpec, checkpoint_identity, load_checkpoint,
                       save_checkpoint, validate_predictions)
from methods import STABLE_KEYS, UPDATES

RECIPE_KEYS = set('model head_count max_depth n_estimators learning_rate target '
                  'long_head_training_scope tail_hours subsample colsample_bytree '
                  'min_child_weight reg_lambda feature_block'.split()) | set(STABLE_KEYS)
BANDS = ((1,6),(7,24),(25,72))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def code_hashes():
    names=('runner.py','contracts.py','gate.py','launch.py','evidence.py','build_inputs.py','poison.py')
    return {name: file_hash(Path(__file__).with_name(name))
            if Path(__file__).with_name(name).is_file() else 'missing' for name in names}


def validate_recipe(recipe):
    gate.require(isinstance(recipe,dict) and RECIPE_KEYS<=set(recipe), 'recipe fields differ from canonical schema')
    for key in set(recipe)-RECIPE_KEYS:
        gate.require(key.startswith('input_') and len(key)>6 and type(recipe[key]) is bool,'only boolean input_<block> flags may extend recipe')
    gate.require(recipe['model'] in ('xgboost','stable_uwr') and recipe['feature_block']=='base111', 'unsupported model or feature block')
    if recipe['model']=='stable_uwr':
        gate.require(set(STABLE_KEYS)<=set(recipe),'stable recipe incomplete')
        gate.require(recipe['stable_update'] in UPDATES,'invalid stable_update')
        gate.require(type(recipe['stable_blocks']) is int and recipe['stable_blocks']>=2,'invalid stable_blocks')
        gate.require(type(recipe['stable_var_depth']) is int and recipe['stable_var_depth']>=1,'invalid stable_var_depth')
        gate.require(type(recipe['stable_var_min_leaf']) is int and recipe['stable_var_min_leaf']>=1,'invalid stable_var_min_leaf')
        for key in ('stable_alpha','stable_beta','stable_eps_rel','stable_gamma_cap'):
            gate.require(type(recipe[key]) in (int,float) and np.isfinite(recipe[key]) and recipe[key]>=0,'invalid '+key)
        if recipe['stable_update']=='uwr':
            gate.require(recipe['stable_alpha']==0 and recipe['stable_beta']>0,'uwr requires alpha=0, beta>0')
        elif recipe['stable_update']=='constant':
            gate.require(recipe['stable_alpha']>0 and recipe['stable_beta']==0,'constant requires alpha>0, beta=0')
        elif recipe['stable_update']=='combined':
            gate.require(recipe['stable_alpha']>0 and recipe['stable_beta']>0,'combined requires alpha>0, beta>0')
    gate.require(recipe['head_count'] in (1,3,4) and type(recipe['head_count']) is int,'head_count must be 1, 3 or 4')
    gate.require(recipe['target'] in ('absolute','delta'), 'target must be absolute or delta for all heads')
    gate.require(recipe['long_head_training_scope'] in ('all','band'), 'invalid long-head scope')
    for key in ('max_depth','n_estimators'):
        gate.require(type(recipe[key]) is int and recipe[key]>0, 'invalid '+key)
    gate.require(type(recipe['tail_hours']) is int and recipe['tail_hours']>=0,'invalid tail_hours')
    for key in ('learning_rate','subsample','colsample_bytree'):
        gate.require(type(recipe[key]) in (int,float) and np.isfinite(recipe[key]) and 0<recipe[key]<=1,'invalid '+key)
    for key in ('min_child_weight','reg_lambda'):
        gate.require(type(recipe[key]) in (int,float) and np.isfinite(recipe[key]) and recipe[key]>=0,'invalid '+key)


def select_features(base,blocks,recipe,train,grids):
    """Select direct input union from prebuilt, key-aligned artifact columns."""
    gate.require(isinstance(blocks,dict),'blocks must map names to ordered columns')
    selected=list(base)
    for key in sorted(k for k in recipe if k.startswith('input_')):
        name=key[6:]
        gate.require(name in blocks,'input flag has no declared block: '+name)
        if not recipe[key]:
            continue
        columns=blocks[name]
        gate.require(isinstance(columns,list) and columns and all(isinstance(c,str) for c in columns),'invalid input block')
        gate.require(not set(columns)&{'origin','h','target_t','y','observed','obs','label'},'target/key columns cannot be input blocks')
        gate.require(len(set(columns))==len(columns) and not set(columns)&set(selected),'overlapping block features')
        for frame in [train,*grids]:
            gate.require(set(columns)<=set(frame.columns),'missing enabled block columns')
            gate.require(not np.isinf(frame[columns].to_numpy(float)).any(),'infinite block features')
        for column in columns:
            values=train[column].to_numpy(float)
            finite=values[np.isfinite(values)]
            gate.require(len(np.unique(finite))>1,'no finite nonconstant block signal: '+column)
            gate.require(not any(np.array_equal(values,train[old].to_numpy(float),equal_nan=True) for old in selected),'duplicate/no-op block signal: '+column)
            selected.append(column)
    return selected


def cpu_budget(n_jobs):
    budget=int(os.environ.get('RESEARCH_THREADS','4'))
    gate.require(type(n_jobs) is int and 0<n_jobs<=budget,'n_jobs exceeds CPU research budget')


def _utc(values,name):
    dates=pd.DatetimeIndex(values)
    gate.require(str(dates.tz)=='UTC' and not dates.hasnans, name+' requires UTC timestamps')
    return dates


def validate_training(frame, features):
    required=set(features)|{'origin','h','target_t','y','v_lag0'}
    gate.require(not frame.empty and required<=set(frame.columns) and not frame.columns.duplicated().any(),'invalid training columns')
    _utc(frame.origin,'origin'); _utc(frame.target_t,'target_t')
    gate.require(not frame.duplicated(['origin','h']).any(),'duplicate training keys')
    gate.require(frame.h.isin(range(1,73)).all(),'invalid training horizons')
    expected=frame.origin+pd.to_timedelta(frame.h,unit='h')
    gate.require(np.array_equal(expected.to_numpy(),frame.target_t.to_numpy()),'target_t does not equal origin+h')
    gate.require(np.isfinite(frame.y.to_numpy(float)).all(),'nonfinite training target')
    gate.require(not np.isinf(frame[list(features)].to_numpy(float)).any(),'infinite training feature')


def mask_hash(train):
    """Bind exact training row keys, target timestamps and label availability."""
    frame=train[['origin','h','target_t']].copy()
    frame['observed']=np.isfinite(train.y.to_numpy(float))
    return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()


def _grid_frame(test):
    gate.require({'origin','h'}<=set(test.columns),'missing prediction keys')
    return pd.DataFrame({'origin':test.origin,'horizon':test.h,'prediction':np.zeros(len(test))})


def fit_heads(train,test,feature_names,recipe,seed,n_jobs=1,checkpoints=None,*,cutoff,context,estimator_factory=None):
    """Fit each head once (or reload exact cache) and return a complete hourly grid."""
    validate_recipe(recipe); cpu_budget(n_jobs)
    gate.require(type(seed) is int,'integer seed required')
    gate.require(set(context)==set('stage manifest_sha256 code_hashes data_hashes grid mask_sha256 source_availability'.split()),'incomplete fit context')
    validate_training(train,feature_names)
    cutoff=pd.Timestamp(cutoff)
    gate.require(cutoff.tzinfo is not None,'cutoff must be timezone-aware')
    gate.require((train.target_t<cutoff).all(),'training target_t must be strictly before cutoff')
    origins=_utc(test.origin.unique(),'prediction origin').sort_values()
    expected=pd.date_range(origins.min(),origins.max(),freq='h')
    validate_predictions(_grid_frame(test),expected)
    gate.require(set(feature_names)<=set(test.columns),'missing prediction features')
    target=TargetSpec(recipe['target'])
    bands={1:((1,72),),3:BANDS,4:((1,6),(7,24),(25,48),(49,72))}[recipe['head_count']]
    output=np.full(len(test),np.nan)
    if estimator_factory is None:
        if recipe['model']=='stable_uwr':
            from methods import StableUWRBoost
            estimator_factory=StableUWRBoost
        else:
            from xgboost import XGBRegressor
            estimator_factory=XGBRegressor
    for lo,hi in bands:
        tail=recipe['tail_hours']
        pool=train if (hi==72 and recipe['long_head_training_scope']=='all') else train[train.h.between(max(1,lo-tail),min(72,hi+tail))]
        gate.require(not pool.empty,'empty head training pool')
        if recipe['model']=='stable_uwr':
            params=dict(seed=seed,n_jobs=n_jobs,
                        n_estimators=recipe['n_estimators'],
                        learning_rate=recipe['learning_rate'],
                        max_depth=recipe['max_depth'],
                        min_child_weight=recipe['min_child_weight'],
                        reg_lambda=recipe['reg_lambda'],
                        subsample=recipe['subsample'],
                        colsample_bytree=recipe['colsample_bytree'],
                        update=recipe['stable_update'],
                        alpha=recipe['stable_alpha'],
                        beta=recipe['stable_beta'],
                        blocks=recipe['stable_blocks'],
                        eps_rel=recipe['stable_eps_rel'],
                        gamma_cap=recipe['stable_gamma_cap'],
                        var_depth=recipe['stable_var_depth'],
                        var_min_leaf=recipe['stable_var_min_leaf'])
        else:
            params={k:recipe[k] for k in ('max_depth','n_estimators','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda')}
            params.update(objective='reg:squarederror',tree_method='hist',device='cpu',random_state=seed,n_jobs=n_jobs)
        hyperparameters=dict(recipe=recipe,estimator=params,stage=context['stage'],head_band=[lo,hi],
                             manifest_sha256=context['manifest_sha256'],mask_sha256=context['mask_sha256'],
                             training_rows_hash=mask_hash(pool))
        identity=checkpoint_identity(code_hashes=context['code_hashes'],data_hashes=context['data_hashes'],target=target,
            seed=seed,hyperparameters=hyperparameters,feature_names=feature_names,cutoff=cutoff.isoformat(),
            grid=context['grid'],source_availability=context['source_availability'])
        path=Path(checkpoints)/f'{identity}.model' if checkpoints is not None else None
        if path is not None and path.exists():
            model=load_checkpoint(path,identity)
            gate.require(model.target==target and model.feature_names==tuple(feature_names),'cached model receipt mismatch')
        else:
            model=ContractModel(estimator_factory(**params),target,feature_names)
            groups=pool.origin.to_numpy() if recipe['model']=='stable_uwr' else None
            model.fit(pool[list(feature_names)],pool.y.to_numpy(),current_speed=pool.v_lag0.to_numpy(),
                      groups=groups)
            if path is not None:
                save_checkpoint(path,model,identity)
                model=load_checkpoint(path,identity)
            if checkpoints is not None and hasattr(model.estimator,'receipt'):
                receipt_dir=Path(checkpoints).parent/'method-receipts'
                receipt_dir.mkdir(parents=True,exist_ok=True)
                gate.atomic(receipt_dir/f"{context['stage']}_band{lo}_{hi}_s{seed}.json",
                            dict(band=[lo,hi],seed=seed,cutoff=cutoff.isoformat(),identity=identity,
                                 stage=context['stage'],**model.estimator.receipt()))
        selected=test.h.between(lo,hi).to_numpy()
        output[selected]=model.predict(test.loc[selected,list(feature_names)],current_speed=test.loc[selected,'v_lag0'].to_numpy(),target=target)
    result=test[['origin','h']].copy();result['prediction']=output
    validate_predictions(result.rename(columns={'h':'horizon'}),expected)
    return result.reset_index(drop=True)


def _read_artifact(ref):
    gate.sha(ref['sha256'])
    gate.require(file_hash(ref['path'])==ref['sha256'],'input artifact hash mismatch')
    frame=pd.read_parquet(ref['path'])
    gate.require(file_hash(ref['path'])==ref['sha256'],'input changed while reading')
    return frame


def atomic_predictions(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.parquet',delete=False) as stream:
        temporary=Path(stream.name)
    try:
        frame.to_parquet(temporary,index=False)
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def run_stage(manifest,inputs,out,stage='preofficial',n_jobs=1):
    """Launch integration. Preofficial CLI never enables official predictions."""
    manifest_id=gate.manifest(manifest)
    gate.require(gate.digest(inputs)==manifest['contract']['inputs_sha256'],
                 'frozen input artifacts/features/block mapping changed')
    validate_recipe(manifest['recipe']);cpu_budget(n_jobs)
    validate_poison(inputs,manifest['recipe'])
    hashes=code_hashes()
    gate.require(manifest['code_sha256']==gate.digest(hashes),'manifest code drift')
    gate.require(set(inputs) in (set('version train features grids source_availability poison'.split()),set('version train features grids source_availability blocks poison'.split())),'unexpected input fields')
    gate.require(inputs['version']==1,'invalid inputs version')
    gate.keys(inputs['train'],'path sha256')
    gate.require(inputs['source_availability']==manifest['contract']['sources'],'source availability differs from manifest')
    base_features=inputs['features']
    gate.require(isinstance(base_features,list) and len(base_features)==111 and len(set(base_features))==111,'base111 requires 111 unique ordered features')
    grid_contract={name:{'start':ref['start'],'end':ref['end']} for name,ref in inputs['grids'].items()}
    gate.require(gate.digest(grid_contract)==manifest['contract']['grid_sha256'],'grid contract differs from manifest')
    train=_read_artifact(inputs['train']);validate_training(train,base_features)
    gate.require(mask_hash(train)==manifest['contract']['mask_sha256'],'training mask differs from manifest')
    gate.require(stage in ('preofficial','official'),'invalid stage')
    if stage=='preofficial':
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from expos.roll3 import fold_bounds,fold_cutoff
        gate.require(manifest['contract']['folds']==['1','2','3'],'canonical roll3 folds required')
        folds=[(f'f{k}',*fold_bounds(k),fold_cutoff(k)) for k in (1,2,3)]
    else:
        gate.require('official' in inputs['grids'],'official grid is missing')
        ref=inputs['grids']['official']
        official_start=pd.Timestamp('2026-05-31T23:00:00Z')
        official_end=pd.Timestamp('2026-09-26T23:00:00Z')
        gate.require(pd.Timestamp(ref['start'])==official_start and pd.Timestamp(ref['end'])==official_end,'official bounds drift')
        official_cutoff=pd.Timestamp(manifest['contract']['cutoff'])
        gate.require(official_cutoff.tzinfo is not None and official_cutoff<=pd.Timestamp('2026-05-29T00:00:00Z'),'official cutoff exceeds historical training boundary')
        folds=[('official',official_start,official_end,official_cutoff)]
    results=[]
    for name,start,end,cutoff in folds:
        ref=inputs['grids'][name];gate.keys(ref,'path sha256 start end')
        gate.require(pd.Timestamp(ref['start'])==start and pd.Timestamp(ref['end'])==end,'fold bounds drift')
        test=_read_artifact(ref)
        expected=pd.date_range(start,end,freq='h')
        validate_predictions(_grid_frame(test),expected)
        eligible=train[train.target_t<cutoff].copy()
        features=select_features(base_features,inputs.get('blocks',{}),manifest['recipe'],eligible,[test])
        context=dict(stage=name,manifest_sha256=manifest_id,code_hashes=hashes,
                     data_hashes={'train':inputs['train']['sha256'],'grid':ref['sha256']},
                     grid=dict(name=name,start=start.isoformat(),end=end.isoformat()),
                     mask_sha256=manifest['contract']['mask_sha256'],source_availability=inputs['source_availability'])
        for seed in manifest['contract']['seeds']:
            result=fit_heads(eligible,test,features,manifest['recipe'],seed,n_jobs,
                             Path(out)/'checkpoints',cutoff=cutoff,context=context)
            result=result.rename(columns={'origin':'origin_last_input_utc','h':'horizon_hours','prediction':'pred_kms'})
            result['origin_last_input_utc']=result['origin_last_input_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            path=Path(out)/'roll3'/f'{name}_s{seed}.parquet' if stage=='preofficial' else Path(out)/f'predictions_official_s{seed}.parquet'
            atomic_predictions(result,path);results.append(str(path))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--inputs',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--n-jobs',type=int,default=1)
    args=parser.parse_args()
    paths=run_stage(gate.read(args.manifest),gate.read(args.inputs),args.out,n_jobs=args.n_jobs)
    print('\n'.join(paths))


if __name__=='__main__':
    main()
