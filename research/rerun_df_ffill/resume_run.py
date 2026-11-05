"""Resume owned immutable calib11/top30 runs after host reboot, without changing kernels.

Byte-compare the experiment snapshot with runner_at_launch.py and require the
launch-record data size/mtime. Only complete, verified blocks/models are reused.
No content hashes, old external outputs, or interrupted partial boosters are used.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
import zipfile
import numpy as np
import pandas as pd
import xgboost as xgb


def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def identity(source,launch_source,data,out,expected_bytes,expected_mtime_ns,family,experiment):
    source,launch_source,data,out=map(lambda p:Path(p).resolve(),(source,launch_source,data,out))
    if source.parent!=out or source.name!='experiment.py':
        raise ValueError('Source must be the owned output directory experiment.py snapshot')
    if source.read_bytes()!=launch_source.read_bytes():
        raise ValueError('Immutable experiment source differs from runner_at_launch.py')
    stat=data.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(expected_bytes,expected_mtime_ns):
        raise ValueError('Data differs from launch-record bytes/mtime')
    return dict(source=str(source),launch_source=str(launch_source),data=dict(path=str(data),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),
                family=family,experiment=experiment)


def import_snapshot(source):
    spec=importlib.util.spec_from_file_location('immutable_resume_experiment',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def owned(path,out):
    path=Path(path).resolve()
    if path.parent!=Path(out).resolve():raise ValueError('Artifact is outside the owned run directory')
    return path


def xgb_complete(path,rounds,features,outputs):
    model=xgb.Booster(model_file=path)
    config=json.loads(model.save_config())['learner']['learner_model_param']
    if model.num_boosted_rounds()!=rounds or model.num_features()!=features or int(config['num_target'])!=outputs:
        raise ValueError('Booster rounds/features/output count disagree')
    return model


def loss_complete(path,rounds,json_format=False):
    if json_format:
        history=json.loads(Path(path).read_text())
        values=np.asarray(history['train']['observed_rmse'],float)
    else:
        table=pd.read_csv(path)
        if not np.array_equal(table['round'],np.arange(1,rounds+1)):raise ValueError('Loss round sequence is incomplete')
        values=table.iloc[:,1:].to_numpy(float)
    if len(values)!=rounds or not np.isfinite(values).all():raise ValueError('Loss curve is incomplete or nonfinite')


def fresh_enough(paths,source,data):
    threshold=max(Path(source).stat().st_mtime_ns,Path(data).stat().st_mtime_ns)
    if any(Path(p).stat().st_mtime_ns<threshold for p in paths):raise ValueError('Artifacts predate the recorded input/source snapshot')


def prediction_complete(path,origins,width,leads=None):
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z['origins'],origins.asi8):raise ValueError('Prediction origins disagree with current input')
        p=z['pred']
        if p.shape!=(len(origins),width) or not np.isfinite(p).all() or not (p>0).all():raise ValueError('Invalid prediction grid')
        if leads is not None and not np.array_equal(z['leads'],leads):raise ValueError('Prediction lead order differs')


def calib_identity(module,record,period,rounds):
    recipe=module.RECIPES[record['experiment']]
    return dict(experiment=record['experiment'],period=period,data=record['data'],rounds=rounds,normalization=recipe[1],scheme=recipe[0])


def validate_calib_block(module,record,out,period,block,rounds,threads,training,query):
    stem=Path(out)/f'{period}_b{block}'
    report=json.loads(stem.with_suffix('.json').read_text())
    expected=calib_identity(module,record,period,rounds)
    for key,value in expected.items():
        if report[key]!=value:raise ValueError('Block identity differs: '+key)
    features=module.feature_names(record['experiment'],block)
    if report['block']!=block or report['features']!=features or report['feature_count']!=len(features):raise ValueError('Block features differ')
    if report['params']!=dict(module.PARAMS,nthread=threads):raise ValueError('Block parameters differ')
    y=training[3];anchor=module.RECIPES[record['experiment']][2]
    if report['residual_anchor']!=anchor:raise ValueError('Residual anchor differs')
    labels=y-training[1][:,0,None].astype(np.float64) if anchor else y
    observed=np.isfinite(labels[:,block*24:(block+1)*24]);keep=observed.any(1)
    if report['train_origins']!=int(keep.sum()) or report['observed_per_lead']!=observed.sum(0).tolist():raise ValueError('Training support differs')
    if report['training_origin_start']!=str(training[0].min()) or report['training_origin_end']!=str(training[0].max()):raise ValueError('Training origins differ')
    prediction_complete(stem.with_suffix('.npz'),query[0],24)
    xgb_complete(stem.with_suffix('.ubj'),rounds,len(features),24)
    loss_complete(stem.with_suffix('.loss.json'),rounds,True)
    fresh_enough([stem.with_suffix(s) for s in ('.json','.npz','.ubj','.loss.json')],record['source'],record['data']['path'])


def validate_top_block(module,record,out,period,block,threads,smoke=False):
    engine,kind,first_weighted,clip=module.RECIPES[record['experiment']]
    if engine=='histogram':raise ValueError('Histogram source has no complete loss history; resume proof unavailable')
    training,query=module.load_block(Path(record['data']['path']),period,block,smoke)
    stem=Path(out)/f'{period}_b{block}';report=json.loads(stem.with_suffix('.json').read_text())
    features=module.feature_names(block,kind=='wavelet');leads=list(range(block*24+1,(block+1)*24+1))
    expected=dict(status='complete',experiment=record['experiment'],period=period,block=block,leads=leads,smoke=smoke,
                  source=record['data']['path'],feature_names=features,feature_count=len(features),maturity_hours=144,training_cutoff=module.CUTOFFS[period])
    for key,value in expected.items():
        if report[key]!=value:raise ValueError('Top30 block identity differs: '+key)
    if report['train_origins_before_target_mask']!=len(training[0]) or report['training_origin_first']!=str(training[0].min()) or report['training_origin_last']!=str(training[0].max()):raise ValueError('Top30 training origins differ')
    prediction_complete(stem.with_suffix('.npz'),query[0],24,leads)
    fits=report['fits'];models=report['model_files']
    if len(fits)!=(1 if engine=='xgb' else 24) or len(models)!=len(fits):raise ValueError('Incomplete model list')
    proof=[stem.with_suffix('.json'),stem.with_suffix('.npz')]
    for i,(fit,name) in enumerate(zip(fits,models)):
        path=owned(name,out)
        y=training[3] if engine=='xgb' else training[3][:,i]
        rounds=2 if smoke else (900 if engine in ('xgb','lgb') else module.LGB_TUNED['n_estimators'])
        if fit['rounds']!=rounds:raise ValueError('Wrong completed rounds')
        if engine=='xgb':
            expected_model=stem.with_suffix('.ubj')
            keep=np.isfinite(y).any(1)
            params=dict(module.XGB_PARAMS,nthread=threads,base_score=float(np.nanmean(y[keep])),disable_default_eval_metric=1)
            expected_loss=kind if kind!='point' or block>=first_weighted else 'plain'
            xgb_complete(path,rounds,len(features),24)
        else:
            expected_model=Path(out)/f'{period}_h{leads[i]:02}.txt'
            if fit['lead']!=leads[i]:raise ValueError('Wrong scalar lead')
            keep=np.isfinite(y)
            params=dict(module.LGB_PARAMS if engine=='lgb' else module.LGB_TUNED,n_jobs=threads)
            if smoke:params['n_estimators']=2
            expected_loss=kind
            import lightgbm as lgb
            model=lgb.Booster(model_file=str(path))
            if model.current_iteration()!=rounds or model.num_feature()!=len(features):raise ValueError('LightGBM rounds/features disagree')
        if path!=expected_model.resolve() or fit['parameters']!=params or fit['loss']!=expected_loss or fit['train_origins']!=int(keep.sum()):raise ValueError('Model recipe/support differs')
        loss=path.with_name(path.stem+'_loss.csv');loss_complete(loss,rounds);proof.extend([path,loss])
    fresh_enough(proof,record['source'],record['data']['path'])


def valid(check):
    try:check();return True,None
    except (OSError,EOFError,zipfile.BadZipFile,ValueError,KeyError,TypeError,IndexError,json.JSONDecodeError,xgb.core.XGBoostError) as e:return False,str(e)


def invalidate_block(out,period,block):
    # Preserve evidence; move only names belonging to this owned block.
    out=Path(out);names=[out/f'{period}_b{block}.json',out/f'{period}_b{block}.npz']
    backup=out/'resume_invalid'/f'{period}_b{block}_{time.time_ns()}'
    for path in names:
        if path.exists():
            backup.mkdir(parents=True,exist_ok=True);path.rename(backup/path.name)


def receipt(out,phase,action,reason=None):
    record=dict(phase=phase,action=action,reason=reason,utc=pd.Timestamp.now(tz='UTC').isoformat())
    with (Path(out)/'resume_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


def phase(out,name,fn):
    runid=uuid.uuid4().hex[:10];directory=Path(out)/'resume_phases';directory.mkdir(exist_ok=True)
    report=dict(phase=name,started_utc=pd.Timestamp.now(tz='UTC').isoformat(),exit_code=None)
    path=directory/f'{name}_{runid}.json';atomic_json(path,report)
    try:
        with (directory/f'{name}_{runid}.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):fn()
    except BaseException:
        report.update(exit_code=1,error=traceback.format_exc(),finished_utc=pd.Timestamp.now(tz='UTC').isoformat());atomic_json(path,report);raise
    report.update(exit_code=0,finished_utc=pd.Timestamp.now(tz='UTC').isoformat());atomic_json(path,report)


def run(source,launch_source,data,out,expected_bytes,expected_mtime_ns,family,experiment,threads=1,smoke=False):
    if threads<1:raise ValueError('threads must be positive')
    if family not in ('calib11','top30'):raise ValueError('Unsupported resume family')
    out=Path(out).resolve();data=Path(data).resolve()
    record=identity(source,launch_source,data,out,expected_bytes,expected_mtime_ns,family,experiment)
    record['threads']=threads;record['smoke']=smoke
    checkpoint=out/'resume_identity.json'
    if checkpoint.exists() and json.loads(checkpoint.read_text())!=record:raise ValueError('Run resume identity changed')
    atomic_json(checkpoint,record)
    complete=out/'resume_complete.json'
    if complete.exists():os.replace(complete,out/'resume_previous_complete.json')
    module=import_snapshot(source)
    if experiment not in module.RECIPES:raise ValueError('Recipe absent from immutable snapshot')
    if family=='top30' and module.RECIPES[experiment][0]=='histogram':raise ValueError('Histogram resumability requires a loss receipt not emitted by this snapshot')
    completed=0
    def recheck_identity():
        identity(source,launch_source,data,out,expected_bytes,expected_mtime_ns,family,experiment)
    for period in (['F1'] if smoke else module.PERIODS):
        recheck_identity()
        if family=='calib11':
            rounds=2 if smoke else module.RECIPES[experiment][3][list(module.PERIODS).index(period)]
            training,query=module.load(data,experiment,period)
            original_fit=module.fit
            def resume_fit(x,y,weights,normalization,rounds,threads,model_path):
                model_path=owned(model_path,out)
                def verify():
                    xgb_complete(model_path,rounds,x.shape[1],y.shape[1]);loss_complete(model_path.with_suffix('.loss.json'),rounds,True)
                    fresh_enough([model_path,model_path.with_suffix('.loss.json')],source,data)
                # Only calibration heads are reusable without a completed block report.
                reuse,reason=valid(verify) if '_cal_b' in model_path.stem else (False,'Final block must pass its complete report')
                if reuse:
                    receipt(out,model_path.stem,'reuse_calibration_model')
                    model=xgb_complete(model_path,rounds,x.shape[1],y.shape[1]);model.set_param({'nthread':threads});observed=np.isfinite(y);keep=observed.any(1)
                    return model,dict(train_origins=int(keep.sum()),dropped_no_target=int((~keep).sum()),observed_targets=int(observed[keep].sum()),observed_per_lead=observed[keep].sum(0).tolist())
                return original_fit(x,y,weights,normalization,rounds,threads,model_path)
            module.fit=resume_fit
            def check_weights():
                if json.loads((out/f'{period}_weights.json').read_text())!=calib_identity(module,record,period,rounds):raise ValueError('Weight recipe identity differs')
                with np.load(out/f'{period}_w.npz') as z:
                    if not np.array_equal(z['origins'],training[0].asi8) or z['w'].shape!=training[3].shape or not np.isfinite(z['w']).all() or (z['w']<0).any():raise ValueError('Weight grid differs')
                    if module.RECIPES[experiment][0]=='sera':
                        expected=module.sera_weights(training[3]).astype(np.float32)
                    else:
                        split=int(len(training[3])*.75)
                        if not np.array_equal(z['cal_origins'],training[0][split:].asi8):raise ValueError('Calibration origins differ')
                        rho=z['cal_mse']/z['cal_timse']
                        if not np.array_equal(rho,z['rho']):raise ValueError('Calibration rho differs')
                        y=training[3]
                        expected=np.where(np.isfinite(y),1+rho*np.isfinite(y).sum(axis=0)*module.timse_weights(y),0).astype(np.float32)
                        for b in range(3):
                            path=out/f'{period}_cal_b{b}.ubj'
                            xgb_complete(path,rounds,len(module.feature_names(experiment,b)),24)
                            loss_complete(path.with_suffix('.loss.json'),rounds,True)
                            fresh_enough([path,path.with_suffix('.loss.json')],source,data)
                    if not np.array_equal(z['w'],expected):raise ValueError('Weight values differ from original recipe')
                fresh_enough([out/f'{period}_weights.json',out/f'{period}_w.npz'],source,data)
            ready,reason=valid(check_weights)
            if not ready:
                receipt(out,period+'_weights','rebuild',reason)
                phase(out,period+'_weights',lambda:module.train_period(data,experiment,period,[],rounds,threads,out,'weights'))
                check_weights()
            else:receipt(out,period+'_weights','reuse')
        for block in ([0] if smoke else range(3)):
            recheck_identity()
            check=(lambda:validate_calib_block(module,record,out,period,block,rounds,threads,training,query)) if family=='calib11' else (lambda:validate_top_block(module,record,out,period,block,threads,smoke))
            ready,reason=valid(check)
            if not ready:
                receipt(out,f'{period}_b{block}','retrain',reason);invalidate_block(out,period,block)
                action=(lambda:module.train_period(data,experiment,period,[block],rounds,threads,out,'fit')) if family=='calib11' else (lambda:module.fit_block(experiment,data,out,period,block,threads,smoke))
                def verified_action():
                    action();recheck_identity();check()
                phase(out,f'{period}_b{block}',verified_action)
            else:receipt(out,f'{period}_b{block}','reuse')
            completed+=1
        if family=='calib11':module.fit=original_fit
    recheck_identity()
    if not smoke:
        if completed!=9:raise ValueError('Not all nine blocks are complete')
        phase(out,'merge',lambda:module.merge(out))
    atomic_json(out/'resume_complete.json',dict(status='smoke_complete' if smoke else 'complete',blocks=completed,identity=record,finished_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','launch-source','data','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--data-bytes',type=int,required=True);p.add_argument('--data-mtime-ns',type=int,required=True)
    p.add_argument('--family',choices=('calib11','top30'),required=True);p.add_argument('--experiment',required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.source,a.launch_source,a.data,a.out,a.data_bytes,a.data_mtime_ns,a.family,a.experiment,a.threads,a.smoke)


if __name__=='__main__':main()
