"""Frozen forecast residual replay; no fitting and no forecast selection."""
from __future__ import annotations
import argparse
import importlib.util
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CAMP = ROOT/'research/post_ablation_five_axes_20261108'
PRIMARY = ROOT/'research/drop035_v650_confirm_20261108'

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

competition=module('official_competition',ROOT/'leaderboard/scorer/competition.py')
kernel=module('audit_kernel',CAMP/'analysis_kernel.py')

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def identity(path):
    path=Path(path);st=path.stat()
    return dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns)

def frame(origins,pred):
    return pd.DataFrame({'origin_last_input_utc':np.repeat(origins,72),
                        'horizon_hours':np.tile(np.arange(1,73),len(origins)),
                        'pred_kms':pred.ravel()})

def summarize(values):
    v=np.asarray(values,float)
    return dict(values=v.tolist(),mean=float(v.mean()),sample_variance=float(v.var(ddof=1)),
                sample_sd=float(v.std(ddof=1)),worst=float(v.max()))

def errorsummary(preds,truth,selected):
    mask=np.broadcast_to(selected,truth.shape)&np.isfinite(truth)
    if not mask.any():return {'n_pairs':0}
    e=(preds-truth[None])[:,mask]
    return dict(n_pairs=int(mask.sum()),seed_mae=np.abs(e).mean(1).tolist(),
                mean_mae=float(np.abs(e).mean()),mean_rmse=float(np.sqrt((e*e).mean(1)).mean()),
                mean_bias=float(e.mean()),mean_squared_error=float((e*e).mean()))

def run(count):
    assignment=json.loads((CAMP/'assignment.json').read_text())
    observations=competition.load_observations(assignment['observations'])
    context=competition.prepare_context(observations)
    truth,origins=context['truth'],context['origins']
    paths=[Path(assignment['observations']),ROOT/'leaderboard/scorer/competition.py',
           CAMP/'analysis_kernel.py',Path(__file__),OUT/'residual_strategy.md']
    names,arrays,official=[],[],{}
    for family in ('best_single_configuration','original_A_plus_B1'):
        for seed in (42,43,44):
            directory=(PRIMARY/f'runs/drop035_v650_s{seed}' if family=='best_single_configuration'
                       else CAMP/f'reference/s{seed}')
            parts=[]
            for period in ('F1','F2','F3'):
                path=directory/f'{period}_predictions.npz';z=np.load(path)
                parts.append(frame(pd.to_datetime(z['origins'],utc=True),z['pred']));paths.append(path)
            f=pd.concat(parts,ignore_index=True);name=f'{family}_s{seed}'
            aligned=competition.align_predictions(f,context)
            assert np.isfinite(aligned).all() and (aligned>0).all()
            names.append(name);arrays.append(aligned);official[name]=competition.score_competition(f,context)
    columns=['origin','h','v_lag0','v_lag24','img_age_h','rec27','enlil_n_pending']
    data=pd.read_parquet(assignment['data'],columns=columns,filters=[('set','==','query')])
    data['origin']=pd.to_datetime(data.origin,utc=True)
    assert not data.duplicated(['origin','h']).any()
    constant=data[data.h.eq(1)].set_index('origin').reindex(origins)
    recurrence=data.pivot(index='origin',columns='h',values='rec27').reindex(index=origins,columns=range(1,73)).to_numpy()
    controls={'official_persistence':np.broadcast_to(context['persistence'][:,None],truth.shape),
              'input_persistence':np.broadcast_to(constant.v_lag0.to_numpy()[:,None],truth.shape),
              'input_rec27':recurrence}
    for name,arr in controls.items():
        official[name]=competition.score_competition(frame(origins,arr),context)
        arr=np.where(np.isfinite(arr)&(arr>0),arr,context['persistence'][:,None])
        arrays.append(arr);names.append(name)
    arrays=np.stack(arrays)
    np.savez_compressed(OUT/'residual_arrays.npz',names=np.array(names),origins=origins.asi8,
                        period=context['period'],predictions=arrays,truth=truth)
    save('residual_official_scores.json',official)
    exact,draws,rejected=kernel.bootstrap_scores(context,arrays,count,168)
    for i,name in enumerate(names):
        for key in kernel.METRICS:
            np.testing.assert_allclose(exact[key][i],official[name]['total'][key],rtol=1e-10,atol=1e-8)
    stats={key:summarize(exact[key][:3]) for key in kernel.METRICS}
    stats['pooled_mae']=summarize([np.abs((a-truth)[np.isfinite(truth)]).mean() for a in arrays[:3]])
    comparisons={}
    for i,name in enumerate(names[6:],6):
        observed=1-exact['s'][:3].mean()/exact['s'][i]
        boot=1-draws['s'][:,:3].mean(1)/draws['s'][:,i]
        bestboot=1-draws['s'][:,0]/draws['s'][:,i]
        comparisons[name]={'configuration_mean_relative_reduction':float(observed),
                           'configuration_ci95':np.quantile(boot,[.025,.975]).tolist(),
                           'best_single_relative_reduction':float(1-exact['s'][0]/exact['s'][i]),
                           'best_single_ci95':np.quantile(bestboot,[.025,.975]).tolist(),
                           'control_s':float(exact['s'][i])}
    paired=1-draws['s'][:,:3]/draws['s'][:,3:6]
    np.savez_compressed(OUT/'residual_bootstrap.npz',**draws)
    save('residual_summary.json',dict(metric='official S, user-confirmed W',names=names,primary=stats,
         secondary={key:summarize(exact[key][3:6]) for key in kernel.METRICS},
         paired_configuration_reduction_vs_A_plus_B1=float((1-exact['s'][:3]/exact['s'][3:6]).mean()),
         paired_configuration_reduction_ci95=np.quantile(paired.mean(1),[.025,.975]).tolist(),
         comparisons=comparisons,bootstrap=dict(replicates=count,block_hours=168,seed=20261104,rejected=rejected),
         best_single_s=float(exact['s'][0]),half_best_single_s_threshold=float(exact['s'][0]/2),
         best_single_s_ci95=np.quantile(draws['s'][:,0],[.025,.975]).tolist(),
         statement='Intervals condition on exposed periods and fitted seeds; no unseen-generalization or 50% attainability claim.'))
    pred=arrays[:3];strata={};leads=np.arange(1,73)[None,:]
    for lo,hi in [(1,6),(7,24),(25,48),(49,72)]:
        strata[f'lead_{lo}_{hi}']=errorsummary(pred,truth,(leads>=lo)&(leads<=hi))
    for lo,hi in [(0,400),(400,550),(550,700),(700,float('inf'))]:
        label=f'{lo}_{hi}';v=constant.v_lag0.to_numpy()[:,None]
        strata['target_speed_'+label]=errorsummary(pred,truth,(truth>=lo)&(truth<hi))
        strata['origin_speed_'+label]=errorsummary(pred,truth,(v>=lo)&(v<hi))
    delta=(constant.v_lag0-constant.v_lag24).to_numpy()[:,None]
    for label,mask in [('fall',delta<=-50),('stable',np.abs(delta)<50),('rise',delta>=50)]:
        strata['origin_delta24_'+label]=errorsummary(pred,truth,mask)
    series=observations.set_index(pd.DatetimeIndex(pd.to_datetime(observations.timestamp_utc,utc=True))).speed_kms.sort_index()
    valid=series.dropna();positions=valid.index.get_indexer(origins,method='ffill');assert (positions>=0).all()
    age=(origins.asi8-valid.index.asi8[positions])/3.6e12
    gap=np.column_stack([series.reindex(origins-pd.Timedelta(hours=h)).isna().to_numpy() for h in range(24)]).sum(1)
    for label,condition in [('official_age_gt6',age>6),('past24_has_gap',gap>0),
                            ('image_age_gt24',constant.img_age_h.to_numpy()>24),
                            ('pending_enlil',constant.enlil_n_pending.to_numpy()>0)]:
        for state in (False,True):strata[f'{label}_{state}']=errorsummary(pred,truth,(condition==state)[:,None])
    for period in ('F1','F2','F3'):strata[period]=errorsummary(pred,truth,(context['period']==period)[:,None])
    save('residual_strata.json',strata)
    timing=[]
    for shift in (-12,-6,0,6,12):
        p=pred[:,:,12+shift:60+shift];y=truth[:,12:60]
        timing.append({'prediction_lead_shift_hours':shift,**errorsummary(p,y,np.ones_like(y,bool))})
    adequate=np.isfinite(truth).sum(1)>=48;y=truth[adequate];yp=pred[:,adequate,:]
    observed_range=np.nanmax(y,axis=1)-np.nanmin(y,axis=1)
    predicted_range=np.max(np.where(np.isfinite(y)[None],yp,-np.inf),axis=2)-np.min(np.where(np.isfinite(y)[None],yp,np.inf),axis=2)
    peak_shift=np.argmax(np.where(np.isfinite(y)[None],yp,-np.inf),axis=2)-np.nanargmax(y,axis=1)[None]
    amplitude={'origins_with_at_least_48_observations':int(adequate.sum()),
               'median_observed_range':float(np.median(observed_range)),
               'median_predicted_range_by_seed':np.median(predicted_range,axis=1).tolist(),
               'median_range_ratio_by_seed':np.median(predicted_range/observed_range[None],axis=1).tolist(),
               'median_absolute_peak_timing_hours_by_seed':np.median(np.abs(peak_shift),axis=1).tolist()}
    save('residual_timing_amplitude.json',dict(fixed_shift_diagnostics=timing,amplitude=amplitude,
         warning='Truth-dependent diagnostic only; no best shift chosen, no correction, not an attainable lower bound.'))
    mw,tw=kernel.metric_weights(truth,np.arange(len(origins)))
    sq=np.where(np.isfinite(truth)[None],(pred-truth[None])**2,0).mean(0)
    targets=origins.asi8[:,None]+np.arange(1,73)[None,:]*3600*10**9
    table=pd.DataFrame({'target_ns':targets.ravel(),'truth':truth.ravel(),
                        'sse':sq.ravel(),'timse_contribution':(sq*tw/72).ravel(),
                        'mse_contribution':(sq*mw/72).ravel()})
    table=table[np.isfinite(table.truth)].groupby('target_ns').agg(
        truth=('truth','first'),forecast_pairs=('truth','size'),sse=('sse','sum'),
        timse_contribution=('timse_contribution','sum'),mse_contribution=('mse_contribution','sum'))
    table['timestamp_utc']=pd.to_datetime(table.index,utc=True)
    table.sort_values('timse_contribution',ascending=False).to_csv(OUT/'residual_target_contributions.csv',index=False)
    for k in (1,5,10,24):print(json.dumps({'top_target_count':k,'timse_fraction':float(table.timse_contribution.nlargest(k).sum()/table.timse_contribution.sum())}),flush=True)
    save('residual_provenance.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,files=[identity(p) for p in paths],
         data=identity(assignment['data']),file_digest_gap='Automated workspace hook rejected collection under older No Hashes policy; not bypassed.',
         command='CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python analysis/w50_audit_20261108/residual_diagnostics.py --bootstrap 2000',
         no_training=True,no_target_fitting=True))
    print(json.dumps({'stage':'complete','primary_s':stats['s'],'comparisons':comparisons}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=2000)
    run(parser.parse_args().bootstrap)
