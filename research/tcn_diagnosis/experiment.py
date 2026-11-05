"""Deterministic F1 pre-cutoff fitting diagnosis for the Top30 120h TCN.
History semantics proven by original results/source/data.py at git309a44ce:
historical varying features use h=1 at the historical origin. No raw labels
are used in features. Leading unavailable windows are excluded, never backfilled.
"""
from __future__ import annotations
import argparse
import json
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
import subprocess
import time
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
H, LOOKBACK = 72, 120
DILATIONS = (1, 2, 4, 8, 16, 32)
BASE_CONFIG = dict(optimizer="Adam", learning_rate=.001, weight_decay=0., width=64, dropout=0., checks_per_epoch=1)


FEATURES = ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last',
            'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26',
            'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last',
            'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6',
            'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed',
            'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark',
            'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28']

CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}

QUERY_COUNTS = {"F1": 2089, "F2": 2113, "F3": 2113}

VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']

def mapped(name):
    return name.replace("ch_ch_", "suvi_", 1) if name.startswith("ch_ch_") else name

def selection_split(origins):
    dates = pd.DatetimeIndex(origins)
    start = dates.max() - pd.Timedelta(days=81)
    train = np.flatnonzero(dates + pd.Timedelta(hours=144) < start)
    valid = np.flatnonzero(dates >= start)
    if not len(train) or not len(valid):
        raise ValueError("insufficient history for purged epoch selection")
    return train, valid

FUTURE = [name for name in FEATURES if name in VARYING]
CURRENT = [i for i,name in enumerate(FEATURES) if name not in VARYING]


class CausalBlock(nn.Module):
    def __init__(self, cin, cout, dilation, dropout=0.):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(cin, cout, 3, dilation=dilation)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.conv(nn.functional.pad(x, (self.pad, 0)))))


class TCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, dropout = config['width'], config['dropout']
        widths = [30] + [width] * len(DILATIONS)
        self.encoder = nn.Sequential(*[CausalBlock(widths[i], widths[i+1], d, dropout) for i,d in enumerate(DILATIONS)])
        layers = [nn.Linear(width+9+1,128), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(128,64), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(64,1))
        self.head = nn.Sequential(*layers)
        self.register_buffer('horizon',torch.arange(1,73,dtype=torch.float32).view(1,72,1)/72)

    def forward(self, history, future):
        context = self.encoder(history.transpose(1,2))[:,:,-1]
        context = context[:,None,:].expand(-1,H,-1)
        return self.head(torch.cat((context,future,self.horizon.expand(len(history),-1,-1)),dim=-1)).squeeze(-1)


def hourly_history(data):
    cols = [mapped(n) for n in FEATURES]
    raw = pd.read_parquet(data,columns=['origin',*cols],filters=[('h','==',1)])
    raw['origin'] = pd.to_datetime(raw.origin,utc=True)
    duplicates = raw[raw.duplicated('origin',keep=False)]
    if len(duplicates) and duplicates.groupby('origin')[cols].nunique(dropna=False).gt(1).any().any():
        raise ValueError('conflicting train/query hourly features')
    raw = raw.drop_duplicates('origin').set_index('origin').sort_index()
    hourly = pd.date_range(raw.index.min(),raw.index.max(),freq='h',tz='UTC')
    values = raw.reindex(hourly).ffill()[cols].to_numpy(np.float32)
    if np.isinf(values).any():
        raise ValueError('infinite history features')
    return hourly,values


def history_positions(hourly, origins):
    positions = hourly.get_indexer(origins)
    return positions


def complete_windows(history, positions):
    good = positions >= LOOKBACK-1
    good &= positions < len(history)
    # Prefix counts avoid materializing N*120*30 windows.
    bad = (~np.isfinite(history).all(axis=1)).astype(np.int64)
    count = np.r_[0,np.cumsum(bad)]
    ix = np.flatnonzero(good)
    ends = positions[ix]
    good[ix] &= count[ends+1] == count[ends-LOOKBACK+1]
    return good


def load_grid(data,period,training):
    filters = [(f'fit_{period}','==',True)] if training else [('set','==','query'),('period','==',period)]
    columns = ['origin', 'h', *FUTURE] + (['y'] if training else [])
    rows = pd.read_parquet(data,columns=columns,filters=filters).sort_values(['origin','h'])
    if len(rows)%H:
        raise ValueError('incomplete horizon grid')
    origins = pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H],utc=True))
    if origins.has_duplicates or not np.array_equal(rows.h.to_numpy().reshape(-1,H),np.broadcast_to(np.arange(1,73),(len(origins),H))):
        raise ValueError('invalid horizon grid')
    if training and not (origins+pd.Timedelta(hours=144)<=pd.Timestamp(CUTOFFS[period],tz='UTC')).all():
        raise ValueError('maturity mask exceeds cutoff')
    targets = rows.y.to_numpy(np.float32).reshape(-1,H) if training else np.full((len(origins),H),np.nan,np.float32)
    return origins,rows[FUTURE].to_numpy(np.float32).reshape(-1,H,9),targets


def normalization(history,positions,future,y):
    used=np.zeros(len(history),dtype=bool)
    for end in positions:
        used[end-LOOKBACK+1:end+1]=True
    stats={}
    for tag,arr in [('history',history[used]),('future',future.reshape(-1,9))]:
        if not np.isfinite(arr).all():
            raise ValueError('unfilled leading inputs: ffill cannot use future values')
        stats[tag]=dict(mean=arr.mean(axis=0,dtype=np.float64).astype(np.float32),
                        std=np.maximum(arr.std(axis=0,dtype=np.float64),1e-6).astype(np.float32))
    observed=np.isfinite(y)
    if not observed.any(axis=0).all():
        raise ValueError('each horizon requires observed training targets')
    stats['y_mean']=float(y[observed].mean())
    stats['y_std']=max(float(y[observed].std()),1e-6)
    stats['weight_mean']=(np.where(observed,np.where(y>=550,4.,1.),0.).sum(axis=0)/observed.sum(axis=0)).astype(np.float32)
    return stats


class Windows(Dataset):
    def __init__(self,history,positions,future,y,stats):
        self.history,self.positions,self.future,self.y,self.stats=history,positions,future,y,stats

    def __len__(self):
        return len(self.positions)

    def __getitem__(self,i):
        end=int(self.positions[i])
        hs,fs=self.stats['history'],self.stats['future']
        history=((self.history[end-119:end+1]-hs['mean'])/hs['std']).astype(np.float32)
        future=((self.future[i]-fs['mean'])/fs['std']).astype(np.float32)
        actual=self.y[i]
        observed=np.isfinite(actual)
        target=np.where(observed,(actual-self.stats['y_mean'])/self.stats['y_std'],0.).astype(np.float32)
        weight=np.where(observed,np.where(actual>=550.,4.,1.)/self.stats['weight_mean'],0.).astype(np.float32)
        return history,future,target,weight


def weighted_mse(pred,target,weight):
    count=(weight>0).sum()
    if count==0:
        return pred.sum()*0.
    return (weight*(pred-target).square()).sum()/count


def configure_determinism(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32)


def predict_eval(net, windows, device):
    net.eval()
    parts=[]
    with torch.inference_mode():
        for history,future,_,_ in DataLoader(windows,batch_size=256,shuffle=False):
            value=net(history.to(device),future.to(device)).cpu().numpy()
            parts.append(value*windows.stats['y_std']+windows.stats['y_mean'])
    result=np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite fitting diagnostic predictions')
    return result


def residual_metrics(pred, target, weight_mean, row_mask=None, speed='all'):
    valid=np.isfinite(target)
    if row_mask is not None:
        valid &= np.asarray(row_mask)[:,None]
    if speed=='below550': valid &= target<550
    elif speed=='atleast550': valid &= target>=550
    elif speed!='all': raise ValueError('unknown speed band')
    count=int(valid.sum())
    if not count:
        return dict(count=0,rmse=None,mae=None,bias=None,weighted_mse=None)
    errors=(pred.astype(np.float64)-target.astype(np.float64))[valid]
    weights=np.broadcast_to(np.where(target>=550,4.,1.)/weight_mean,target.shape)[valid]
    return dict(count=count,rmse=float(np.sqrt(np.mean(errors**2))),
                mae=float(np.abs(errors).mean()),bias=float(errors.mean()),
                weighted_mse=float(np.mean(weights*errors**2)))


def partition_masks(origins, train, valid):
    masks={name:np.zeros(len(origins),dtype=bool) for name in ['fit_all','fit_older','fit_recent81d','stopping_holdout']}
    recent_start=origins[train].max()-pd.Timedelta(days=81)
    masks['fit_all'][train]=True
    masks['fit_older'][train]=origins[train]<recent_start
    masks['fit_recent81d'][train]=origins[train]>=recent_start
    masks['stopping_holdout'][valid]=True
    return masks


def decompose(pred, target, origins, masks, weight_mean, trial, epoch):
    records=[]
    months=origins.strftime('%Y-%m').to_numpy()
    for subset,mask in masks.items():
        for month in ['all',*sorted(set(months[mask]))]:
            selected=mask if month=='all' else mask&(months==month)
            for speed in ['all','below550','atleast550']:
                records.append(dict(trial=trial,epoch=epoch,subset=subset,month=month,speed=speed,
                                    **residual_metrics(pred,target,weight_mean,selected,speed)))
    return records


def compare_arrays(left,right):
    if left.shape!=right.shape:
        return dict(exact=False,max_abs=None,shape_equal=False)
    return dict(exact=bool(np.array_equal(left,right)),shape_equal=True,
                max_abs=float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)))) if left.size else 0.)


def compare_trials(left,right):
    result={'curves':compare_arrays(np.asarray(left['curve']),np.asarray(right['curve'])), 'checkpoints':{}}
    for epoch in left['snapshots']:
        a,b=left['snapshots'][epoch],right['snapshots'][epoch]
        tensors={name:compare_arrays(value.numpy(),b['state'][name].numpy()) for name,value in a['state'].items()}
        result['checkpoints'][str(epoch)]=dict(predictions=compare_arrays(a['pred'],b['pred']),
            tensors_exact=all(v['exact'] for v in tensors.values()),
            tensor_max_abs=max(v['max_abs'] for v in tensors.values()),tensors=tensors)
    result['exact']=result['curves']['exact'] and all(v['predictions']['exact'] and v['tensors_exact'] for v in result['checkpoints'].values())
    return result


def train_trial(fitting,all_windows,valid,optimizer_name,seed,device,epochs=5,progress=None):
    configure_determinism(seed)
    net=TCN(BASE_CONFIG).to(device)
    cls=torch.optim.Adam if optimizer_name=='Adam' else torch.optim.AdamW
    optimizer=cls(net.parameters(),lr=.001,weight_decay=0.)
    loader=DataLoader(fitting,batch_size=256,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
    snapshots={};curve=[]
    for epoch in range(1,epochs+1):
        net.train();total=count=0
        for history,future,target,weight in loader:
            history,future,target,weight=[x.to(device) for x in [history,future,target,weight]]
            n=int((weight>0).sum())
            if not n: raise ValueError('empty fitting batch')
            optimizer.zero_grad(set_to_none=True)
            loss=weighted_mse(net(history,future),target,weight)
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            loss.backward();optimizer.step()
            total+=float(loss.detach())*n;count+=n
        # Always evaluate the stopping holdout at the end of an epoch.
        holdout_pred=predict_eval(net,valid,device)
        holdout_loss=residual_metrics(holdout_pred,valid.y,valid.stats['weight_mean'])['weighted_mse']
        curve.append([epoch,total/count*fitting.stats['y_std']**2,holdout_loss])
        if epoch in {1,epochs}:
            snapshots[epoch]=dict(state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                                  pred=predict_eval(net,all_windows,device))
        if progress: progress(epoch,curve[-1])
    return dict(curve=curve,snapshots=snapshots)


def run(args):
    start=time.perf_counter();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback for fitting diagnostics')
    torch.set_num_threads(args.threads)
    settings=configure_determinism(args.seed)
    hourly,history=hourly_history(args.data)
    origins,future,y=load_grid(args.data,'F1',True)
    positions=history_positions(hourly,origins)
    good=complete_windows(history,positions)&np.isfinite(future).all(axis=(1,2))&np.isfinite(y).any(axis=1)
    origins,positions,future,y=origins[good],positions[good],future[good],y[good]
    tr,va=selection_split(origins)
    if args.smoke:
        tr=tr[np.linspace(0,len(tr)-1,min(32,len(tr))).astype(int)]
        va=va[np.linspace(0,len(va)-1,min(4,len(va))).astype(int)]
    # Exclude the purge gap from residual populations and diagnostic prediction output.
    selected=np.r_[tr,va]
    origins,positions,future,y=origins[selected],positions[selected],future[selected],y[selected]
    tr,va=np.arange(len(tr)),np.arange(len(tr),len(selected))
    stats=normalization(history,positions[tr],future[tr],y[tr])
    def windows(ix): return Windows(history,positions[ix],future[ix],y[ix],stats)
    fitting,valid,all_windows=windows(tr),windows(va),windows(np.arange(len(origins)))
    masks=partition_masks(origins,tr,va)
    trials={};records=[];epochs=2 if args.smoke else 5
    for index,(name,opt) in enumerate([('adam_a','Adam'),('adam_b','Adam'),('adamw0','AdamW')]):
        def progress(epoch,row):
            print(json.dumps(dict(period='F1',stage=name,block='diagnosis',unit=index*epochs+epoch,
                                  units=3*epochs,epoch=epoch,train_weighted_mse=row[1],
                                  stopping_holdout_weighted_mse=row[2])),flush=True)
        result=train_trial(fitting,all_windows,valid,opt,args.seed,args.device,epochs,progress)
        trials[name]=result
        pd.DataFrame(result['curve'],columns=['epoch','train_online_weighted_mse','stopping_holdout_weighted_mse']).to_csv(out/f'{name}_loss.csv',index=False)
        for epoch,snapshot in result['snapshots'].items():
            torch.save(snapshot['state'],out/f'{name}_epoch{epoch}_model.pt')
            np.savez_compressed(out/f'diagnostic_predictions_{name}_epoch{epoch}.npz',pred=snapshot['pred'],
                                training_target=y,origins=origins.asi8,fit_mask=np.isin(np.arange(len(origins)),tr))
            records.extend(decompose(snapshot['pred'],y,origins,masks,stats['weight_mean'],name,epoch))
    comparison=dict(adam_repeat=compare_trials(trials['adam_a'],trials['adam_b']),
                    adam_vs_adamw0=compare_trials(trials['adam_a'],trials['adamw0']))
    fit_losses=[residual_metrics(trials['adam_a']['snapshots'][e]['pred'],y,stats['weight_mean'],masks['fit_all'])['weighted_mse'] for e in [1,epochs]]
    holdout_losses=[residual_metrics(trials['adam_a']['snapshots'][e]['pred'],y,stats['weight_mean'],masks['stopping_holdout'])['weighted_mse'] for e in [1,epochs]]
    fitting_decrease=1-fit_losses[1]/fit_losses[0]
    holdout_increase=holdout_losses[1]/holdout_losses[0]-1
    gate=dict(fitting_weighted_mse=fit_losses,holdout_weighted_mse=holdout_losses,
              fitting_relative_decrease=fitting_decrease,holdout_relative_increase=holdout_increase,
              memorization_pattern=fitting_decrease>=.2 and holdout_increase>=.1,
              followup_allowed=not args.smoke and comparison['adam_repeat']['exact'] and fitting_decrease>=.2 and holdout_increase>=.1)
    report=dict(experiment=args.experiment,seed=args.seed,smoke=args.smoke,period='F1',
                reproducibility_passed=comparison['adam_repeat']['exact'],comparisons=comparison,gate=gate,
                deterministic=settings,official_labels_read=False,official_scoring=False,
                diagnostic_only=True,cutoff=CUTOFFS['F1'],maturity_hours=144,
                fit_origins=len(tr),holdout_origins=len(va),train_start=str(origins[tr].min()),
                train_end=str(origins[tr].max()),holdout_start=str(origins[va].min()),holdout_end=str(origins[va].max()),
                history_features=FEATURES,future_features=FUTURE,config=BASE_CONFIG,
                target_caveat='campaign causally forward-filled training y; not raw official observations',
                versions=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__),
                input_bytes=args.data.stat().st_size,source_bytes=Path(__file__).stat().st_size,
                device=args.device,seconds=time.perf_counter()-start)
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True)
    provenance=Path(__file__).with_name('provenance.json')
    report['git_commit']=revision.stdout.strip() if revision.returncode==0 else (
        json.loads(provenance.read_text()).get('git_commit') if provenance.exists() else os.environ.get('SUNRUN_GIT_COMMIT'))
    for name in ['diagnostics.json','F1_metadata.json']:
        (out/name).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(records).to_csv(out/'diagnosis_metrics.csv',index=False)
    (out/'complete.json').write_text(json.dumps(dict(publish='not_required',publish_reason='Fitting-only reproducibility/residual diagnosis; no official forecasts',
        reproducibility_passed=report['reproducibility_passed']),indent=2)+'\n')
    print(json.dumps(dict(period='F1',stage='complete',unit=3*epochs,units=3*epochs,
                          reproducibility_passed=report['reproducibility_passed'])),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description='Deterministic F1 pre-cutoff fitting diagnosis only')
    p.add_argument('--experiment',default='diagnostic')
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cuda:0'],default='cuda')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.threads<1: p.error('threads must be positive')
    run(args)

if __name__=='__main__': main()
