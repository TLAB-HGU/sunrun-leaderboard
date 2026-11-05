"""Original two Top30 120h TCNs, retrained on the new feature-ffilled df.
History semantics proven by original results/source/data.py at git309a44ce:
historical varying features use h=1 at the historical origin. No raw labels
are used in features. Leading unavailable windows are excluded, never backfilled.
"""
from __future__ import annotations
import argparse
import json
import os
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
H, LOOKBACK, SEED = 72, 120, 42
DILATIONS = (1, 2, 4, 8, 16, 32)
EXPERIMENTS = {"e2000fs_feat30_pointweight4_tcn_lb120": False,
               "e2000fs_feat30_pointweight4_tcn_lb120_regskip": True}

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

def merge(out):
    out = Path(out)
    frames = []
    for period in CUTOFFS:
        z = np.load(out / f"{period}_predictions.npz")
        origins = pd.DatetimeIndex(z["origins"], tz="UTC")
        pred = z["pred"]
        if len(origins) != QUERY_COUNTS[period] or pred.shape != (len(origins), H) or not np.isfinite(pred).all():
            raise ValueError(f"{period}: incomplete full-run predictions")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), len(origins)), "pred_kms": pred.ravel()}))
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["origin_last_input_utc", "horizon_hours"]).any():
        raise ValueError("duplicate output keys")
    result.to_parquet(out / "predictions.parquet", index=False)
    return len(result)

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
    def __init__(self, cin, cout, dilation, regskip=False):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(cin, cout, 3, dilation=dilation)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout1d(.2) if regskip else nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.conv(nn.functional.pad(x, (self.pad, 0)))))


class TCN(nn.Module):
    def __init__(self, regskip=False):
        super().__init__()
        self.regskip = regskip
        width = 32 if regskip else 64
        widths = [30] + [width] * len(DILATIONS)
        self.encoder = nn.Sequential(*[CausalBlock(widths[i], widths[i+1], d, regskip) for i,d in enumerate(DILATIONS)])
        if regskip:
            self.register_buffer('current', torch.tensor(CURRENT, dtype=torch.long))
            self.head = nn.Sequential(nn.Linear(width+21+9+1,64), nn.ReLU(), nn.Dropout(.2),
                                      nn.Linear(64,32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32,1))
        else:
            self.head = nn.Sequential(nn.Linear(width+9+1,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
        self.register_buffer('horizon',torch.arange(1,73,dtype=torch.float32).view(1,72,1)/72)

    def forward(self, history, future):
        context = self.encoder(history.transpose(1,2))[:,:,-1]
        if self.regskip:
            context = torch.cat((context,history[:,-1,self.current]),dim=-1)
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
    rows = pd.read_parquet(data,columns=['origin','h','y',*FUTURE],filters=filters).sort_values(['origin','h'])
    if len(rows)%H:
        raise ValueError('incomplete horizon grid')
    origins = pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H],utc=True))
    if origins.has_duplicates or not np.array_equal(rows.h.to_numpy().reshape(-1,H),np.broadcast_to(np.arange(1,73),(len(origins),H))):
        raise ValueError('invalid horizon grid')
    if training and not (origins+pd.Timedelta(hours=144)<=pd.Timestamp(CUTOFFS[period],tz='UTC')).all():
        raise ValueError('maturity mask exceeds cutoff')
    return origins,rows[FUTURE].to_numpy(np.float32).reshape(-1,H,9),rows.y.to_numpy(np.float32).reshape(-1,H)


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


def train(windows,validation,epochs,regskip,curve,device="cpu"):
    if device not in ("cpu", "cuda:0"):
        raise ValueError("device must be cpu or cuda:0")
    torch.manual_seed(SEED)
    net=TCN(regskip).to(device)
    optimizer=(torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01) if regskip
               else torch.optim.Adam(net.parameters(),lr=.001))
    batches=DataLoader(windows,batch_size=256,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
    best,best_epoch,wait,trace=float('inf'),0,0,[]
    for epoch in range(1,epochs+1):
        if validation is not None and wait>=10:
            break
        net.train()
        total=count=0
        for hist,future,target,weight in batches:
            hist,future,target,weight=[a.to(device) for a in (hist,future,target,weight)]
            n=int((weight>0).sum())
            if not n:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss=weighted_mse(net(hist,future),target,weight)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            if regskip:
                nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total+=float(loss.detach())*n
            count+=n
        if not count:
            raise ValueError('no observed training labels')
        row=dict(epoch=epoch,train_weighted_mse=total/count*windows.stats['y_std']**2)
        if validation is not None:
            net.eval();total=count=0
            with torch.inference_mode():
                for hist,future,target,weight in DataLoader(validation,batch_size=256):
                    hist,future,target,weight=[a.to(device) for a in (hist,future,target,weight)]
                    n=int((weight>0).sum())
                    if n:
                        total+=float(weighted_mse(net(hist,future),target,weight))*n
                        count+=n
            if not count:
                raise ValueError('no observed validation labels')
            row['validation_weighted_mse']=total/count*windows.stats['y_std']**2
            if row['validation_weighted_mse']<best:
                best,best_epoch,wait=row['validation_weighted_mse'],epoch,0
            else:
                wait+=1
        trace.append(row)
        pd.DataFrame(trace).to_csv(curve,index=False)
        print(json.dumps(row),flush=True)
    return net,best_epoch


def run_period(experiment,data,out,period,threads=1,smoke=False,device="cpu"):
    start=time.perf_counter()
    torch.set_num_threads(threads)
    regskip=EXPERIMENTS[experiment]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    hourly,history=hourly_history(data)
    origins,future,y=load_grid(data,period,True)
    positions=history_positions(hourly,origins)
    good=complete_windows(history,positions)&np.isfinite(future).all(axis=(1,2))&np.isfinite(y).any(axis=1)
    removed=int((~good).sum())
    origins,positions,future,y=origins[good],positions[good],future[good],y[good]
    tr,va=selection_split(origins)
    if smoke:
        tr,va=tr[np.linspace(0,len(tr)-1,min(32,len(tr))).astype(int)],va[np.linspace(0,len(va)-1,min(4,len(va))).astype(int)]
    stats=normalization(history,positions[tr],future[tr],y[tr])
    def windows(indices,stats):
        return Windows(history,positions[indices],future[indices],y[indices],stats)
    _,best_epoch=train(windows(tr,stats),windows(va,stats),2 if smoke else 100,regskip,out/f'{period}_selection_loss.csv',device)
    if best_epoch<1:
        raise ValueError('epoch selection failed')
    fit=np.arange(len(origins)) if not smoke else np.linspace(0,len(origins)-1,min(32,len(origins))).astype(int)
    stats=normalization(history,positions[fit],future[fit],y[fit])
    net,_=train(windows(fit,stats),None,best_epoch,regskip,out/f'{period}_train_loss.csv',device)
    torch.save(net.state_dict(),out/f'{period}_model.pt')
    # Plain NumPy statistics accompany the network and feature order for replay.
    np.savez_compressed(out/f'{period}_normalization.npz',
                        **{f'{name}_{key}':value for name in ['history','future'] for key,value in stats[name].items()},
                        y_mean=stats['y_mean'],y_std=stats['y_std'],weight_mean=stats['weight_mean'])
    qo,qf,_=load_grid(data,period,False)
    if len(qo)!=QUERY_COUNTS[period]:
        raise ValueError('unexpected query count')
    qp=history_positions(hourly,qo)
    if not complete_windows(history,qp).all() or not np.isfinite(qf).all():
        raise ValueError('query history/future unavailable after causal ffill')
    if smoke:
        qo,qf,qp=qo[:2],qf[:2],qp[:2]
    query=Windows(history,qp,qf,np.full((len(qo),H),np.nan),stats)
    begin=time.perf_counter();parts=[];net.eval()
    with torch.inference_mode():
        for hist,fut,_,_ in DataLoader(query,batch_size=256):
            parts.append(net(hist.to(device),fut.to(device)).cpu().numpy()*stats['y_std']+stats['y_mean'])
    inference=time.perf_counter()-begin
    pred=np.clip(np.concatenate(parts).astype(np.float64),1.,3000.)
    if pred.shape!=(len(qo),H) or not np.isfinite(pred).all():
        raise ValueError('invalid predictions')
    np.savez_compressed(out/f'{period}_predictions.npz',pred=pred,origins=qo.asi8)
    report=dict(experiment=experiment,period=period,smoke=smoke,seed=SEED,device=device,backend='original CUDA FP32 restored' if device.startswith('cuda') else 'CPU FP32',
                source=str(Path(data).resolve()),history_features=FEATURES,mapped_history_features=[mapped(n) for n in FEATURES],
                future_features=FUTURE,history_rule='120 hourly h1 input snapshots ending at origin; causal ffill only',
                original_history_source='git309a44ce:research/e2000fs_feat30_pointweight4_tcn_lb120/results/source/data.py',
                training_cutoff=CUTOFFS[period],maturity_hours=144,train_start=str(origins[fit].min()),train_end=str(origins[fit].max()),
                excluded_incomplete_history_or_future_or_all_missing_target=removed,train_origins=len(fit),
                selection_train_end=str(origins[tr].max()),selection_valid_start=str(origins[va].min()),selection_valid_end=str(origins[va].max()),
                selected_epoch=best_epoch,epoch_budget=2 if smoke else 100,patience=10,batch_size=256,
                width=32 if regskip else 64,dilations=DILATIONS,kernel=3,dropout=.2 if regskip else 0.,
                optimizer='AdamW' if regskip else 'Adam',learning_rate=.0003 if regskip else .001,
                weight_decay=.01 if regskip else 0.,gradient_clip=1. if regskip else None,
                target_rule='campaign training-y ffill; remaining missing labels zero loss; point4 weights normalized over finite supplied labels per horizon',
                inference_seconds_per_fold=inference/len(qo),end_to_end_seconds=time.perf_counter()-start,
                versions=dict(numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__))
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False)
    report['git_commit']=revision.stdout.strip() if revision.returncode==0 else None
    (out/f'{period}_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True,choices=sorted(EXPERIMENTS))
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--raw-data',type=Path,help='Accepted for campaign CLI consistency; not used')
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--period',choices=list(CUTOFFS))
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--merge',action='store_true')
    args=parser.parse_args()
    if args.threads<1:
        parser.error('threads must be positive')
    if args.merge:
        print(json.dumps({'prediction_rows':merge(args.out)}));return
    for period in [args.period] if args.period else (['F1'] if args.smoke else list(CUTOFFS)):
        print(json.dumps(run_period(args.experiment,args.data,args.out,period,args.threads,args.smoke,args.device)),flush=True)
    if not args.period and not args.smoke:
        print(json.dumps({'prediction_rows':merge(args.out)}))


if __name__ == "__main__":
    import sys
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", 'e2000fs_feat30_pointweight4_tcn_lb120'])
    main()
