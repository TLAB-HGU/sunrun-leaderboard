"""Conditional deterministic Top30 TCN control versus frozen dropout=0.2.
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
# Must be fixed before torch imports or initializes any CUDA context.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
H, LOOKBACK = 72, 120
DILATIONS = (1, 2, 4, 8, 16, 32)
BASE_CONFIG = dict(optimizer="Adam", learning_rate=.001, weight_decay=0., width=64, dropout=0., checks_per_epoch=1)
EXPERIMENTS = {"control": {}, "dropout02": {"dropout": .2}}
ALIASES = {f'tcndrop_{arm}_s{seed}': (arm, seed)
           for arm in EXPERIMENTS for seed in (42, 43, 44)}


def deterministic_policy():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must remain :4096:8 before CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def configuration(experiment):
    arm = ALIASES[experiment][0] if experiment in ALIASES else experiment
    return {**BASE_CONFIG, **EXPERIMENTS[arm]}


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


def make_optimizer(net, config):
    optimizer = torch.optim.AdamW if config['optimizer'] == 'AdamW' else torch.optim.Adam
    return optimizer(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


def checkpoint_batches(n_batches, checks):
    return {int(np.ceil(n_batches*k/checks)) for k in range(1,checks+1)}


def validation_loss(net, validation, device):
    net.eval()
    total=count=0
    with torch.inference_mode():
        for values in DataLoader(validation,batch_size=256):
            hist,future,target,weight=[a.to(device) for a in values]
            n=int((weight>0).sum())
            if n:
                total+=float(weighted_mse(net(hist,future),target,weight))*n
                count+=n
    if not count:
        raise ValueError('no observed selection labels')
    return total/count*validation.stats['y_std']**2


def train(windows, validation, epochs, config, curve, device="cpu", seed=42,
          period="F1", stage="selection", max_steps=None, patience=10):
    # CPU is available only to bounded unit tests; experiment CLI requires CUDA.
    deterministic_policy()
    torch.manual_seed(seed)
    net=TCN(config).to(device)
    optimizer=make_optimizer(net, config)
    batches=DataLoader(windows,batch_size=256,shuffle=True,num_workers=0,
                       generator=torch.Generator().manual_seed(seed))
    if not len(batches) or (max_steps is not None and max_steps<1):
        raise ValueError('positive fitting step budget required')
    checkpoints=checkpoint_batches(len(batches),config['checks_per_epoch'])
    budget=max_steps if max_steps is not None else epochs*len(batches)
    best,best_epoch,best_step,best_position,steps,trace=float('inf'),0,0,0.,0,[]
    epoch=0
    while steps<budget:
        epoch+=1
        net.train()
        total=count=0
        for batch_index,values in enumerate(batches,1):
            hist,future,target,weight=[a.to(device) for a in values]
            n=int((weight>0).sum())
            if not n:
                raise ValueError('zero-observation training batch violates fitted window filter')
            optimizer.zero_grad(set_to_none=True)
            loss=weighted_mse(net(hist,future),target,weight)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            steps+=1
            total+=float(loss.detach())*n
            count+=n
            check=validation is not None and batch_index in checkpoints
            if check or batch_index==len(batches) or steps==budget:
                position=epoch-1+batch_index/len(batches)
                row=dict(epoch=epoch,epoch_fraction=position,step=steps,
                         train_weighted_mse=total/count*windows.stats['y_std']**2)
                if check:
                    value=validation_loss(net,validation,device)
                    row['validation_weighted_mse']=value
                    if value<best:
                        best,best_epoch,best_step,best_position=value,epoch,steps,position
                    net.train()
                trace.append(row)
                pd.DataFrame(trace).to_csv(curve,index=False)
                print(json.dumps(dict(row,period=period,stage=stage,unit=steps,units=budget,block=stage)),flush=True)
                if check and position-best_position>=patience:
                    return net,dict(best_epoch=best_epoch,best_step=best_step,steps=steps,batches_per_epoch=len(batches))
            if steps>=budget:
                break
    return net,dict(best_epoch=best_epoch,best_step=best_step,steps=steps,batches_per_epoch=len(batches))


def run_period(experiment,data,out,period,threads=1,smoke=False,device="cuda",seed=42):
    start=time.perf_counter()
    policy=deterministic_policy()
    torch.set_num_threads(threads)
    config=configuration(experiment)
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
    _,selection=train(windows(tr,stats),windows(va,stats),2 if smoke else 100,config,
                      out/f'{period}_selection_loss.csv',device,seed,period,'selection')
    best_epoch=selection['best_epoch']
    if best_epoch<1:
        raise ValueError('epoch selection failed')
    fit=np.arange(len(origins)) if not smoke else np.linspace(0,len(origins)-1,min(32,len(origins))).astype(int)
    stats=normalization(history,positions[fit],future[fit],y[fit])
    refit_steps=selection['best_step'] if config['checks_per_epoch']>1 else None
    net,refit=train(windows(fit,stats),None,best_epoch,config,out/f'{period}_train_loss.csv',
                    device,seed,period,'refit',max_steps=refit_steps)
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
    report=dict(experiment=experiment,period=period,smoke=smoke,seed=seed,device=device,backend='deterministic CUDA FP32' if device.startswith('cuda') else 'CPU FP32 unit fixture',
                deterministic_policy=policy,
                gpu_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
                source=str(Path(data).resolve()),history_features=FEATURES,mapped_history_features=[mapped(n) for n in FEATURES],
                future_features=FUTURE,history_rule='120 hourly h1 input snapshots ending at origin; causal ffill only',
                original_history_source='git309a44ce:research/e2000fs_feat30_pointweight4_tcn_lb120/results/source/data.py',
                training_cutoff=CUTOFFS[period],maturity_hours=144,train_start=str(origins[fit].min()),train_end=str(origins[fit].max()),
                excluded_incomplete_history_or_future_or_all_missing_target=removed,train_origins=len(fit),
                selection_train_end=str(origins[tr].max()),selection_valid_start=str(origins[va].min()),selection_valid_end=str(origins[va].max()),
                selected_epoch=best_epoch,epoch_budget=2 if smoke else 100,patience=10,batch_size=256,
                config=config,selection=selection,refit=refit,
                width=config['width'],dilations=DILATIONS,kernel=3,dropout=config['dropout'],
                optimizer=config['optimizer'],learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay'],gradient_clip=None,
                history_feature_count=len(FEATURES),future_feature_count=len(FUTURE),
                query_start=str(qo.min()),query_end=str(qo.max()),query_origins=len(qo),
                target_rule='campaign training-y ffill; remaining missing labels zero loss; point4 weights normalized over finite supplied labels per horizon',
                inference_seconds_per_fold=inference/len(qo),end_to_end_seconds=time.perf_counter()-start,
                versions=dict(numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__))
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False)
    provenance_path=Path(__file__).resolve().parent/'provenance.json'
    provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    report['git_commit']=revision.stdout.strip() if revision.returncode==0 else provenance.get('git_commit',os.environ.get('SUNRUN_GIT_COMMIT'))
    source_stat=Path(data).stat()
    report['input_provenance']=dict(bytes=source_stat.st_size,mtime_ns=source_stat.st_mtime_ns)
    report['source_provenance']=dict(path=str(Path(__file__).resolve()),bytes=Path(__file__).stat().st_size)
    report['selection_use']='pre-cutoff fitting holdout only; no official-query target access'
    (out/f'{period}_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(period=period,stage='complete',block='complete',unit=1,units=1)),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True,choices=sorted(set(EXPERIMENTS) | set(ALIASES)))
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--raw-data',type=Path,help='Accepted for campaign CLI consistency; not used')
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--period',choices=list(CUTOFFS))
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--device',choices=['cuda','cuda:0'],default='cuda')
    parser.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--merge',action='store_true')
    args=parser.parse_args()
    if args.experiment in ALIASES and ALIASES[args.experiment][1] != args.seed:
        parser.error('experiment ID seed must match --seed')
    if args.threads<1:
        parser.error('threads must be positive')
    if args.merge:
        print(json.dumps({'prediction_rows':merge(args.out)}));return
    deterministic_policy()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this DL experiment; wait for a GPU, no CPU fallback')
    for period in [args.period] if args.period else (['F1'] if args.smoke else list(CUTOFFS)):
        print(json.dumps(run_period(args.experiment,args.data,args.out,period,args.threads,args.smoke,args.device,args.seed)),flush=True)
    if not args.period and not args.smoke:
        print(json.dumps({'prediction_rows':merge(args.out)}))
    if not args.period:
        (args.out/'complete.json').write_text(json.dumps(dict(experiment=args.experiment,seed=args.seed,smoke=args.smoke,
            publish='not_required',publish_reason='smoke only' if args.smoke else 'controlled ablation; multi-seed official scoring and submission gate pending'),indent=2)+'\n')


if __name__=='__main__':
    main()
