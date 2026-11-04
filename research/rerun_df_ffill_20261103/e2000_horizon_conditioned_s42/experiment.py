"""Frozen E2000 direct/pooled recipes on the new causal feature-ffill snapshot.

Original feature-major interleaved order is preserved. Corrected new-data input
availability supersedes historical ACE+2h timing. Only observed training targets
enter fits; official query targets are never used. CPU QuantileDMatrix batching
avoids materializing the pooled float matrix (XGBoost 2.1 DataIter API).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np
import pandas as pd
import xgboost as xgb

H = 72
EXPERIMENTS = ("e2000_direct_s42", "e2000_horizon_conditioned_s42")
PARAMS = {"max_depth":3, "learning_rate":.03, "subsample":.8, "colsample_bytree":.7,
          "min_child_weight":20, "reg_lambda":5, "seed":42, "objective":"reg:squarederror",
          "tree_method":"hist", "device":"cpu", "eval_metric":"rmse", "max_bin":256}
CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}
QUERY_COUNTS = {"F1": 2089, "F2": 2113, "F3": 2113}
CONSTANT = ['v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']
FEATURES = ['rec26', 'rec27', 'rec28', 'rec_mean', 'v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing', 'enlil_dt_target']

def mapped(name):
    return name.replace("ch_ch_", "suvi_", 1) if name.startswith("ch_ch_") else name

def stale_series(raw):
    """Reconstruct original no-image/age>6 rule from raw table image observation ends."""
    raw = raw.sort_values("origin", kind="stable")
    duplicates = raw[raw.duplicated("origin", keep=False)]
    if len(duplicates) and (duplicates.groupby("origin")["img_age_h"].nunique(dropna=False) > 1).any():
        raise ValueError("train/query image timestamps disagree")
    raw = raw.drop_duplicates("origin").set_index("origin")
    ends = pd.Series(raw.index, index=raw.index) - pd.to_timedelta(raw.img_age_h, unit="h")
    if (ends.dropna() > ends.dropna().index).any():
        raise ValueError("image observation lies after its input origin")
    last_end = ends.ffill()
    age = (pd.Series(raw.index, index=raw.index) - last_end).dt.total_seconds() / 3600
    return (last_end.isna() | age.gt(6.0)).astype(np.float32)

def read_origin_table(path, columns):
    frame = pd.read_parquet(path, columns=["origin", *columns], filters=[("h", "==", 1)])
    frame["origin"] = pd.to_datetime(frame.origin, utc=True)
    return frame

def load_grid(path, period, training, stale):
    filters = [(f"fit_{period}", "==", True)] if training else [("set", "==", "query"), ("period", "==", period)]
    available = [mapped(c) for c in CONSTANT if c != "img_stale"]
    constants = pd.read_parquet(path, columns=["origin", *available], filters=[*filters, ("h", "==", 1)]).sort_values("origin", kind="stable")
    origins = pd.DatetimeIndex(pd.to_datetime(constants.origin, utc=True))
    if origins.has_duplicates:
        raise ValueError("duplicate origins in selected grid")
    constants["img_stale"] = stale.reindex(origins).to_numpy()
    C = constants[[mapped(c) for c in CONSTANT]].to_numpy(np.float32)
    rows = pd.read_parquet(path, columns=["origin", "h", "y", *VARYING], filters=filters).sort_values(["origin", "h"], kind="stable")
    if len(rows) != len(origins) * H or not np.array_equal(rows.h.to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (len(origins), H))):
        raise ValueError("incomplete origin/h grid")
    row_origins = pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H], utc=True))
    if not origins.equals(row_origins):
        raise ValueError("constant and varying origin grids differ")
    E = rows[VARYING].to_numpy(np.float32).reshape(len(origins), H, len(VARYING))
    Y = rows.y.to_numpy(np.float32).reshape(len(origins), H)
    if training and not (origins + pd.Timedelta(hours=144) <= pd.Timestamp(CUTOFFS[period], tz="UTC")).all():
        raise ValueError("training maturity mask exceeds cutoff")
    return origins, wide(C, E), Y

def feature_names(pooled=False):
    names = []
    for name in FEATURES:
        names.extend([f'{mapped(name)}@h{h}' for h in range(1, H+1)] if name in VARYING else [mapped(name)])
    return names + (['horizon_hours'] if pooled else [])


def wide(constants, varying):
    return np.hstack([varying[:, :, VARYING.index(name)] if name in VARYING
                      else constants[:, CONSTANT.index(name):CONSTANT.index(name)+1]
                      for name in FEATURES]).astype(np.float32)


class ObservedCells(xgb.DataIter):
    """Origin-major observed cells, releasing each dense float batch after ingestion."""
    def __init__(self, x, y, batch_size=2048):
        super().__init__(release_data=True)
        self.x, self.y = x, y
        self.rows, self.leads = np.nonzero(np.isfinite(y))
        self.batch_size = batch_size
        self.position = 0

    def reset(self):
        self.position = 0

    def next(self, input_data):
        if self.position >= len(self.rows):
            return False
        stop = min(self.position + self.batch_size, len(self.rows))
        rows, leads = self.rows[self.position:stop], self.leads[self.position:stop]
        batch = np.empty((len(rows), self.x.shape[1]+1), dtype=np.float32)
        batch[:, :-1], batch[:, -1] = self.x[rows], leads+1
        input_data(data=batch, label=self.y[rows, leads].astype(np.float32))
        self.position = stop
        return True


def fit(x, y, rounds, threads, pooled=False, batch_size=2048):
    if pooled:
        if not np.isfinite(y).any():
            raise ValueError('No observed training targets')
        matrix = xgb.QuantileDMatrix(ObservedCells(x, y, batch_size), nthread=threads, max_bin=256)
    else:
        keep = np.isfinite(y)
        if not keep.any():
            raise ValueError('No observed training targets')
        matrix = xgb.QuantileDMatrix(x[keep], label=y[keep], nthread=threads, max_bin=256)
    history = {}
    model = xgb.train(dict(PARAMS, nthread=threads), matrix, num_boost_round=rounds,
                      evals=[(matrix, 'train')], evals_result=history, verbose_eval=False)
    return model, history


def predict(model, xq, pooled=False):
    if pooled:
        return np.column_stack([model.inplace_predict(np.column_stack([xq, np.full(len(xq), h, dtype=np.float32)]))
                                for h in range(1, H+1)])
    return model.inplace_predict(xq)


def identity(path):
    stat = path.stat()
    return {'path':str(path.resolve()), 'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns}


def run_period(experiment, data, raw_data, out, period, threads=1, smoke=False, lead=None):
    stale = stale_series(read_origin_table(raw_data, ['img_age_h']))
    origins, x, y = load_grid(data, period, True, stale)
    qo, xq, _ = load_grid(data, period, False, stale)
    if len(qo) != QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query count')
    if smoke:
        take = np.linspace(0, len(origins)-1, min(96, len(origins))).astype(int)
        origins, x, y = origins[take], x[take], y[take]
        qo, xq = qo[:4], xq[:4]
    pooled = experiment == EXPERIMENTS[1]
    rounds = 2 if smoke else 900
    leads = [None] if pooled else ([lead] if lead is not None else list(range(1,H+1)))
    commit = subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True, check=False).stdout.strip()
    for current in leads:
        tag = f'{period}_pooled' if pooled else f'{period}_h{current:02d}'
        target = y if pooled else y[:,current-1]
        started = time.perf_counter()
        model, history = fit(x, target, rounds, threads, pooled)
        model.save_model(out / f'{tag}.ubj')
        t0 = time.perf_counter()
        pred = predict(model, xq, pooled)
        inference = time.perf_counter()-t0
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite predictions')
        np.savez_compressed(out / f'{tag}.npz', origins=qo.asi8, pred=pred)
        metadata = {'experiment':experiment,'period':period,'lead':current,'rounds':rounds,'params':dict(PARAMS,nthread=threads),
                    'features':feature_names(pooled),'feature_count':x.shape[1]+int(pooled),
                    'data':identity(data),'raw_data':identity(raw_data),'git_commit':commit,
                    'train_origin_start':str(origins.min()),'train_origin_end':str(origins.max()),
                    'cutoff':CUTOFFS[period],'maturity_hours':144,'observed_labels':int(np.isfinite(target).sum()),
                    'missing_labels':int((~np.isfinite(target)).sum()), 'queries':len(qo),
                    'availability':'new raw df corrected availability replaces historical ACE+2h timing',
                    'imputation':'features and training y causally ffilled upstream; query y unchanged; img_stale recovered from raw image observations',
                    'official_target_fits':False,'clipping':False,'inference_seconds':inference,
                    'seconds':time.perf_counter()-started,'history':history,
                    'libraries':{'numpy':np.__version__,'pandas':pd.__version__,'xgboost':xgb.__version__}}
        (out / f'{tag}.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps({'tag':tag,'rounds':rounds,'seconds':metadata['seconds']}),flush=True)


def merge(out, experiment):
    pooled = experiment == EXPERIMENTS[1]
    frames, identities, seconds = [], set(), 0.
    for period, count in QUERY_COUNTS.items():
        predictions, origins = [], None
        tags = [f'{period}_pooled'] if pooled else [f'{period}_h{h:02d}' for h in range(1,H+1)]
        for tag in tags:
            report = json.loads((out/f'{tag}.json').read_text())
            if report['experiment'] != experiment or report['period'] != period or report['rounds'] != 900 or report['features'] != feature_names(pooled):
                raise ValueError('Smoke or mismatched artifacts')
            identities.add(json.dumps([report['data'],report['raw_data']],sort_keys=True))
            seconds += report['inference_seconds']
            with np.load(out/f'{tag}.npz') as z:
                if origins is not None and not np.array_equal(origins,z['origins']):
                    raise ValueError('Origins differ')
                origins = z['origins'].copy()
                predictions.append(z['pred'].copy())
        prediction = predictions[0] if pooled else np.column_stack(predictions)
        if prediction.shape != (count,H) or not np.isfinite(prediction).all():
            raise ValueError('Invalid prediction grid')
        dates = pd.DatetimeIndex(origins,tz='UTC')
        frames.append(pd.DataFrame({'origin_last_input_utc':np.repeat(dates.strftime('%Y-%m-%dT%H:%M:%SZ'),H),
                                    'horizon_hours':np.tile(np.arange(1,H+1),count),'pred_kms':prediction.ravel()}))
    if len(identities) != 1:
        raise ValueError('Input snapshots differ')
    frame = pd.concat(frames,ignore_index=True)
    frame.to_parquet(out/'predictions.parquet',index=False)
    (out/'merge.json').write_text(json.dumps({'experiment':experiment,'rows':len(frame),
        'inference_seconds_per_fold':seconds/sum(QUERY_COUNTS.values())},indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=EXPERIMENTS,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--raw-data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--period',choices=QUERY_COUNTS)
    parser.add_argument('--lead',type=int,choices=range(1,H+1))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--merge',action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or (args.lead is not None and (args.period is None or args.experiment != EXPERIMENTS[0])):
        parser.error('Positive threads required; lead requires direct experiment and period')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.merge:
        merge(args.out,args.experiment)
        return
    for period in ([args.period or 'F1'] if args.period or args.smoke else QUERY_COUNTS):
        run_period(args.experiment,args.data,args.raw_data,args.out,period,args.threads,args.smoke,args.lead)
    if not args.smoke and args.period is None:
        merge(args.out,args.experiment)


if __name__ == "__main__":
    import sys
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", 'e2000_horizon_conditioned_s42'])
    main()
