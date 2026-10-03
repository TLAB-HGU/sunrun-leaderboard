"""Frozen V-JEPA video fusion, May refit and causal 72h leaderboard forecasts.

Requires the pinned local ACE/CH/raw snapshots and hourly SUVI originals.
Downloads only the pinned public upstream and weights when no reference run is supplied.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
CONFIG=json.loads('{"data":{"arm":"video","checkpoint_sha256":"848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d","development_note":"Selection MSE improved5.675%; precision/recall regressed; uploaded for comparison at explicit user request","encoder":"V-JEPA2.1 ViT-B384 EMA encoder; frozen eval/inference_mode; BF16","evaluation":"2026-06-01..2026-09-30; 2833 hourly origins x72; retrospective exploratory","evaluation_snapshot":"38f1e75d4f4a864c04f5977f979378bb6f5df3f9","folds_sha256":"9c8f24f2cd82009cd7d071e99f45c9e15179326f7cccb4f46eb6805788332daf","frame_stride_hours":3,"frames_per_clip":16,"historical_image_arrival_verified":false,"image_size":384,"input_sha256":{"ace":"700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af","ch_hourly":"96f38e2728211345fcbbd2bb119e526cb429b74cae5370a5b41b864599a0870a","raw_ace_years":"695990b036c88073ed4d3f4b0338bc9a522872dba65f520884598e50e0ba7736"},"intensity_sha256":"f6ee9d396ed5a60f45a791ce303ed9955efbd2460ec96b61772a67e17884932d","lookback_hours":45,"max_past_fill_hours":6,"official_selection_or_hpo":false,"point_in_time_vintage_verified":false,"preprocessing":"physical RAD/DQF; north-up full disk; fixed train-only log intensity; replicate3","recurrence_days":[26,27,28],"reference_sha256":{"comparison_code_identity.json":"e2a6e72bc13081dcbeba089de8b095cd5552a5ca0126d3fe66d1914784bd3293","comparison_protocol.json":"08037c9df45c2146e78a618f5e0ffb8cada2372fdcff94a25ad99aeb37227dc5","execution-source-hashes.json":"3109a3011738837d4df23002bd90ef2187572458e86a3931a66334e3c527294b","intensity.json":"f6ee9d396ed5a60f45a791ce303ed9955efbd2460ec96b61772a67e17884932d","protected-source-hashes.json":"7b658878e7ca560fdba9288eb9bfc1751830cc0bca1e3419950efcb680e6f0de"},"source":"sun-db ACE + GOES-18 SUVI Fe195 original radiance + CH geometry/physics/recurrence","source_sha256":{"experiments/ch_breakthrough_v2/common.py":"23ec9a09ead017f4682749fdcb70d370d992f67a799eccd8c2c451b072c9b4c2","experiments/ch_breakthrough_v2/hpo.py":"b2aea0a61bcf43e9a4bfe9ea8e272884c013eaf623ed6883320513e3d4ffeef5","experiments/ch_breakthrough_v2/physical_quality.py":"a08a070ae2a9b244f078cc79597f04782e74dfdbf5fde301998c6b36f33aeec6","experiments/ch_breakthrough_v2/recurrence.py":"ce166d9acfee0f9ce7c2ffb635aa3bb5f25182119e0bfad291bb6169557bdfbb","experiments/mse20_four_regimes/protocol.py":"114c5e9108ec349977e68d39baa96a7de33daeee7dc01bb1d12ba0349e5b41e2","experiments/mse20_four_regimes/tree.py":"8935656403c08121d5bb2d37c77b52271495cdac086b96f6c54d6eb35085adeb","experiments/vjepa_frozen_fusion/compare.py":"150f0e144add6afb1368bd1cbbba9be53ffaf63bc69dbea7cc6f064455719abf","experiments/vjepa_frozen_fusion/encoder.py":"29d055d4a176a5b4b1cc67980ad39ffc07cf84ed9cb579af0e8c4679678d230c","experiments/vjepa_frozen_fusion/extract.py":"9a51b3793dc2c12a4e76cb1ac949537a652e4737bf83afd28402cbc6b9117423","experiments/vjepa_frozen_fusion/images.py":"af168515a2b5316961e7849a31fa40b1dd61541ccbd3f62eb7cecaf8f59c94c0","experiments/vjepa_frozen_fusion/numeric.py":"cef2fb72f0b903ffcd376451be905bd2a1dcffb28729573808f75077cc9f7bcc","experiments/vjepa_frozen_fusion/official.py":"f79a8b20d3120f3a3d04c4938f9bea8358cac2b19b84c8bb5683238423cbbdea","experiments/xgboost_suvi_fusion/events.py":"3a8482fe9f973d1a0390be4f5df0d88ba7832a196bd1549303d59cbb0c6475c2","experiments/xgboost_suvi_fusion/features.py":"f35276fbeeba307256cee2ffc005aea46b129e3615f51e9a75c83f6c8139152d"},"spatial_pools":["disk","central_meridian","equatorial_band","left","right"],"training_cutoff":"2026-05-31T23:00:00+00:00","upstream_sha":"204698b45b3712590f06245fbfba32d3be539812"},"hyperparameters":{"max_train_origins":6000,"model":"pooled72h residual XGBoost + frozen visual features","n_jobs":4,"output_range_kms":[1,3000],"pca_components":16,"pca_seed":42,"pca_solver":"randomized","reported_inference_device":"cpu","seed":42,"train_only_pca_imputation":true,"training_device":"cuda:0","xgboost":{"colsample_bytree":0.9,"learning_rate":0.07,"max_depth":5,"min_child_weight":2.0,"n_estimators":100,"reg_lambda":0.5,"subsample":0.7}}}')
CONFIG_SHA256='6f614c2197d82c404496f6024fce0f6e7c1f59f52284d4b7ed64b267ea80004f'
CHECKPOINT='vjepa2_1_vitb_dist_vitG_384.pt'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def validate(root,folds,reference):
    canonical=json.dumps(CONFIG,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest()!=CONFIG_SHA256 or Path(__file__).stem!=CONFIG_SHA256:
        raise ValueError('registered script/config identity mismatch')
    for rel,expected in CONFIG['data']['source_sha256'].items():
        if digest(ROOT/rel)!=expected:raise ValueError('source identity mismatch: '+rel)
    if digest(folds)!=CONFIG['data']['folds_sha256']:raise ValueError('folds identity mismatch')
    inputs={
        'ace':digest(root/'store/ch-v1/ace.parquet'),
        'ch_hourly':digest(root/'store/ch-v1/ch-hourly.parquet'),
        'raw_ace_years':hashlib.sha256(''.join(digest(p) for p in sorted((root/'store/ch-breakthrough-v2/upload/raw-ace-snapshot').glob('year=*/part.parquet'))).encode()).hexdigest(),
    }
    if inputs!=CONFIG['data']['input_sha256']:raise ValueError('numeric input identity mismatch')
    if reference is not None:
        for rel,expected in CONFIG['data']['reference_sha256'].items():
            if digest(reference/rel)!=expected:raise ValueError('reference identity mismatch: '+rel)
        if json.loads((reference/'comparison_protocol.json').read_text())['config']['params']!=CONFIG['hyperparameters']['xgboost']:
            raise ValueError('frozen tree params differ')

def bootstrap(run):
    ref=run.with_name(run.name+'-reference');ref.mkdir(parents=True,exist_ok=True)
    frozen=ROOT/'experiments/vjepa_frozen_fusion/leaderboard_reference'
    for name,expected in CONFIG['data']['reference_sha256'].items():
        if digest(frozen/name)!=expected:raise ValueError('packaged reference identity mismatch')
        target=ref/name
        if not target.exists():shutil.copyfile(frozen/name,target)
        if digest(target)!=expected:raise ValueError('existing bootstrap reference mismatch')
    upstream=ref/'upstream'
    if not upstream.exists():
        subprocess.run(['git','clone','--quiet','https://github.com/facebookresearch/vjepa2.git',str(upstream)],check=True)
        subprocess.run(['git','-C',str(upstream),'checkout','--quiet','--detach',CONFIG['data']['upstream_sha']],check=True)
    checkpoints=ref/'checkpoints';checkpoints.mkdir(exist_ok=True)
    path=checkpoints/CHECKPOINT
    if not path.exists():
        temporary=path.with_suffix('.partial')
        with urlopen('https://dl.fbaipublicfiles.com/vjepa2/'+CHECKPOINT) as source,temporary.open('wb') as target:
            shutil.copyfileobj(source,target)
        if digest(temporary)!=CONFIG['data']['checkpoint_sha256']:raise ValueError('checkpoint identity mismatch')
        temporary.replace(path)
    if digest(path)!=CONFIG['data']['checkpoint_sha256']:raise ValueError('checkpoint identity mismatch')
    return ref

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--reference-run',type=Path);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--metadata-root',type=Path,default=Path('/home/t-lab01/.local/state/sundb/stage/suvi_hourly'))
    p.add_argument('--images-root',type=Path,default=Path('/home/t-lab01/sunrun/sun-img/data'))
    p.add_argument('--mode',choices=['all','prepare','extract','predict'],default='all')
    p.add_argument('--output',type=Path)
    a=p.parse_args();run=a.run_dir.resolve()
    reference=a.reference_run.resolve() if a.reference_run else None
    validate(a.root,a.folds,reference)
    from experiments.vjepa_frozen_fusion import official,extract
    if a.mode in ('all','prepare') and not (run/'official_protocol.json').exists():
        reference=reference or bootstrap(run)
        validate(a.root,a.folds,reference)
        official.prepare(a.folds,run,reference,a.root,a.metadata_root,a.images_root)
    if a.mode in ('all','extract'):extract.extract(run,batch_size=4,workers=6)
    if a.mode in ('all','predict'):
        from experiments.ch_breakthrough_v2.common import acquire_gpu_lease
        with acquire_gpu_lease(ROOT/'store/ch-breakthrough-v2/gpu-leases') as gpu:
            result=official.predict(a.folds,run,a.root,f'cuda:{gpu}')
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/'predictions.parquet',a.output)
        print(json.dumps({'config_sha256':CONFIG_SHA256,'manifest':str(run/'official_manifest.json'),'predictions':str(a.output or run/'predictions.parquet'),'inference_seconds_per_fold':result['inference_seconds_per_fold']},indent=2))

if __name__=='__main__':main()
