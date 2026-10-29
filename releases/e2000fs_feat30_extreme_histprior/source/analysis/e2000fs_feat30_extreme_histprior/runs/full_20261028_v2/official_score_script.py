"""Official-window scoring with the pinned protected scorer (no upload, no new files outside owned paths).

``submit.py --dry-run`` would need a reproducible-script file under
leaderboard/model-release-publish/scripts/<member>/<experiment>/<config_sha>.py, which is outside
this worker's owned paths.  This reproduces the identical metric from the same hash-pinned scorer
and truth/fold tables that submit.py uses.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path('/home/t-lab01/sunrun/analysis/e2000fs_feat30_extreme_histprior')
sys.path.insert(0, str(STUDY))
sys.path.insert(1, '/home/t-lab01/sunrun/analysis/e2000fs_feat30_followups')
import evaluation  # noqa: E402
import run  # noqa: E402

RUN = STUDY / 'runs' / 'full_20261028_v2'
OUT = RUN / 'official'


def main():
    frames = []
    for stage in ('f1', 'final'):
        with np.load(OUT / f'{stage}_predictions.npz') as z:
            pred, origins = z['pred'], pd.DatetimeIndex(z['origins'], tz='UTC')
        frames.append(evaluation.frame(origins, pred))
    combined = pd.concat(frames, ignore_index=True)
    key = ['origin_last_input_utc', 'horizon_hours']
    if len(combined) != 357840 or combined.duplicated(key).any():
        raise RuntimeError('submission frame is incomplete or duplicated')
    parent_manifest = json.loads((run.PARENT / 'manifest.json').read_text())
    official = evaluation.official_score(combined, parent_manifest)
    truth_path = [Path(p) for p in parent_manifest['hashes'] if Path(p).name == 'truth.parquet'][0]
    folds_path = [Path(p) for p in parent_manifest['hashes'] if Path(p).name == 'folds.parquet'][0]
    truth = pd.read_parquet(truth_path)
    folds = pd.read_parquet(folds_path)
    joined = combined.merge(truth, on=key, validate='one_to_one')
    if len(joined) != len(combined):
        raise RuntimeError('pinned truth does not cover every prediction')
    joined = joined.merge(folds, on='origin_last_input_utc', how='left', validate='many_to_one')
    joined['err'] = joined.pred_kms - joined.target_kms
    fold_column = [c for c in folds.columns if c != 'origin_last_input_utc'][0]
    per_fold = {}
    for fold, part in joined.groupby(fold_column):
        high = part.target_kms >= run.FAST
        per_fold[str(fold)] = dict(
            n_origins=int(part.origin_last_input_utc.nunique()), n_pairs=int(len(part)),
            mse=float(np.mean(part.err ** 2)),
            mse_lt550=float(np.mean(part.err[~high] ** 2)),
            mse_ge550=float(np.mean(part.err[high] ** 2)),
            bias_ge550=float(np.mean(part.err[high])), bias_lt550=float(np.mean(part.err[~high])),
            share_ge550=float(high.mean()))
    high = joined.target_kms >= run.FAST
    result = dict(
        arm='hist_weighted_raw', rounds=run.ROUNDS, scorer=str(evaluation.SCORER),
        scorer_sha256=evaluation.digest(evaluation.SCORER),
        truth_sha256=evaluation.digest(truth_path), folds_sha256=evaluation.digest(folds_path),
        official_metric=official, per_fold=per_fold,
        pooled_band=dict(mse=float(np.mean(joined.err ** 2)),
                         mse_lt550=float(np.mean(joined.err[~high] ** 2)),
                         mse_ge550=float(np.mean(joined.err[high] ** 2)),
                         share_ge550=float(high.mean())),
        parent_official=dict(mse=3585.6357500367762, mse_ge550=6811.733770862298,
                             f1=4025.0, f2=3254.2, source='PROTOCOL.md / leaderboard'),
        reference_official=dict(mse=3249.842329310296, mse_ge550=8903.255190735916,
                                source='PROTOCOL.md / leaderboard e2000fs_dirmo_24_feat30'),
        note='submit.py --dry-run not run: its reproducible-script path is outside this '
             'worker\'s owned directories. Same pinned scorer and same pinned tables are used here.')
    run.write(OUT / 'official_score.json', result)
    print(json.dumps({k: result[k] for k in ('official_metric', 'per_fold', 'pooled_band')}, indent=1))
    total = official['mse']
    verdict = dict(
        total_mse=total,
        beats_parent=total < 3585.6357500367762,
        ge550_le_8903_3=per_fold.get('f1', {}).get('mse_ge550', 0) * 0 + result['pooled_band']['mse_ge550'] <= 8903.255190735916 * 0.95,
        beats_reference_total=total < 3249.842329310296,
        beats_reference_ge550=result['pooled_band']['mse_ge550'] < 8903.255190735916)
    run.write(OUT / 'verdict.json', verdict)
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
