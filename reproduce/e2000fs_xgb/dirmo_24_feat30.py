"""e2000fs_dirmo_24_feat30: DirMO-24 XGBoost joint multi-output trees on the Top-30 feature groups.

Three independent 24-output blocks (h1-24, h25-48, h49-72), each on 237 columns (21 constant + 9 horizon-varying
groups, 24 h of horizon features), fit separately for F1 (targets < 2026-02-26) and F2 (targets < 2026-05-29).
The Top-30 groups are the permutation-importance ranking of the all-feature DirMO-24 model, weighted
(2137*F1 + 2833*F2) / 4970. That ranking used already scored official periods: official-exposed, post-hoc research.

    python reproduce/e2000fs_xgb/dirmo_24_feat30.py --cache reproduce/outputs/pointweight4_cache [--smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xgb_common as C  # noqa: E402
from common import paths  # noqa: E402
import train_predict as tp  # noqa: E402  (design, shared with the LightGBM arms)


def select_top30(importance_csv):
    ranking = pd.read_csv(importance_csv).pivot(index='feature', columns='stage', values='perm_delta_mse_mean')
    ranking['weighted_delta_mse'] = (ranking.f1 * 2137 + ranking.final * 2833) / 4970
    ranking = ranking.reset_index().sort_values(['weighted_delta_mse', 'feature'], ascending=[False, True])
    return ranking.head(30).feature.tolist(), ranking


def run(cache, out, smoke=False, reference=None):
    out = Path(out)
    (out / 'loss').mkdir(parents=True, exist_ok=True)
    started = time.time()
    selected, ranking = select_top30(paths.data('inputs/feature_importance.csv'))
    assert selected == C.FROZEN['selected_groups'], 'Top-30 ranking differs from the frozen selection'
    ranking.to_csv(out / 'feature_ranking.csv', index=False)
    names, bundles, series = C.load_cache(cache)
    params = C.xgb_params(smoke)
    assert sum(n in selected for n in names['constant']) == 21 and sum(n in selected for n in names['varying']) == 9
    train = bundles['train']
    oo_list, preds = [], []
    for stage, fit, val, target, oo, origins, y in C.stages(bundles, series, smoke):
        pred = np.empty((len(oo), 72))
        for b, (s, e) in enumerate(C.BLOCKS):
            tag = f'{stage}_b{b:02}'
            print(f'[{time.time() - started:6.0f}s] {tag}: fitting {len(fit)} origins, rounds={params["n_estimators"]}', flush=True)
            X, Xt = tp.design(train, names, selected, s - 1), tp.design(target, names, selected, s - 1)
            assert X.shape[1] == Xt.shape[1] == 237
            model = xgb.XGBRegressor(**params)
            model.fit(X[fit], y[fit, s - 1:e], eval_set=[(X[fit], y[fit, s - 1:e]), (X[val], y[val, s - 1:e])], verbose=False)
            C.loss_frame(model, params['n_estimators']).to_csv(out / 'loss' / f'{tag}_loss.csv', index=False)
            pred[:, s - 1:e] = np.clip(model.predict(Xt), 1., 3000.)
        assert np.isfinite(pred).all()
        oo_list.append(oo)
        preds.append(pred)
    result = C.finish(out, oo_list, preds, smoke, reference, 'dirmo_24_feat30',
                      None if smoke else C.CONFIG['expected']['dirmo_24_feat30']['mse'])
    result['elapsed_seconds'] = round(time.time() - started, 1)
    (out / 'result.json').write_text(json.dumps(C.ev.json_safe(result), indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default=str(paths.OUTPUT_ROOT / 'pointweight4_cache'))
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reference', default=None, help='published forecast.npz to diff against')
    args = parser.parse_args()
    out = args.out or paths.OUTPUT_ROOT / ('dirmo_24_feat30_smoke' if args.smoke else 'dirmo_24_feat30')
    result = run(Path(args.cache), out, args.smoke, args.reference)
    print(json.dumps({k: v for k, v in result.items() if k != 'official'}, indent=2))


if __name__ == '__main__':
    main()
