"""Finish the registered comparison from completed immutable model artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from experiment import END, START, TRUTH, score_arrays
from summarize import ARMS, SEEDS, aggregate

HERE = Path(__file__).resolve().parent


def load_predictions(arm, seed):
    frame = pd.read_parquet(HERE / 'results' / arm / f'seed{seed}' / 'predictions.parquet')
    frame = frame.sort_values(['origin', 'h'])
    origins = pd.DatetimeIndex(frame.origin.iloc[::72])
    if not np.array_equal(frame.h.to_numpy().reshape(-1, 72), np.tile(np.arange(1, 73), (len(origins), 1))):
        raise ValueError('Unexpected prediction grid')
    return origins, frame.pred_kms.to_numpy().reshape(-1, 72), frame.y.to_numpy().reshape(-1, 72)


def paired_bootstrap(arm, replicates=1000):
    """Paired seven-day blocks, same resampled origins across all seeds/models."""
    cases = []
    common_origins = None
    for seed in SEEDS:
        origin, pred, truth = load_predictions(arm, seed)
        ref_origin, ref, ref_truth = load_predictions('sera_l2', seed)
        if not origin.equals(ref_origin) or not np.array_equal(truth, ref_truth, equal_nan=True):
            raise ValueError('Paired comparison needs exactly identical observations')
        if common_origins is not None and not origin.equals(common_origins):
            raise ValueError('Seed origin grids differ')
        common_origins = origin
        cases.append((pred, ref, truth))
    groups = ((common_origins - common_origins[0]).total_seconds() // (7 * 86400)).astype(int)
    blocks = [np.where(groups == block)[0] for block in np.unique(groups)]
    rng = np.random.default_rng(20261103)
    deltas = []
    for _ in range(replicates):
        selected = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        delta = [score_arrays(pred[selected], truth[selected])['W']
                 - score_arrays(ref[selected], truth[selected])['W'] for pred, ref, truth in cases]
        deltas.append(float(np.mean(delta)))
    interval = np.quantile(deltas, [.025, .975])
    return {'arm': arm, 'reference': 'sera_l2', 'replicates': replicates, 'block_days': 7,
            'blocks': len(blocks), 'random_seed': 20261103,
            'mean_W_difference_bootstrap': float(np.mean(deltas)),
            'ci95_W_difference': interval.tolist(), 'ci_excludes_no_gain': bool(interval[1] < 0),
            'limit': 'Seven-day blocks do not exhaust possible longer solar recurrence dependence.'}


def main():
    observations = pd.read_parquet(TRUTH, columns=['target_time', 'y'],
                                   filters=[('set', '==', 'train'), ('h', '==', 1),
                                            ('target_time', '>=', START), ('target_time', '<', END)],
                                   use_threads=False).set_index('target_time').y
    for arm in ARMS:
        for seed in SEEDS:
            origin, pred, truth = load_predictions(arm, seed)
            target_times = origin.asi8[:, None] + pd.to_timedelta(np.arange(1, 73), unit='h').asi8[None, :]
            independent = observations.reindex(pd.to_datetime(target_times.ravel(), utc=True)).to_numpy().reshape(truth.shape)
            if not np.array_equal(truth, independent, equal_nan=True):
                raise ValueError(f'{arm}/{seed}: saved truth differs from original observed target')
            actual = score_arrays(pred, truth)
            stored = json.loads((HERE / 'results' / arm / f'seed{seed}' / 'metrics.json').read_text())
            for key in ('W', 'mae', 'rmse', 'bias'):
                if not np.isclose(actual[key], stored[key], rtol=1e-10, atol=1e-10):
                    raise ValueError(f'{arm}/{seed}: persisted {key} differs from fresh scoring')
    result = aggregate()
    bootstrap = [paired_bootstrap(arm) for arm in result['bootstrap_pending_candidates']]
    result['bootstrap'] = bootstrap
    result['bootstrap_pending_candidates'] = []
    result['internal_supported_candidates'] = [b['arm'] for b in bootstrap if b['ci_excludes_no_gain']]
    result['baseline_update'] = False
    result['official_submission'] = False
    result['decision_limit'] = 'Internal 2025Q3 only; official same-condition multi-seed comparison still required.'
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    rows = result['summary']
    lines = ['# 조건부 노이즈 손실 3축 실험 결과', '',
             '2025 Q3의 사전 지정 내부 시간순 검증 결과다. 공식 리더보드 S 및 SOTA와 직접 비교하지 않는다.', '',
             '| arm | W 평균 ± seed SD | 최악 W | MAE | RMSE | ΔW vs SERA |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['W_mean']:.3f} ± {row['W_std']:.3f} | "
                     f"{row['W_worst']:.3f} | {row['mae_mean']:.3f} | {row['rmse_mean']:.3f} | "
                     f"{row['W_change_pct_vs_sera']:+.2f}% |")
    lines += ['', '| 분포 arm | NLL(km/s 단위) | 50% coverage | 90% coverage | 95% coverage |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        if 'nll_mean' in row:
            lines.append(f"| {row['arm']} | {row['nll_mean']:.4f} | {row['coverage50_mean']:.1%} | "
                         f"{row['coverage90_mean']:.1%} | {row['coverage95_mean']:.1%} |")
    lines += ['', '## 축별 판정', '']
    by_arm = {r['arm']: r for r in rows}
    for axis in result['axes']:
        name = axis['arm']
        supported = axis['distribution_hypothesis_supported']
        lines.append(f"- {name} vs {axis['parent']}: 분포 가설 {'지지' if supported else '미충족'}, "
                     f"부모 대비 W {axis['W_change_pct_vs_parent']:+.2f}%.")
        row = by_arm[name]
        if not row['absolute_coverage_pass']:
            lines.append('  사전 기준의 90/95% 구간 포함률 오차 조건을 만족하지 못했다. '
                         '분산 추정 및 학습 y 보간과 관측 truth 차이를 원인 후보로 남긴다.')
        if not row['point_candidate']:
            lines.append('  SERA 대비 W 1% 개선 및 MAE/RMSE 악화 제한을 동시에 만족하지 못해 성능 채택 후보가 아니다.')
        elif not any(b['arm'] == name and b['ci_excludes_no_gain'] for b in bootstrap):
            lines.append('  점성능 후보 조건은 통과했으나 시간 블록 bootstrap으로 개선을 확정하지 못했다.')
        else:
            lines.append('  내부 점성능·bootstrap 조건 통과. 공식 조건 검증 전이므로 기준선에는 편입하지 않는다.')
    lines += ['', '## 한계와 다음 판단', '',
              '- 혼합 주변분포를 조건부 분포의 증거로 간주하지 않았다. 모든 초기값은 학습 구간에서만 적합했다.',
              '- 학습 y ffill을 고정했다. NLL은 보간 타깃에 적합되며 calibration은 실제 관측으로 검증했다.',
              '- solver는 고정된 Fisher 대각 근사이며 L2와 분포 arm의 출력 모수 수가 다르다.',
              '- 로컬/Colab CPU의 XGBoost는2.1.4로 맞췄다. 그 외 라이브러리 버전 차이는 각 실행 metadata에 기록했다.',
              '- seed 42/43/44 모두 보고하며 최고 seed만 선택하거나 예측을 앙상블하지 않았다.',
              ('- 공식 최고 단일 TCN S=100.002051, 최고 GBM S=100.228038은 별도 참조값이다. '
               '이번 내부 점수로 공식 최고 또는 SOTA 대비 2% 개선을 판정할 수 없다.'),
              '- 기준선 갱신 및 리더보드 제출 없음. 실패한 arm의 설정을 자동 조정·재실행하지 않는다.', '']
    report = '\n'.join(lines)
    (HERE / 'RESULTS.md').write_text(report)
    strategy = HERE / 'strategy.md'
    text = strategy.read_text()
    marker = '\n<!-- VERIFIED_RESULTS -->\n'
    if marker in text:
        text = text.split(marker)[0]
    strategy.write_text(text + marker + report)
    print('Verified results written to RESULTS.md and summary.json')


if __name__ == '__main__':
    main()
