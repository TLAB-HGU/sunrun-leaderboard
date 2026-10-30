"""Generate forecast_strategies_e2000.ipynb (python make_notebook.py forecast_strategies_e2000.ipynb)."""
import nbformat as nbf
import sys

out = sys.argv[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# E2000 계열 다중 horizon 예측 전략 비교 (Direct 제외) — 리더보드 F1 + F2

**목적**: E2000 입력(111 features: ACE·27일 재귀·SUVI strip·CH·WSA-ENLIL)과 frozen XGBoost recipe(900 rounds, depth 3, lr 0.03, seed 42)를 그대로 두고 **multi-step forecasting 전략만** 바꿔, 현재 리더보드 규칙(2026-10-26~)으로 평가합니다.
- **F1**: 타깃 2026-03-01 ~ 06-01 (origin 2,137) — 모델은 **2026-02-26 이전 데이터로만 학습·튜닝**
- **F2**: 타깃 2026-06-01 ~ 09-30 (origin 2,833) — 올해 6~9월
- 순위는 F1+F2 전체 MSE, 그리고 **MSE≥550**(truth ≥ 550 km/s 셀, ffill 포함) 열을 함께 봅니다.

| 전략 | 구현 |
|---|---|
| Recursive | 1-step 공유 모델, 72회 반복 (= RecMO σ=1) |
| DirRec | h별 모델 72개 + 1..h−1 궤적 입력 (= DirRecMO σ=1) |
| MIMO | joint multi-output tree 1개가 72개 동시 출력 (= DirMO σ=72) |
| RecMO(σ) | σ-출력 공유 모델, 72/σ회 반복 |
| DirMO(σ) | 블록별 σ-출력 모델 |
| DirRecMO(σ) | 블록별 모델 + 이전 블록 궤적 입력 |
| Rectify | Recursive base + h별 residual (purged OOF base 예측으로 학습) |
| RectifyMO(σ) | RecMO base + DirMO residual, σ는 select 단계로 선택 |
| Stratify | base·residual ∈ {RecMO, DirMO, DirRecMO} × σ ∈ {6, 24}, 18개 조합 중 select 단계로 선택 |

**단계**
| stage | 학습 타깃 | 용도 |
|---|---|---|
| `select` | < 2025-12-01, 검증 origin 2025-12-04 ~ 2026-02-22 | RectifyMO σ·Stratify 조합 선택 (F1·F2 공통) |
| `f1` | < 2026-02-26 | F1 예측 |
| `final` | < 2026-05-29 | F2 예측 |

**결정사항**: ACE 계열 **2시간 지연 없음**(origin 시각 재계산, lag 2h parity 확인) · 결측 **prefill(ffill)** · 리더보드 점수는 어떤 선택에도 쓰지 않음."""))

cells.append(code("""import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE = Path(os.environ.get('FS_HOME', '/home/t-lab01/sunrun/analysis/forecast_strategies_e2000'))
sys.path.insert(0, str(HERE))
import data, evaluate, run, strategies as S

ROUNDS = int(os.environ.get('FS_ROUNDS', '0')) or None      # None -> frozen 900 rounds
SUBSAMPLE = int(os.environ.get('FS_SUBSAMPLE', '1'))          # smoke runs only
OUT = HERE / os.environ.get('FS_OUT', 'outputs')
SMOKE = ROUNDS is not None or SUBSAMPLE > 1
CFG = S.recipe(n_estimators=ROUNDS)
pd.set_option('display.width', 250, 'display.max_columns', 80)
print('config:', CFG, '| threads shared/block/workers', S.SHARED_THREADS, S.BLOCK_THREADS, S.WORKERS,
      '| subsample', SUBSAMPLE, '| out', OUT)"""))

cells.append(md("## 1. 데이터: lag-0 재계산 + prefill (train · F1 · F2 grid)"))
cells.append(code("""cache = HERE / 'outputs/cache'
if (cache / 'receipt.json').exists():
    train, f1, official, series, receipt = data.load(cache)
else:
    train, f1, official, series, receipt = data.build(cache=cache)
print(json.dumps({k: v for k, v in receipt.items() if k != 'inputs'}, indent=1))
print('train', train.C.shape, '| F1', f1.C.shape, '| F2', official.C.shape)"""))

cells.append(md("## 2. select 단계 (선택 전용, 모든 데이터 < 2026-02-26)\n\n검증 MSE는 관측 행 기준 `val_mse_obs`. RectifyMO σ와 Stratify 조합을 여기서만 고르고, 같은 선택을 F1·F2 모델에 씁니다."))
cells.append(code("""sel = run.Stage('select', train, f1, official, series, OUT, CFG, SUBSAMPLE)
print('fit origins', len(sel.origins), '| validation origins', len(sel.Yv))
NAMES = S.STANDALONE + S.RECTIFY + S.STRATIFY
irows, ihor = {}, {}
for name in NAMES:
    t0 = time.perf_counter()
    pred = sel.run(name)
    irows[name], ihor[name] = evaluate.internal(pred, sel.Yv, sel.val_observed)
    irows[name].update(family=run.family(name), fit_seconds=sel.meta[name]['fit_seconds'])
    print(f'{name:30s} val_mse_obs={irows[name]["val_mse_obs"]:8.1f}  ({time.perf_counter() - t0:7.1f}s)', flush=True)

itable = pd.DataFrame.from_dict(irows, orient='index').sort_values('val_mse_obs')
SIGMA_RMO = run.select(irows, [n for n in S.RECTIFY if n.startswith('RectifyMO')])
STRAT = run.select(irows, S.STRATIFY)
FINAL = S.STANDALONE + ['Rectify', SIGMA_RMO, STRAT]
json.dump({'RectifyMO': SIGMA_RMO, 'Stratify': STRAT, 'canonical': run.canonical(STRAT), 'final': FINAL},
          open(OUT / 'selection.json', 'w'), indent=1)
print('selected RectifyMO:', SIGMA_RMO, '| selected Stratify:', STRAT, '->', run.canonical(STRAT))
itable.round(1)"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(11, 5))
for name in FINAL:
    ax.plot(range(1, 73), ihor[name], label=name, lw=1.2)
ax.set(xlabel='horizon (h)', ylabel='MSE (km/s)$^2$, observed', title='select-stage validation MSE by horizon (2025-12 ~ 2026-02)')
ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.3)
fig.savefig(OUT / 'select_mse_by_horizon.png', dpi=130, bbox_inches='tight'); plt.show()"""))

cells.append(md("## 3. F1 모델(타깃 < 2026-02-26) · F2 모델(타깃 < 2026-05-29) 학습 → F1+F2 결합 예측 357,840행"))
cells.append(code("""st_f1 = run.Stage('f1', train, f1, official, series, OUT, CFG, SUBSAMPLE)
st_f2 = run.Stage('final', train, f1, official, series, OUT, CFG, SUBSAMPLE)
scorer = evaluate.Official()
print('leaderboard store revision', scorer.revision)
rows, hor, preds, files = {}, {}, {}, {}
for name in FINAL:
    t0 = time.perf_counter()
    p1, p2 = st_f1.run(name), st_f2.run(name)
    preds[name] = np.vstack([p1, p2])
    files[name] = run.combined_file(OUT, name, st_f1, st_f2, p1, p2)
    rows[name], hor[name] = scorer.row(files[name])
    n1, n2 = len(st_f1.target_origins), len(st_f2.target_origins)
    rows[name].update(family=run.family(name), val_mse_obs=irows[name]['val_mse_obs'],
                      fit_seconds=st_f1.meta[name]['fit_seconds'] + st_f2.meta[name]['fit_seconds'],
                      inference_seconds_per_fold=(st_f1.meta[name]['inference_seconds_per_fold'] * n1
                                                  + st_f2.meta[name]['inference_seconds_per_fold'] * n2) / (n1 + n2))
    print(f'{name:30s} MSE={rows[name]["mse"]:8.1f} F1={rows[name]["mse_f1"]:8.1f} F2={rows[name]["mse_f2"]:8.1f} '
          f'MSE>=550={rows[name]["mse550"]:9.1f}  ({time.perf_counter() - t0:7.1f}s)', flush=True)

# Reference only: existing E2000 Direct multioutput run (lag 2h inputs, F2 only), same scorer on F2 folds.
for eid, label in (('E1987', 'ref:E1987 Direct multioutput (lag2h, F2 only)'),):
    found = sorted((data.ROOT / f'research/expos/runs/{eid}').glob('predictions_official*.parquet'))
    if found:
        rows[label], hor[label] = scorer.row(found[0])
        rows[label]['family'] = 'reference'"""))

cells.append(md("""## 4. 리더보드 (F1 + F2, 전체 MSE 오름차순)

`mse`: 순위 지표(F1+F2, ffill 포함 357,840행) · `mse_f1` / `mse_f2`: 기간별 · **`mse550`, `mse550_f1`, `mse550_f2`: truth ≥ 550 km/s 셀 MSE** (`n550` 셀 수, `bias550*` = mean(pred − truth)) · `mse550_obs`: 관측 행 중 ≥ 550 · LLH/LHH/HLH/HHH: F2 고난도 regime 4셀 · `event_f1_*`: unique-event F1 · `F2_vs_E1813_%`, `F2_vs_E1789_%`: F2 MSE 기준 상대 변화 · `val_mse_obs`: select 단계 검증(선택 근거) · `inference_seconds_per_fold`: predict 구간만."""))
cells.append(code("""board = evaluate.leaderboard(rows)
board.to_csv(OUT / 'leaderboard.csv')
fmt = {c: '{:.3f}' for c in ('skill_vs_naive', 'event_f1_F1', 'event_f1_F2')}
fmt.update({c: '{:.0f}' for c in ('n550', 'tp_F2', 'fp_F2', 'fn_F2', 'fit_seconds')})
fmt.update({'inference_seconds_per_fold': '{:.5f}'})
fmt.update({c: '{:+.2f}' for c in board.columns if c.endswith('_%')})
fmt.update({c: '{:.1f}' for c in board.columns if c not in fmt and c not in ('family', 'periods')})
with open(OUT / 'leaderboard.md', 'w') as f:
    cols = ['strategy'] + list(board.columns)
    f.write('| ' + ' | '.join(cols) + ' |\\n|' + '---|' * len(cols) + '\\n')
    for name, r in board.iterrows():
        vals = [name] + ['' if pd.isna(r[c]) else (fmt[c].format(r[c]) if c in fmt else str(r[c])) for c in board.columns]
        f.write('| ' + ' | '.join(vals) + ' |\\n')
board.style.format(fmt, na_rep='—').background_gradient(subset=['mse', 'mse550', 'mse550_f1', 'mse550_f2'], cmap='RdYlGn_r')"""))

cells.append(md("## 5. 그림: horizon별 오차 전파, ≥550 구간 bias, 대표 HSS 궤적"))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for name in FINAL + [k for k in hor if k.startswith('ref:')]:
    axes[0].plot(range(1, 73), hor[name], label=name, lw=1.6 if name.startswith('ref:') else 1.1,
                 ls='--' if name.startswith('ref:') else '-')
axes[0].set(xlabel='horizon (h)', ylabel='MSE (km/s)$^2$', title='Leaderboard MSE by horizon (F1+F2; ref = F2)'); axes[0].grid(alpha=.3)
axes[0].legend(ncol=2, fontsize=7)
b = board[~board.index.str.startswith('ref:')].sort_values('mse550')
y = np.arange(len(b))
axes[1].barh(y - .2, b['mse550_f1'], .4, label='F1 ≥550', color='#7a9cc6')
axes[1].barh(y + .2, b['mse550_f2'], .4, label='F2 ≥550', color='#d08c60')
axes[1].set_yticks(y, b.index, fontsize=8)
for i, (v, bias) in enumerate(zip(b['mse550'], b['bias550'])):
    axes[1].text(max(b['mse550_f1'].iloc[i], b['mse550_f2'].iloc[i]), i, f' bias {bias:+.0f}', va='center', fontsize=7)
axes[1].set(xlabel='MSE, truth ≥ 550 km/s', title='High-speed cells by period (bias = overall pred − truth)'); axes[1].legend()
fig.tight_layout(); fig.savefig(OUT / 'leaderboard_horizon_and_550.png', dpi=130, bbox_inches='tight'); plt.show()"""))

cells.append(code("""# Representative HSS onsets: per period, the origin with the largest 72h rise in truth (evaluation-side only).
truth = scorer.truth.assign(origin=lambda d: pd.to_datetime(d.origin_last_input_utc, utc=True))
tw = truth.pivot(index='origin', columns='horizon_hours', values='target_kms')
allo = st_f1.target_origins.append(st_f2.target_origins)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
pick = list(dict.fromkeys(['Recursive', 'DirRec', 'MIMO', SIGMA_RMO, STRAT, board.index[0]]))
for ax, period in zip(axes, (st_f1.target_origins, st_f2.target_origins)):
    rise = tw.loc[period].max(axis=1) - tw.loc[period, 1]
    o = rise.idxmax()
    i = allo.get_loc(o)
    ax.plot(range(1, 73), tw.loc[o].to_numpy(), 'k', lw=2.2, label='truth')
    for name in pick:
        if name in preds:
            ax.plot(range(1, 73), preds[name][i], lw=1.1, label=name)
    ax.axhline(550, color='gray', ls=':', lw=1); ax.set(title=f'origin {o:%Y-%m-%d %H}Z', xlabel='horizon (h)'); ax.grid(alpha=.3)
axes[0].set_ylabel('km/s'); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / 'hss_trajectories.png', dpi=130, bbox_inches='tight'); plt.show()"""))

cells.append(md("## 6. Loss history (round별 train/validation MSE)\n\n모든 fit의 native per-round loss: `outputs/loss_history/<stage>/<strategy>/<model>_s42.csv` (`step,train_mse,validation_mse`). f1·final 단계의 validation은 select 검증 블록(학습 구간과 겹치는 진단용)입니다."))
cells.append(code("""for stage in ('f1', 'final'):
    for name in dict.fromkeys(run.canonical(n) for n in FINAL):
        d = OUT / 'loss_history' / stage / name
        found = sorted(d.glob('*.csv'))
        if not found:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for f in found:
            h = pd.read_csv(f)
            ax.plot(h.step, h.validation_mse, lw=.7, alpha=.7)
        ax.set(title=f'{stage}/{name}: validation MSE per round ({len(found)} fits)', xlabel='round', ylabel='MSE', yscale='log')
        ax.grid(alpha=.3); fig.savefig(d / 'loss_history.png', dpi=110, bbox_inches='tight'); plt.close(fig)
print('loss plots written under', OUT / 'loss_history')"""))

cells.append(md("## 7. 요약"))
cells.append(code("""own = board[~board.index.str.startswith('ref:')]
best, best550 = own['mse'].idxmin(), own['mse550'].idxmin()
print(f'F1+F2 MSE 최저: {best}  mse={own.loc[best, "mse"]:.1f} (F1 {own.loc[best, "mse_f1"]:.1f}, F2 {own.loc[best, "mse_f2"]:.1f}; '
      f'F2 기준 E1789 대비 {own.loc[best, "F2_vs_E1789_%"]:+.2f}%)')
print(f'MSE≥550 최저: {best550}  mse550={own.loc[best550, "mse550"]:.1f} (F1 {own.loc[best550, "mse550_f1"]:.1f}, '
      f'F2 {own.loc[best550, "mse550_f2"]:.1f}), bias550={own.loc[best550, "bias550"]:+.1f}')
print(f'select 단계 선택: RectifyMO={SIGMA_RMO}, Stratify={STRAT} -> 리더보드 순위 {list(own.index).index(STRAT) + 1}/{len(own)}')
print('리더보드 점수는 선택에 쓰지 않았습니다.' + (' [SMOKE RUN]' if SMOKE else ''))"""))

nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3 (forecast_strategies_e2000)', 'language': 'python'}
nbf.write(nb, out)
