# B0 대비 부모(`e2000fs_calib11g_serarounds2`) 코드 변경 기록

기계적 전체 diff: `results/diff-vs-calib11g.diff` (추가 383줄 / 삭제 166줄).
아래는 **의미 있는 변경**만 정리한 것이다. diff 줄 수가 큰 이유는 파이프라인을 재작성했기 때문이며
(재개·저메모리 실행), 모델 레시피는 바뀌지 않았다.

## A. 모델·피처 (변경 없음)

| 항목 | 부모 | B0 |
| --- | --- | --- |
| 피처 수 | 30 (constant 21 + varying 9) | 30 (동일) |
| 블록당 열 수 | 237 | 237 |
| 표적 | level (y − 0) | 동일 (`anchor_tr = zeros`) |
| 가중 | SERA (y−q0)/(q100−q0), 중앙값 앵커 | 동일 |
| 블록 | 3 × 24h | 동일 |
| 하이퍼파라미터 | depth 3, lr 0.03, subsample 0.8, colsample 0.7, mcw 20, λ 5, multi_output_tree, seed 42 | 동일 |

## B. 입력 표 때문에 불가피한 변경 3개

1. **열 이름**: `ch_ch_t035_*`·`ch_ch_t055_*` 7개 → `suvi_t035_*`·`suvi_t055_*`.
   값은 다르다(새 표는 SUVI를 raw NetCDF에서 다시 계산; `analysis/A_oldvsnew/report.md` §2).
2. **masked squared error**: 새 표의 `y`는 status 0만 값이 있고 32.18% NaN이다.
   - grad·hess에 관측 마스크를 곱한다(`keep * w`). 결측 target은 기여하지 않는다.
   - `weights`는 결측 위치에서 NaN이므로 `np.where(mask>0, weights, 0.0)`로 먼저 가린다
     (0 × NaN = NaN이 grad로 새는 것을 막는 부분이며, 첫 구현에서 실제로 발생해 수정한 지점).
   - 중앙값 앵커는 관측된 항목에서만 계산한다(`nanmedian`).
   - 한 블록 24h 안에 관측이 하나도 없는 origin은 학습 행에서 제외(period별 7개).
3. **rounds 재선택**: 부모의 구 표 선택값(300/300/1200)을 복사하지 않고 새 표에서 다시 선택한다.

## C. 실행 구조 변경 (수치에 영향 없는 것)

- long-format DataFrame → 창별 float32 배열 1회 변환(`prepare_arrays`). 동시 fit 메모리를 줄이기 위함.
- 완료된 `(period, rounds, block)` npz를 건너뛰는 재개 로직.
- `--phase observations`: 원문 ACE에서 canonical truth 생성 + 새 표의 `y`와 완전 일치 검사(불일치 0 확인).
- `--phase diagnose`: period별 제곱오차 기여율·lead 구간·손실 집중도.
- `--origin-stride`: 학습 origin 3시간 격자 ablation(표본 효과 측정).
- `phase_weights`는 `--all` 경로에서 호출하지 않는다. 동일 가중을 `sera_weights`로 fit 시점에 계산하며,
  계산식·정규화(중앙값 앵커)는 부모와 같다. `--phase weights`는 단독 실행 시 여전히 사용 가능.

## D. 부모 주석과의 차이

부모 docstring은 "anchor-residual target"을 언급하지만 코드의 `anchor_tr`는 0으로 고정되어 있어
실제 실행은 level 표적이다. B0는 이 사실을 docstring에 그대로 적고 level 표적을 유지했다(코드 동작 동일).
