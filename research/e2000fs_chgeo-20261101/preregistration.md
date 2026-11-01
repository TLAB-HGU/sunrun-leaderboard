# e2000fs_chgeo_point4_144h — 사전 등록 (2026-11-01T02:35Z, 결과 확인 전)

근거: `research/notes/lit-review-round7-20261101.md` §1.3(500–600 밴드 판별력), §1.5(미사용 CH 기하 열), §4.1.

## 비교 단위

| 항목 | 값 |
|---|---|
| 부모 레시피 | `e2000fs_calib_point4_144h` — 3블록 × 24출력 벡터트리, Top30 237열, 900라운드, depth 3, eta .03, subsample .8, colsample .7, min_child_weight 20, reg_lambda 5, seed 42 |
| 목적함수 | point-weighted squared error (`y>=550` 가중 4, horizon별 평균 1) — 부모와 동일 |
| 마스크 | `origin+144h <= cutoff` (F1 2025-12-29 / F2 2026-03-29 / F3 2026-06-29) |
| 입력 | `data-v2/model-inputs/leaderboard-store/input-df/df.parquet` 사본, sha256 `e1a4cf61…c09d` |
| 후보 | control + CH 기하 22열 (아래) |
| 유일한 차이 | 후보에만 기하 블록 추가 (259열 vs 237열) |

기하 22열 (임계 t035·t055 각각): `top1_area`, `top1_lat`, `top1_lon`, `top1_lat_width`, `top1_lon_width`,
`share = top1_area/area`, `A_lat0`, `A_lat1`, `A_lat2`(위도대 면적합), `A_asym = (A_lat2−A_lat0)/area`,
`d27_share = share(t) − share(t−27 d)`. 값은 `slot <= origin − 1h`인 최신 슬롯(fill: hourly ffill).
물리 기하량만 사용하며 임계·가중·계수를 새로 정하지 않는다. t035/t055는 기존 피처가 이미 쓰는 임계다.

## 판정 (성공 선언용, 제출 게이트 아님)

1. F1·F2·F3 **각각**에서 `≥550 MSE`가 control 이하.
2. 전체 MSE가 control 대비 +1% 이내.
3. origin 단위 paired block bootstrap 4,000회에서 pooled `≥550` 차이의 95% 구간이 0 아래.
4. 판별력: 예측 500–600 밴드에서 `P(y≥550)` 분리도(AUC)가 control보다 개선.
5. 내부창 수치는 성공 근거로 쓰지 않는다(4·5차에서 부호 반전 확인). 공식 fold 방향만 인정.

부호가 fold별로 뒤집히거나 3번 구간이 0을 포함하면 **기각**으로 기록하고, 그 경우 CH 정보는
이 문제에서 판별력이 아니다(§4.1 리스크, A6 선례)로 결론한다.

## 산출물

- `runs/v1/predictions_control.parquet`, `runs/v1/predictions_chgeo.parquet` (각 454,680행)
- `runs/v1/summary.json`, `compare.py` → `runs/v1/comparison.json`
- 제출: `meta_control.yaml`, `meta_chgeo.yaml` (양쪽 모두 F1/F2/F3 무조건 산출·제출)
