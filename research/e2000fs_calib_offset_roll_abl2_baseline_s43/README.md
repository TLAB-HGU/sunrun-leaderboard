# e2000fs_calib_offset_roll_abl2_baseline_s43

- 작성자: `seongeun`
- 리더보드 상태: 보드 표시

## 설명 (최신 제출)

Diagnostic ablation of e2000fs_calib_offset_roll: round-2 single-change ablation (8 preregistered axes x seeds 42/43/44). Arm 'baseline' (category baseline), seed 43; config {"seed": 43}. Single preregistered change vs the parent recipe; same fresh-df ffill input and F1/F2/F3 cutoffs; no leaderboard-based selection. Preregistered as not-for-publication, published on user request 2026-11-05. Strategy: research/offset_roll_ablation2/strategy.md

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-11-05 | 99.932 | [`research/offset_roll_ablation2/experiment.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/81397847b260141ec79c5bb9f266176fb51cbb12/research/offset_roll_ablation2_20261105/experiment.py) @ `81397847b` |

## 이 폴더의 파일

- `config.json` — 이 arm이 공용 러너 `research/offset_roll_ablation2/experiment.py --config`에 넘긴 설정(부모 레시피 대비 단일 변경 + seed).

## 참고

- 공용 러너와 사전등록은 [`research/offset_roll_ablation2/`](../offset_roll_ablation2/) (`strategy.md`)에 있다. 여러 arm이 같은 러너를 쓰므로 여기에 코드를 복제하지 않는다.
- 재실행: `python research/offset_roll_ablation2/experiment.py --experiment baseline_s43 --config research/e2000fs_calib_offset_roll_abl2_baseline_s43/config.json --data <df_ffill_trainy.parquet> --observations <competition-observations.parquet> --out runs/e2000fs_calib_offset_roll_abl2_baseline_s43 --device cpu --threads 4` (`CUDA_VISIBLE_DEVICES=''`).

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
