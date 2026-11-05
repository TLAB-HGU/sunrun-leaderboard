# e2000fs_blk0prev

- 작성자: `seongeun`
- 리더보드 상태: 숨김(visibility.json)

## 설명 (최신 제출)

Negative result: block-0 (1-24h) uses the t0-12h SUVI strip, blocks 25-72h keep t0. Internal C window: overall 8,184.0 (+5.4% vs parent) but >=550 11,154.3 (-4.2%); official combined 3,600.9 / 6,431.9 = +3.1% / +11.5% vs the parent refit under the same maturity rule (3,494.2 / 5,769.5). The 1-24h block-only gain did not transfer.

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-10-30 | 123.939 | — |

## 이 폴더의 파일

- `replay_script.py` — `52feefa58`에 고정된 원 제출 스크립트 사본. 손상본이다: `main` 미정의, recipe helper 누락, strip 루프가 다섯 채널을 마지막 속도로 덮어씀. 그대로는 실행되지 않는다.

## 참고

- 재현 불가로 fresh-df 재실행에서 제외됨 (`research/rerun_df_ffill/special-blockers.md`, `strategy.md`).

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
