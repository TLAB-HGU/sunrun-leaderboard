# 추론 손풀이 — 자기완결 HTML

제출된 모델의 추론을 **원자료부터 채점까지** 한 단계씩 따라가는 자기완결 HTML입니다.
외부 폰트·라이브러리·네트워크 요청이 없고 파일 하나로 열립니다.

- **detpool 문서형(기본)**: `templates/flow.html` → `e2000fs_calib4_detpool_gpd.html`. 12절 스크롤 문서.
  1~10절은 원자료 → 산식 → 실제 값 순서의 **바텀업 서술**(픽셀 좌표식, logmed/dark 정의, λ(h,v) 계산기,
  E 배열 조립, Tweedie 그래디언트, detroute·gpd 조립, pool의 MSE(w) 2차식과 제약)입니다.
- **앱형(3D, 참고용)**: `templates/app_detpool.html`·`app_mixture.html` → `code/build.py detpool|mixture`
  로 `*.3d.html`. 왼쪽 해설 패널 + 오른쪽 등각 투영 장면이 단계마다 함께 움직입니다(bbycroft.net/llm 참조).
  `code/build.py all`은 이 두 앱형만 빌드하며, 문서형은 `code/build_flow.py`가 만듭니다.
- 직전 버전(바텀업 레이어 없던 9절 흐름 문서)은 `templates/flow.v1.html`에 남아 있습니다.

| 파일 | 모델 | 한 줄 요약 |
| --- | --- | --- |
| [`e2000fs_calib4_detpool_gpd.html`](e2000fs_calib4_detpool_gpd.html) | base `detroute` + arm `calib3_gpd` (이미지 → 30개 피처 → 트리 모델 4개 → 블렌드) | **실제 SUVI 이미지 1장**(2026-04-03 19:00 UTC)이 픽셀 → 12×3 패치 통계 → 경도 줄 → 30개 피처 → 237열 입력이 되고, body·tail·탐지기·gpd가 72시간 예측을 만든 뒤 2,160 시점 창의 잔차로 w를 골라 섞기까지 (스크롤형 12절, 3D 없음) |
| [`e2000fs_calib_tweedie15.html`](e2000fs_calib_tweedie15.html) | 단일 arm: point4 가중 L2 → **Tweedie deviance**(p=1.5, S=500) | 같은 입력·같은 트리·같은 마스크에서 **손실만 교체**하고 짝 control과 비교. 손실 유도(g/h), 300~700 구간 그래디언트·헤시안·deviance 비용, 900라운드 누적 곡선, 밴드별 MSE/deviance, 사전등록 3조건 판정 (스크롤형 13절) |
| [`e2000fs_calib2_mixture2comp.html`](e2000fs_calib2_mixture2comp.html) | level / P(y≥550) / gap 3헤드 × 3블록 | `pred = level + p·(gap − level)` 로 72시간 조립 (F2 단계 부스터 9개) |

## detpool 페이지 12절

| # | 절 | 계산 내용 |
| --- | --- | --- |
| 0 | 한눈에 | 전체 파이프라인 지도, 30개 피처의 출처 비율 |
| 1 | 이미지 | RAD 복원 → 유효 픽셀 → `x=(PC·d)/RSUN` → `lon=atan2(x, z cosB₀ − y sinB₀)` → 12×3 칸. 적도 줄 12칸 실값 표 |
| 2 | 패치 통계 | `logmed=median(log1p(RAD))`, `dark=#(RAD<0.5·R_ref)/#valid`, `cover` — 5픽셀 계산 예시 포함 |
| 3 | 경도 줄 | `τ(v)=147,858,620.7/(3600v)`, `λ=0.55(τ−h−나이)`, 겹침 비율 가중평균(슬라이더, 실값 ✓검증), stale/cover 규칙 |
| 4 | 코로나홀 | `RAD < t·R_ref` (t=0.35/0.45/0.55) → 3×3 opening·closing → 면적 `1/(z·2πR²)` 합, 최소 0.05%, 구역 7개 |
| 5 | 나머지 17개 | ACE 15개 산식표(m6는 관측 행만 평균), rec26/27/28의 h 이동, Enlil 2개 |
| 6 | 30개→237열 | C 21 + E 9 선택, `(N,72,20)→(N,24,9)→(N,216)` reshape 순서, 전방 채움, E 24×9 실값 표 |
| 7 | 모델 | point4 손실의 g/h와 리프 갱신, tail 3헤드 조립, 탐지기 238열·τ=0.234(비용비 1233/5278), 72시간 곡선 |
| 8 | gpd arm | level·P·excess 3헤드, `Z=log(max(y−550,1))`, `P≥0.5 → 550+exp(Z)` |
| 9 | 트리 계산 | 실제 모델의 분기 경로·잎 누적, Tweedie `g=(S/μ)^p(μ−y)` 수치표 |
| 10 | w 선택 | `se_base/se_arm/cross` → A·B·C → `MSE(w)` 2차식 → `w*=(A−C)/(A+B−2C)` → 격자 스냅, 고속 제약, 48/50/2160 안전장치 |
| 11 | 최종·채점 | 저장 파일 3종, F1+F2 = 4,970×72 = 357,840행, detroute 대비 지표표와 목표선 |

### tweedie15 페이지 13절

| # | 절 | 계산 내용 |
| --- | --- | --- |
| 0 | 한눈에 | 통합/F1/F2 성적, 파이프라인 지도 |
| 1 | 대상·정답 | origin t → t+1..72, 144h 성숙 마스크, 4,970×72 셀 |
| 2 | 입력 237열 | 상수 21 + 시간별 9 × 24, 블록 3개 |
| 3 | 모델·레시피 | 파라미터 12개 실제 값, 스테이지 3 × 블록 3 = 모델 18개, 블록별 base_score |
| 4 | point4 편향 | `μ* = Σwy/Σw` 유도, 가중 평균 손계산, **실제 정규화 계수**(14.6% → 1.438, 실효 2.78:0.695), bias 표 |
| 5 | Tweedie 손실 | deviance 정의 `4(√μ−√y)²/√μ`, 코드의 g/h, `S^p` 이유, 리프 보폭 `−(μ−y)/(k+λ/c)` 표 |
| 6 | 그래디언트 수치 | μ=300…700에서 \|g\|·h·deviance 비용, 실제 예측 구간별 \|g\| (Tweedie 6.4→3.1 vs control 23.4→89.8) |
| 7 | 학습 곡선 | losses CSV 실측(로그 y축), 최저 valid 라운드, train/valid 간격 |
| 8 | 900그루 누적 | base_score → 라운드별 누적 예측, 기여 상위 라운드, 451~900 합 +0.49 |
| 9 | 예측 차이·밴드 | (tw−ctl) 히스토그램, 밴드×폴드 MSE/dev/bias, 대표 셀 표, 이긴/진 원점 72h 곡선 |
| 10 | 채점·판정 | internal/F1/F2/통합 표, 3조건 판정(통과·실패·통과), 내부↔공식 부호 반전 |
| 11 | 최종 예측에서 | detroute body · detpool base(A=3,586.30, w=0.25) |
| 12 | 재현·한계 | 산출물 지도, 주장할 수 없는 것 3개, 남은 질문 |

페이지의 **실제 값**은 `data/flow.json`(이미지·패치·strip·코로나홀·30피처·예측 곡선·w 스캔·채점)에서 오고,
`계산 예시`로 표시한 블록(픽셀 5개 중앙값, gap/tail 조립, Tweedie 그래디언트, gpd 조립)만 산식 설명용 숫자입니다.


## 화면 구성 (참조 페이지에서 확인한 값)

| 요소 | 값 |
| --- | --- |
| 배경 / 글자 | `#f3f3f3` / `#000` |
| 분할 | 좌 sidebar `flex:3` : 우 캔버스 `flex:5` (실측 0.60) |
| 타임라인 | 선 `#ccccdf` 5px · 캐럿 `#0c6fdf` 10px |
| 단계 칩 | `#d1d5e7` (높이 30px, radius 2px) → 활성 `#8587f7` / 흰 글자 |
| 버튼 | 1px `#0000004d`, radius 4px, 높이 30px, prev/next 60px |
| 글꼴 | 본문 Roboto 계열 · 제목 세리프 이탤릭 · 숫자 monospace |

## 조작

- 단계 칩 / 타임라인 눈금 / `◀ ▶` / `←` `→` : 단계 이동, `space` : 자동 재생
- 장면 드래그 : 이동, 휠 : 확대, 우하단 패드(← ↑ → ↓ − + ⟲) : 카메라
- **셀 클릭** : 그 칸의 실제 계산이 오른쪽 위 카드로 (detpool은 base/arm/w, mixture는 level/p/gap)

## 계산 패널이 붙는 단계

| 페이지 | 단계 | 변환 |
| --- | --- | --- |
| detpool | 05 | ACE 원자료 6행 → `ace_speed_kms_m6` (평균) |
| detpool | 08 | base·arm·실측 → 잔차² (`(base−y)²`) |
| detpool | 09 | 제곱오차 8셀 → 표본 평균 → a0 |
| detpool | 10 | 6개 상수 → `MSE(w)` 31점 곡선 + 가드 |
| detpool | 11 | `(1−w)·base + w·arm` 셀별 계산 |
| detpool | 12 | 오차² → MSE |
| mixture | 05 | `y → 1[y≥550]` 라벨 |
| mixture | 07 | `w·(level−y)²` (w=4 if y≥550) |
| mixture | 09 | `m + tree → gap` |
| mixture | 10 | `level + p·(gap−level)` 셀별 계산 |
| mixture | 12 | 오차² → MSE |

## 단계 (각 12)

**원자료 이전 (Source)** — 세 단계는 `data/raw_chain.json`의 `upstream`에서 그대로 옵니다.

1. **관측 · 모델** — ACE(L1, SWEPAM·MAG·EPAM·SIS) · GOES-18 SUVI(Fe195) · GONG MRZQS · WSA-Enlil
2. **원본 파일** — CDF(SPDF FTPS) · netCDF 35,848편 112 GB · FITS 5,950장 · DONKI JSON 53개 4,345 실행
3. **수집 · 1시간 격자** — collector → RawDB(`solar_raw.duckdb`) → 시간별 파케이(2022–2026, 5 파티션)

**추론 (Model)**

- **detpool**: 원자료 → 1시간 격자 → 번들 → 두 예측 → 성숙 창 → 6개 상수 → 31점 스캔 → 72개 조립 → 채점
- **mixture2comp**: 원자료 → 정답 72h → 입력 237열 → level → 확률 → gap → 합성 → 72개 출력 → 채점

## 숫자의 출처

페이지의 모든 수치는 `data/*.json`에서 주입되고, 그 JSON은 아래 스크립트가 아카이브 산출물에서 직접
계산합니다. 손으로 옮겨 적은 상수는 없습니다.

```bash
cd ~/sunrun

# tweedie15: 실행 산출물에서 다시 계산 → 인라인
.venv/bin/python docs/explainers/code/extract_tweedie15.py
.venv/bin/python docs/explainers/code/build_tweedie15.py

# 0) 원자료 사슬(두 문서 공통): ACE/SUVI/코로나홀/ENLIL → 특징 → 번들
.venv/bin/python docs/explainers/code/extract_raw_chain.py

# 1) detpool: pool_lane.choose_weights 재실행 → weights.npz와 array_equal 확인 후 추출
.venv/bin/python docs/explainers/code/extract_detpool_gpd.py

# 2) mixture: 3헤드 × 3블록(F2) 재학습 → level/p/gap 노출 (약 10분, 3 프로세스 × nthread 3)
.venv/bin/python docs/explainers/code/refit_mixture2comp.py
.venv/bin/python docs/explainers/code/extract_mixture2comp.py

# 3) 인라인 빌드
.venv/bin/python docs/explainers/code/build.py all
```

검증 영수증은 `data/refit_receipt.json`. 재학습한 3헤드를 합성한 예측은 아카이브
`research/e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz`와 **최대 오차 0.0 km/s**로 일치하고,
SUVI 스트립 특징은 원자료 패치에서 다시 계산해 번들 값과 대조합니다(`data/raw_chain.json`).

## 이미지에서 예측까지 (`e2000fs_calib4_detpool_gpd.html`)

```bash
.venv/bin/python docs/explainers/code/extract_flow.py   # 원본 .nc, 번들, 아카이브 예측, 모델 파일을 읽어 data/flow.json
.venv/bin/python docs/explainers/code/build_flow.py     # templates/flow.html + flow.json -> 단일 HTML
```

`extract_flow.py`는 strip 72시점×5속도, 코로나홀 7개, detroute 라우팅, 최종 블렌드, w 선택을 모델이 실제로 받은
번들·아카이브와 대조하고 하나라도 다르면 중단합니다. 트리 경로와 로짓 누적은 `detector_f1_b*.ubj`에서 직접
꺼낸 값입니다. 멀티아웃풋 트리(body)는 xgboost 2.1.4에서 `get_score`가 죽고 gain을 저장하지 않아, 모델 JSON의
분할 횟수로 입력 출처별 비중을 셉니다. gpd arm 세 머리는 보관돼 있지 않아 최종 곡선만 보여 줍니다.

## 파일

```
docs/explainers/
  e2000fs_calib2_mixture2comp.html    # 산출물 (앱)
  e2000fs_calib4_detpool_gpd.html     # 산출물 (이미지→피처→모델 시각화)
  e2000fs_calib4_detpool_gpd.3d.html  # 이전 3D 단계별 앱 (build.py detpool 로 재생성, 기본 비보관)
  templates/flow.html  code/extract_flow.py  code/build_flow.py  data/flow.json
  app/app.css  app/app.js             # 참조 페이지 스타일 셸 + 등각 3D 장면 엔진
  templates/app_*.html                # 페이지별 장면·단계 정의
  code/*.py                           # 데이터 추출·재학습·빌드
  data/*.json                         # 주입 데이터
  templates/*.html, shared/           # (보관) 이전 문서형 버전 — 기본 빌드에서 제외
```

`data/*.npz`(재학습 산출물, 2.7 MB)는 `code/refit_mixture2comp.py`가 다시 만들 수 있어
`.gitignore`로 제외했습니다.

## 주의

- 설명 도구입니다. 제출물 생성은 각 연구 디렉터리의 러너(`pool_lane.py`, `lanes2.py`)가 담당합니다.
- mixture2comp의 특징 30개는 상위 실험의 permutation importance로 고른 목록이라 fresh holdout이
  아닙니다. 리더보드 점수는 2026-10-31 제출 시점 값입니다.
