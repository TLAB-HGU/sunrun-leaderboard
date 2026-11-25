# data-v2: DB 레이어

소스 원문(`raw/`)과, 그 원문만으로 만든 모델 입력 `input/df.parquet`를 둡니다.

## 레이어 규칙

```
data-v2/  (DB)   raw/<소스> -> notebooks/raw_to_input_df.ipynb    -> input/df.parquet
                              -> notebooks/raw_to_input_df_ts.ipynb -> input/df_ts.parquet
research/        df.parquet    -> {row}/experiment.py (+ _module/gbm.py) -> result/{row}/results/*_predictions.parquet
                 df_ts.parquet -> {row}/experiment.py (+ _module/seq.py) -> result/{row}/results/*_predictions.parquet
```

- 의존 방향은 한쪽뿐입니다. research 코드는 `df.parquet`(GBM 행)나 `df_ts.parquet`(시계열 DL 행)만 읽고, `data-v2/`는 `research/`를 import하거나 읽지 않습니다.
- 최상위는 데이터 종류이고 그 아래가 소스입니다(`raw/<소스>/`). 소스 전체를 합친 모델 입력 `input/`은 최상위에 둡니다.
  - `raw/`: 웹에서 받은 원문 파일. `fetch.py`만 씁니다.
  - `input/`: `notebooks/raw_to_input_df.ipynb`(`df.parquet`, `MANIFEST.json`)와 `notebooks/raw_to_input_df_ts.ipynb`(`df_ts.parquet`, `MANIFEST_ts.json`)만 씁니다.
- 중간 가공 폴더(`processed/`)는 없습니다(2026-11-02 사용자 삭제). 노트북은 실행할 때마다 raw에서 다시 계산하고, 캐시를 남기지 않습니다.
- 해당 내용이 없는 폴더는 만들지 않습니다.

## 구성

| 경로 | 내용 |
| --- | --- |
| `fetch.py` | raw 신규분 insert 도구(아래 "raw 갱신" 참고). `raw/`만 씁니다 |
| `raw/ACE/monthly-txt/` | NASA ACE 월별 원문: `{YYYYMM}_ace_{swepam,mag,epam,sis,loc}_1h.txt`, 2022-08부터. 출처 https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/ |
| `raw/SUVI/nc/g18/YYYY/DOY/` | `fetch.py`가 받은 GOES-18 SUVI Fe195 L1b NetCDF. 시간 H당 1개이고 파일의 `obs_end`가 `[H-1h, H)` 안에 있습니다. **이 파일이 프레임 선택의 유일한 기록입니다** |
| `raw/ENLIL/donki/` | WSA-Enlil 실행 JSON(월별 파일, 2022-07부터). 출처 https://ccmc.gsfc.nasa.gov/DONKI/WS/get/WSAEnlilSimulations |
| `raw/DONKI/notifications/` | DONKI CME 알림(발행 월별 JSON, 2022-01부터). 출처 https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications |
| `raw/NOAA/srs/`, `raw/NOAA/discussion/{txt,hdr}/` | NOAA Solar Region Summary(2022-09-25부터)와 SWPC forecast discussion 원문·응답 헤더(2024-03부터). 출처 https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/ |
| `raw/ACE/daily-txt/swepam/` | ACE SWEPAM 1분 일별 원문(2022-08-01부터, `:Created:`가 공개 시각). 출처 https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/ |
| `raw/ENLIL/wsa-s3/` | NOAA WSA-Enlil open-data 버킷의 `evo.earth.nc`와 `listing.json`(LastModified). 출처 https://noaa-wsa-enlil-pds.s3.amazonaws.com |
| `notebooks/raw_to_input_df.ipynb` | raw → `df.parquet`. 모든 로직이 노트북 안에 있습니다(아래 "df.parquet") |
| `input/df.parquet` | 모델 입력. (`set`, `origin`, `h`) 긴 형식, `MANIFEST.json`에 행·열·규칙이 있습니다. 데이터라 git에 올리지 않고, 2026-11-25 빌드(466열, NCEI 제거판)는 GitHub Release [`data-v2-df-20261125`](https://github.com/TLAB-HGU/sunrun-leaderboard/releases/tag/data-v2-df-20261125)에 `MANIFEST.json`과 함께 첨부했습니다 |
| `notebooks/raw_to_input_df_ts.ipynb` | raw → `df_ts.parquet`. 모든 로직이 노트북 안에 있습니다(아래 "df_ts.parquet") |
| `input/df_ts.parquet` | 시계열 모델 입력. 매시 칸 1행, `MANIFEST_ts.json`에 행·열·규칙이 있습니다. 데이터라 git에 올리지 않습니다 |

## df.parquet

```bash
cd /home/t-lab01/sunrun
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 data-v2/notebooks/raw_to_input_df.ipynb
# DF_SMOKE=1 DF_OUT=<dir>: 짧은 구간만 <dir>에 씀 / DF_WORKERS=N: SUVI 추출 worker 수 (기본 12)
```

전체 실행은 SUVI NetCDF 33,842개를 모두 읽으므로 1시간 안팎 걸립니다. 길게 돌릴 때는 `systemd-run --user`로 띄웁니다.

- **형식**
  - 한 행은 (`set`, `origin`, `h`)이고 origin 하나가 h = 1…72의 72행입니다.
  - 메타 열: `set, origin, h, target_time, period, fit_F1, fit_F2, fit_F3, y`.
  - `set = train`: origin 2022-10-01T00Z ~ 2026-06-23T00Z, 1h 간격.
  - `set = query`: origin 2025-12-31T23Z ~ 2026-09-26T23Z, 1h 간격, 6,457개. `period`는 F1/F2/F3입니다.
  - `fit_P = (set == train) & (origin + 144h ≤ cutoff_P)`이고, cutoff는 F1 2025-12-29, F2 2026-03-29, F3 2026-06-29입니다.
- **처리하지 않는 것**: 입수지연, 결측, 이상치. 앞값 채움, 이전 프레임으로 되돌아가기, stale/cover 마스크, clip·log 변환이 없고 피처만 계산합니다.
- **시간 칸**
  - 모든 소스의 시각 t는 칸 `floor(t, 1h) + 1h`에 들어갑니다([H-1h, H) → H). 19:49·19:59·19:00은 모두 20시 칸입니다.
  - 한 칸에 여러 개면 가장 늦은 것을 씁니다. origin O는 칸 ≤ O만 씁니다.
  - ACE 1h 행은 라벨 시각 + 1h 칸, SUVI는 `obs_end`의 칸, ENLIL은 `modelCompletionTime`의 칸입니다.
- **`y`**: target_time T 라벨의 SWEPAM 속도입니다. 시각을 이동하지 않으며, `status == 0`이고 유한한 양수일 때만 값이 있고 나머지는 NaN입니다(공식 채점과 같음).
- **피처**: raw에서 나오는 전체입니다. 열 이름에 `ch`를 쓰지 않습니다.
  - ACE 22개 변수의 `_last`/`_m6`/`_m24`. `status ≠ 0`이거나 결측 표시값이면 NaN입니다.
  - 속도 파생 `v_lag*`, `v_m*`, `v_sd*`, `speed_d24`, `epam_p1_d6`.
  - SUVI: `img_age_h`, 셀 통계 148열 `suvi_disk_*`/`suvi_lonII_latJ_*`, 어두운 영역 면적 180열 `suvi_t035_*`/`suvi_t045_*`/`suvi_t055_*`, h별 strip 15열 `strip_v*`.
  - ENLIL 5열과 h별 `enlil_dt_target`.
  - 재귀 `rec26/27/28`, `rec_mean`.
  - **추가 20열(6b절, 2026-11-25)**: `e2000fs_…_union9_lbsel_plus7` 행의 입력 중 446열 밖에 있던 열입니다. DONKI 알림 8열 `pub_*`, ACE 파생 3열, SRS·예보 토의 2열, SUVI 파생 2열, `cme_phase_rec27`, `rec27_ramp`, `rec54_consensus`, `enlil_earth_speed`, `onset700_hazard`. 정의는 노트북 6b절 표에 있습니다.
  - 이 20열은 원 정의를 그대로 따르므로 위의 "처리하지 않는 것"의 예외입니다. 공개 시각이 origin 이전인 원문만 쓰고, 결측은 `-999`이며, `ace_proton_ram_pressure_npa`는 h별 앞값 채움 뒤의 n, v로 계산합니다. `onset700_hazard`는 2025-09-23 이전 학습 y로 적합한 보조 XGBoost(threads 1)의 출력입니다.
  - `enlil_earth_speed`는 NOAA open-data S3(`noaa-wsa-enlil-pds`) 실행만 씁니다. NCEI archive(개당 약 145 MB, 공개가 실행 후 약 25 h)는 실시간 예측에서 같은 시점에 받을 수 없어 2026-11-25에 원천에서 뺐습니다. 그 결과 값이 있는 칸이 F1 78.7 %→37.7 %, F2 90.0 %→81.4 %, F3 98.8 %→72.1 %로 줄었고(나머지는 더 이른 S3 실행 값 또는 `-999`), 다른 465열은 그대로입니다.
  - 지금 `df.parquet`(466열)는 2026-11-02 빌드의 446열에 6b절 셀로 20열을 더한 것입니다. 이전 파일: NCEI 포함판 `input/*.prev-ncei`(frozen plus7 입력 표와 전 열·전 행이 같음), 446열판 `input/*.prev-20261125`. 노트북 전체 연결은 `DF_SMOKE=1`로 확인했습니다.
- **이전 df(2026-11-01)와의 관계**: 이전 df는 frozen 번들에서 만든 30개 피처 표였고, 새 df로 덮어썼습니다. 그래서 `research/e2000fs_calib_point4_144h/experiment.py`(열 이름 `ch_ch_*`, 이전 값으로 공식 예측 재현)는 새 df로는 그대로 실행되지 않습니다.

## df_ts.parquet

```bash
cd /home/t-lab01/sunrun
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 data-v2/notebooks/raw_to_input_df_ts.ipynb
# DF_SMOKE=1 DF_OUT=<dir>: 짧은 구간만 <dir>에 씀 / DF_WORKERS=N: SUVI 추출 worker 수 (기본 12)
```

시계열 DL 행(`vanila_gru`, `vanila_itransformer`, `vanila_tsmixer`)의 입력입니다(2026-11-17). `df.parquet`와 같은 raw·같은 시간 칸 규칙·같은 `y` 정의를 쓰고, 형식만 다릅니다.

- **형식**: 한 행은 매시 칸 `time`입니다. 첫 ACE 칸(2022-08-01T01Z)부터 마지막 query origin + 72h까지 빈칸 없이 이어집니다.
- **분할 열**: `train_origin`, `query_origin`, `period`, `fit_F1/F2/F3`. 정의는 `df.parquet`의 `set`/`period`/`fit_P`와 같습니다.
- **`y`**: 행 `time` = T 라벨의 SWEPAM 속도(status 0, 채움 없음). origin O의 리드 h 정답은 행 O + h의 `y`입니다.
- **ACE 22열** `ace_*`: 칸 값(라벨 T−1h 행)이고 `df.parquet`의 `_last`와 같은 값입니다. `_m6`/`_m24`·lag 같은 요약은 없습니다(모델이 시계열로 봅니다).
- **SUVI**: `img_age_h`, `suvi_disk_log_median`, 그리고 칸 T 프레임 한 장의 **경도 stripe 프로파일** `suvi_strip_{logmed,dark,cover}_k{m072…p143}` 648열. 전체 표는 36,503행 × 1,012열이고, 전체 빌드는 worker 20개로 약 1시간 50분 걸립니다.
  - stripe k의 영역은 |경도 − 0.55°/h × (k − `img_age_h`)| ≤ 5°, |위도| ≤ 30°입니다. 서쪽이 +이고, k는 "칸 시각보다 k시간 전에 중앙 자오선을 지난 경도"입니다(k < 0은 앞으로 지날 경도).
  - `logmed`는 유효 픽셀 log1p radiance 중앙값, `dark`는 디스크 중앙값 0.5배 미만 비율, `cover`는 유효 픽셀 비율입니다. `df.parquet`의 `strip_v*`(15° 셀 겹침 가중 평균, h·속도별)와 달리 픽셀에서 직접 계산한 프레임 고유 값입니다.
- **ENLIL 5열**: `df.parquet`와 같은 정의를 매시 칸에 적용한 값입니다.
- **프레임 요약 327열**(2026-11-17 추가): 칸 T 프레임의 셀 통계 147열 `suvi_disk_*`/`suvi_lonII_latJ_*`와 어두운 영역 면적 180열 `suvi_t035/t045/t055_*`. `df.parquet`와 같은 코드·같은 이름이고, 둘 다 프레임이 있는 칸에서 값이 같습니다. h별 `strip_v*`는 `lat1` 셀 통계와 `img_age_h`의 함수라 표에 넣지 않고 `research/_module/summary.py`가 계산합니다.
- **처리하지 않는 것**: `df.parquet`와 같습니다. 앞값 채움, 결측 마스크, 표준화, 프레임 되돌아가기는 `research/_module/seq.py`가 합니다.
- 노트북의 검증 셀이 같은 origin의 ACE·`img_age_h`·ENLIL·fit 마스크·`y`가 `df.parquet`와 일치하는지 확인합니다.

## 원문과 정답에 관한 주의

- SUVI 픽셀 원본은 S3 `noaa-goes18`의 `SUVI-L1b-Fe195/`에서 받습니다. 선택한 `.fits` 프레임과 이름이 같은 `.nc` 객체입니다. SUVI는 GOES-18 Fe195만 쓰고, 2022-08-04~08-21은 S3에 `.fits`가 없어 프레임을 고를 수 없습니다.
- SWEPAM `status == 0`은 전체 시간의 약 67%입니다. 나머지(status 1~8, 9)는 `y`와 ACE 피처에서 NaN입니다. EPAM anisotropy index는 원문 전체가 결측값(-1.00)입니다.
- 원문은 받은 시점의 최신 값입니다. 각 시점에 실제로 공개돼 있던 값인지는 검증되지 않았습니다.
- `metric/metric.ipynb`(공식 채점 레퍼런스)는 NASA SWEPAM 월별 txt를 `raw/ACE/monthly-txt/`에 캐시하고 결과를 `metric/`에 씁니다(둘 다 git 제외, Run All로 재생성).

## raw 갱신

```bash
/home/t-lab01/sunrun/.venv/bin/python data-v2/fetch.py [SOURCE ...] [--full] [--dry-run] [--workers N]
# SOURCE: ace enlil suvi donki srs discussion ace-daily wsa (기본: 전부)
```

수동 실행 전용이고 타이머는 없습니다. **insert-only**입니다. 없는 기록만 추가하고, 이미 저장한 기록은 웹 값이 달라져도 바꾸지 않으며 이력도 남기지 않습니다.

| 소스 | 대상 | 추가 단위 |
| --- | --- | --- |
| ACE | 로컬 마지막 달 − 1 ~ 이번 달 (`--full`: 2022-08부터) | 시간 행. 새 파일은 원문 그대로, 기존 파일은 없는 행만 추가 |
| ENLIL | 로컬 마지막 달 − 1 ~ 이번 달 (`--full`: 2022-07부터) | `simulationID`. 새 달은 원문 그대로, 기존 달은 배열 끝에 추가 |
| SUVI | 2022-08-22 21Z 이후 닫힌 시간 중 로컬 NetCDF가 없는 시간 | NetCDF 1개 |
| donki | 로컬 마지막 달 − 1 ~ 이번 달 (`--full`: 2022-01부터) | `messageID`. 기존 달은 배열 끝에 추가 |
| srs | 2022-09-25 ~ 어제 중 로컬에 없는 날 | 파일 |
| discussion | 2024-03 ~ 이번 달 목록에서 로컬에 없는 파일 | 원문 + 응답 헤더 |
| ace-daily | 2022-08-01 ~ 그저께(다음 날이 닫힌 날만) 중 로컬에 없는 날 | 파일 |
| wsa | 버킷 전체 목록, 로컬에 없는 `evo.earth.nc` | key(LastModified는 처음 본 값 유지) |

- SUVI 프레임 선택: G18 Fe195의 `[H-1h, H)` 프레임 중 `obs_end + 2s <= H`인 것에서 일식 없는 프레임 > 긴 노출(0.1초 단위) > 최신 순입니다. 후보가 없거나 S3에 `.nc`가 없으면 아무것도 저장하지 않습니다. 다른 위성으로 대체하지 않습니다.
- 지금 있는 33,842개(2022-11-22 22시 칸 ~ 2026-11-01)는 2026-11-01~02에 S3 업로드 시각 조건을 더 걸어서 받은 것입니다. 이 조건은 2026-11-02에 없앴고, insert-only라 기존 파일은 그대로 둡니다. 다음 `fetch.py suvi` 실행은 파일이 없는 시간(2022-08-22~11-22, 그 뒤 약 720시간)을 새 조건으로 다시 확인하므로 일부가 채워질 수 있습니다.
- 파일이 없는 시간은 네트워크 오류, 후보 없음과 상관없이 실행할 때마다 다시 확인합니다.
- 많은 시간을 한 번에 받을 때는 2024년 5월 이전 시간은 `suvi --workers 12`, 그 뒤 시간은 `--workers 18`이 적당합니다.
  - 시간당 FITS 헤더를 약 59개 읽어야 해서 워커가 적으면 느립니다(워커 2개는 분당 4개).
  - NetCDF는 2024년 5월 전까지 약 5MB, 그 뒤로 약 1.9MB입니다.
  - 회선은 3~4MB/s에서 막혀 있고, 이를 넘기면 이 컴퓨터의 DNS까지 실패하니 더 늘리지 마세요(2026-11-01 측정).
- 진행 중인 달의 ACE 행은 처음 받은 값으로 고정됩니다. `loc` 파일에는 아직 오지 않은 시간의 예측 위치도 들어 있습니다.
- `metric/metric.ipynb`는 예외입니다. Run All 하면 `raw/ACE/monthly-txt/`의 SWEPAM 최신 월 파일을 통째로 다시 받아 덮어씁니다.
- `input/df.parquet`는 갱신하지 않습니다. raw를 추가한 뒤 노트북을 다시 실행하세요.

## 삭제한 레거시

- 2026-11-01: point4가 쓰지 않는 데이터, 수집 체계(RawDB, collectors, 타이머), `raw/SUVI/slots/`, `bundles/`, `builders/`. 목록은 `docs/cleanup-20261101.md`에 있습니다. 재빌드 코드는 git 기록(`1358d5d64`, `9637c1192^`)에 있습니다.
- 2026-11-02: `processed/`(사용자 삭제), `build_df.py`(삭제된 builder를 import하던 조립 코드, 노트북으로 대체).
- `docs/explainers/code/extract_raw_chain.py`는 삭제된 슬롯 표를 읽으므로 다시 실행하면 실패합니다.
- GPU lease 잠금 파일은 `data-v2/`가 아니라 `leaderboard/store/ch-breakthrough-v2/gpu-leases/`에 있습니다(`leaderboard/expos/core.py`).
