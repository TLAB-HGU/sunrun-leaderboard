# pane11 ACK — cumulative worker contract (2026-10-24)

## 읽음 확인 (실제 파일, 2026-10-24)
| 문서 | bytes | sha256(앞16) |
|---|---:|---|
| `/home/t-lab01/sunrun/AGENTS.md` (compact) | 9,607 | `03ffed49ee46aebe` |
| `/home/t-lab01/sunrun/research/CUMULATIVE-WORKER-CONTRACT.md` | 6,627 | `9397d582949fea81` |
| `/home/t-lab01/sunrun/leaderboard/experiments/cumulative_20261024/README.md` (canonical, 1–98) | 12,150 | `7e44583363b14972` |
| `/home/t-lab01/sunrun/research/gjc-worker` | 280 | `ebf36c651240b00c` |

## 역할 / 소유권
- 이 pane = **lane11 (HI lane)**. 자기 전용 파일만 수정:
  - `research/cumulative-worker-acks-20261024/pane11.md`
  - `research/hi-lane-20261024/` (HI 감사: `AUDIT.md`, `availability_sample.json`, `preserved.json`)
  - `research/breakthrough-20261024/hi_full_run/` (기존 진단 산출물, 보존)
- 수정 금지: `leaderboard/experiments/cumulative_20261024/`(Codex 전용), root `AGENTS.md`,
  `research/worker-prompt.md`, `research/CUMULATIVE-WORKER-CONTRACT.md`, `expos/`, `scorer/`,
  등록된/shared runner, 다른 lane 코드. 패키지/root prompt 수정 없음.

## 금지 경계 (이해)
- 최종 성공 기준은 **전체 leaderboard 기간 총 MSE ≤ comparable champion × 0.95** 하나.
  OOF/dev/selection/audit/event/peak 개선은 진단이며 성과가 아니다.
- guard 실패를 old runner/ad-hoc 공식 run으로 우회하지 않는다. 새 후보·새 공식평가·공식점수
  기반 튜닝은 새 계약 전에 시작하지 않는다.
- 미입증 HI source를 champion 입력에 넣지 않는다. 후속 HI 편입은 같은 새 guard +
  별도 명시된 campaign rebase가 필요하다.
- CPU-only/no-DL, admitted SW-ID only, ACE obs+2h, no downloads, no evaluator 변경,
  no upload/submission.

## 진행 상태
1. **ACK 완료** (이 파일).
2. **HI 감사 완료** (가역적, read-only):
   - raw source/availability: SW-030/031 16,465 frames, official window HI1 1,113 + HI2 1,089,
     official-window 20/20 byte+SHA256+available_at 검증.
   - full official-window audit (`verify_availability.py`, 12 tests, 2026-10-24):
     HI1 **1,991 verified / 842 verified_stale / 0 failed / 0 unknown**;
     HI2 **2,004 / 829 / 0 / 0**. 1,538 HI1 / 1,484 HI2 origins have a fresher
     frame published after the origin (publication delay median 0.3/0.8 h,
     p90 ~205 h). Canonical source receipt booleans were **not** modified; the
     audit is a separate row-level record for the leader's source audit.
   - full-manifest `--verify-all`: HI1 **8,004/8,004**, HI2 **8,461/8,461**
     byte+SHA256 match; 0 missing / 0 mismatch / 0 unparseable (16,465/16,465).
   - ACE/SUVI source provenance (`verify_source_provenance.py`, 24 tests):
     sidecar 3 file entries pass + 1 raw-ace-snapshot 2026 copy fail + 3
     directory entries unknown (directory SHA recipe absent); ACE
     13,521/13,521 pass the +2h rule (234 available-at pass,
     5,470 daily-list late, 7,817 2h-proxy-only); SUVI 13,047/13,521 pass,
     0 fail, 474 unknown (424 2022 no-slot, 50 `nc_size` missing). Handoff:
     `research/hi-lane-20261024/HANDOFF_SOURCE_PROVENANCE.md`.
   - 82/1 diagnosis (`diagnose_82_1.py`, prereg frozen): true-track median SNR
     p50 0.040; event−N1 +0.036, event−N2 −0.008; positive control A=3 → 51%
     detection. This is a **signal-weakness/representation** result, not a
     detector-sensitivity or background-removal bug.
   - Earth-related signal: HI2 J-map 82 covered / 1 detected (1.2%), INFO-TEST not-identifiable,
     lane11 TBD 진단 base −0.88% / champion OOF −0.13%.
   - 양성대조/poison: prereg frozen, real-source pass=false는 scientific negative,
     E1517 poison PASS, lane11 runner poison PASS.
3. **통합하지 않음**: champion 입력/HI feature, 새 공식평가, lb-eval 없음.
4. **과학적 상태**: 전체 기간 총 MSE 기준 5% 개선은 **아직 없음**. 연구 판정 변경 없음.

다음: 새 계약/guard 전달 시까지 위 감사 기록을 보존하고, 지정된 가역적 준비 범위를 벗어나지 않는다.
