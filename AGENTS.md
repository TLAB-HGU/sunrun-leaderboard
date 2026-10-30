# Repository Guidelines

## Project Structure & Module Organization

This workspace supports solar-wind forecasting, data collection, and reproducible research.

- `research/`: experiment records, runners, and verification receipts. Read its `AGENTS.md` and `CUMULATIVE-WORKER-CONTRACT.md` before experiments.
- `analysis/`: topic-specific studies containing scripts, tests, reports, and figures.
- `data-v2/`: the only data root — raw RawDB and pinned snapshots under `raw-store/`, model-input store under `model-inputs/leaderboard-store/`, plus feature builders, manifests and tests. There is no `store/` path any more (removed 2026-10-30); use `paths.STORE_ROOT`/`paths.STORE`/`paths.RAW_STORE`.
- `nrt-collect/`: near-real-time collection and source availability evidence.
- `leaderboard/`: evaluation board, status poster and submission tooling; merged into this repository as a subtree (commit `bfd7c9cc2`, 2026-10-30), so it is versioned here. `lb-f1f2/`, `lb-maint/` and `model-release-publish/` code is tracked too; their `docs/`, `releases/`, `experiments/` and caches stay local.
- `leaderboard/model-release-publish/`: submission tools, reproducible model scripts, and release tests.

The root `.gitignore` excludes most workspace directories. Check repository ownership and `git status` before staging; do not force-add datasets or model artifacts.

### Git topology (single repository)

This workspace is one Git repository at `/home/t-lab01/sunrun`. The former standalone `leaderboard/` clone (remote `TLAB-HGU/sunrun-leaderboard`) was merged on 2026-10-30: its history is the second parent of `bfd7c9cc2`, its branches are preserved as `refs/remotes/lb/*`, and every path is now tracked at `leaderboard/`.

- Remote `github` is `TLAB-HGU/sunrun-leaderboard`, a public repository. Local history holds blobs over GitHub's 100 MB limit, so `master` is never pushed directly: its `main` carries snapshot commits made with `git commit-tree` on top of `github/main`, with the same tree as `master` minus `research/policy-archive`. No full-history backup bundle exists any more; `.git` here is the only full copy.
- Publication pipelines still clone `origin/main` in their own checkouts and push their release there, at the repository root. Reconcile deliberately with the monorepo layout before running one.
- The experiment catalog those pipelines link into is local (`leaderboard/experiments`, untracked); the `leaderboard/scripts` and `leaderboard/releases` entries that pointed at the missing `/home/t-lab01/sunrun/experiments` catalog were dropped in `2e72b60b5`/`5c74f5c58` and remain readable in earlier commits.
- `research/chatgpt-skill/solar-queue-harness/` is still a separate clone inside an ignored path; it is not part of this repository.

- Prefill to empty data


## Build, Test, and Development Commands

There is no workspace-wide build command. Use the relevant component’s README and the shared uv environment below.

### Shared Python environment

- Use `/home/t-lab01/sunrun/.venv` for all Python work in this workspace, including subprojects, tests, and experiments. This is the user-selected uv environment (Python 3.12).
- Activate with `source /home/t-lab01/sunrun/.venv/bin/activate`, or invoke `/home/t-lab01/sunrun/.venv/bin/python` explicitly. Prefer the absolute interpreter path in tool calls and background jobs, where activation may not persist.
- Install dependencies with `uv pip install --python /home/t-lab01/sunrun/.venv/bin/python <packages>` (or `-r <requirements.txt>`). Do not create or select another virtual environment for new work.
- `wandb` is installed in this shared environment. Install additional task dependencies here as needed.

- `python3 -m pytest analysis/peak72_eda/`: run focused EDA tests from the workspace root.
- `python3 -m pytest data-v2/tests/`: run data infrastructure tests with component dependencies installed.
- `uv pip install --python /home/t-lab01/sunrun/.venv/bin/python -r leaderboard/model-release-publish/requirements.txt`: install submission-client dependencies in the shared environment.
- From `leaderboard/model-release-publish/`, run `python3 submit.py my_predictions.parquet meta.yaml --dry-run` to validate and score without uploading. Private evaluation data access is required.

## Input Timing (t0 rule)

For an origin t0 (`origin_last_input_utc`), a model may use every input row with `timestamp_utc <= t0`, as the leaderboard rule states. Per user instruction 2026-10-29, the former ACE +2h availability delay is removed: new experiments, features and runners apply no 2h input shift, and training-label maturity is target + 72h (not 74h). Runners, frozen inputs (e.g. the E0472/E1149/cumulative base111 parquets) and results made under the old rule keep their original timing as historical records; label them ACE+2h when comparing them with new runs, and do not copy their 2h shift into new code.

## Leaderboard Experiment Status (required, Codex and Claude Code)

Post every solar-wind forecasting experiment to the leaderboard Space (https://huggingface.co/spaces/tlabtlab/sunrun-leaderboard) before it runs, whatever the launcher (expos, `launch.py`, analysis `run.py`, notebooks). `leaderboard/lbstatus.py` adds each 대기 / 진행 experiment as a row at the top of the ranking table, shown as `<experiment> (진행)` or `<experiment> (대기)` with dashes for the metrics. A finished experiment leaves those rows by entering the ranking: submit its result to the leaderboard, and its entry is retired (완료) once the dataset holds a result with the same `team_member`/`experiment` submitted after the entry was registered. Per user instruction 2026-10-29, this replaces the no-upload rule for leaderboard results. Unit tests and smoke checks need no entry.

- When an experiment is planned, register it as 대기: `/home/t-lab01/sunrun/.venv/bin/python /home/t-lab01/sunrun/leaderboard/lbstatus.py set <experiment> --status 대기 --desc "<what it tests>"`. Name it exactly as its result will be submitted (`experiment` in meta.yaml, `[A-Za-z0-9._-]`); a run that submits several experiment names gets one entry per name. `--member` defaults to `seongeun` (env `SUNRUN_MEMBER`); the agent (codex / claude-code) is detected automatically.
- Launch through the wrapper: `lbstatus.py run <experiment> -- <command ...>`. It posts 진행 at start and puts the rows and the best/second-best marks back every 5 minutes while the run lasts. On rc 0 the entry stays 진행 ("awaiting leaderboard submission"); on failure it posts 실패, on SIGTERM/SIGINT 중단.
- After a successful run, submit the result (`leaderboard/model-release-publish/submit.py` or the study's publication pipeline), then run `lbstatus.py sync`. Publishing rewrites `index.html` and drops the status rows and marks. `sync` retires the submitted entry and restores them.
- A run that cannot use the wrapper: `lbstatus.py set <experiment> --status 진행` right before it starts, then submit and `sync` as above.
- No internal gate may stop official evaluation (user instruction 2026-10-29). A study may compute internal or development checks, such as seasonal folds, a confirmation window or a parent comparison, and record them. It must not use them to skip the F1/F2 fits, the official forecasts or the leaderboard submission. Every experiment whose training completes fits F1 (targets < 2026-02-26) and F2 (targets < 2026-05-29), writes the 357,840-row official forecast, submits it and runs `sync`. Record the internal result in the description (e.g. "internal check failed; not promoted"). Do not write `completed_rejected`-style exits that end a run before F1/F2.
- An experiment that ends without a leaderboard result (failed or interrupted run, or a diagnostic that produces no official forecast by design): `lbstatus.py set <experiment> --status 실패|중단|완료 --note "<reason>"` so it leaves the list. "Gate not passed" is not such a reason; see the rule above.
- If posting 대기 fails, fix it (HF token in `~/.cache/huggingface/token`) before launching. Do not start an unposted experiment.
- `lbstatus.py list` shows active entries and recent history; `lbstatus.py rm <experiment>` deletes a mistaken entry.

## Long-Running Experiments

The agent's bash tool kills processes it started when the call ends, so `&`, `nohup`, and `expos run --background` die silently (empty log, no run record).

- Launch any run longer than a few minutes outside the tool's process tree, wrapped by the status poster: `systemd-run --user --unit=<eid>-run --working-directory=$HOME/sunrun/leaderboard /home/t-lab01/sunrun/.venv/bin/python /home/t-lab01/sunrun/leaderboard/lbstatus.py run <eid> -- /home/t-lab01/sunrun/.venv/bin/python -m expos run <eid> --timeout 21600`. Check it with `systemctl --user status <eid>-run` and the `research/expos/runs/<eid>/run*.log` file.
- If a background launch fails once, run it in the foreground right away (`expos run <eid> --timeout ...` with a matching tool timeout). Do not diagnose the launcher or run extra timing/benchmark scripts instead of the requested experiment.
- When the user asks for an experiment, the goal is a started run. Once code and smoke checks pass, launch it; do not add further pre-checks.
- Tell the user in one line what you are doing at least every 5 minutes. Do not spend long stretches on tool calls with no message.

## Coding Style & Naming Conventions

Follow nearby Python code: four-space indentation, `snake_case` functions and modules, and uppercase constants. Use descriptive study directories and `test_*.py` test files. Preserve timestamp, unit, and manifest conventions. No root formatter or lint configuration is established; use component-specific tooling where available.

## Testing Guidelines

Use pytest and run the smallest suite covering changed behavior. Add regression tests for fixes, especially timestamp alignment, source availability, and data integrity. No workspace-wide coverage threshold is defined. Record validation commands, outcomes, and required data or environment limitations alongside research evidence.

## Commit & Pull Request Guidelines

Recent commits use experiment identifiers followed by concrete findings or corrections, such as `G004: align rounding across the dispersion note`. Follow that pattern when applicable. PRs should describe scope, affected experiments, validation commands, and linked issues; include figures for visual changes. Preserve historical results and protected inputs. Never commit tokens or private datasets.
