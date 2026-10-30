"""D143 Axis A2b spec: residual dv on a strong base with 96-dim scenario.

Single-axis rule: base is the E0408 recipe (per-lead mixture, CPU,
existing_all plus 17 strip columns, pinned per-lead weights, depth 3,
900 trees, lr 0.03); the only mechanism change vs the reference is the
A2b residual corrector plus OOF blend. A2 used an 84-dim past plus path
scenario with direct replay; A2b extends it to 96 dims by adding the
recent base error and replays only residuals.

Scenario (A2b, 96 dims = 12 past + 12 error + 72 path):
  past(O) = [v(O-11h),...,v(O)] from hourly filled speed, trailing only;
  err(O) = past(O) minus hind(O), where hind(O) is the seed OOF-fit base
    path scored at O-12h, first 12 horizons (the 12h forecast made 12h ago
    for the past-12h window; parameters predate the cutoff);
  path(O) = seed OOF-fit base mixture prediction for O..O+72h;
  distance^2 = mean(past diff^2) plus mean(err diff^2) plus mean(path
    diff^2), single unit km/s, no scaler, equal weight per subspace.

Residual definition:
  dv = y minus yhat_base, where yhat_base is the fixed base mixture
    prediction and y is the observed outcome. For a candidate H,
    dv(H,h) = filled[H+h] minus basepath(H,h). The query estimate is the
    uniform mean of the K nearest candidates dv curves. Final per seed is
    base_full plus w times dv_analog (NaN dv falls back to base).
    HLH targeting comes from conditioning on the forecast path: past OOF
    trajectories similar to the forecast path where HLH errors occur
    supply the residual, so high-speed over and under estimates are
    corrected without touching the base.

Candidates/pool/blend:
  candidates H hourly with H <= O-654.48h (27.27d exclusion),
    H >= O-8760h (365d bound), H+72h < O, and for official origins
    additionally H+72h < cutoff (pre-cutoff pool);
  blend weight w = grid argmin of OOF MSE over {0,0.05,...,1.0}, per seed,
    on OOF origins (3h grid 2026-03-04..2026-05-28, pre-cutoff labels only).

Official-window gate is read against E0368 (lb-eval, 3 seeds).
"""

DIRECTION = "D143"
OWNER = "worker-t1a2"
KIND = "full"
REFERENCE = "E0368"

SEEDS = (0, 1, 2)
SMOKE_SEED = 0

CUTOFF_ISO = "2026-05-29T00:00:00Z"
ORIGIN_START = "2026-05-31 23:00"
ORIGIN_END = "2026-09-26 23:00"
N_OFFICIAL_ORIGINS = 2833
TRAIN_FREQ = "3h"
SPLIT_H = 24

OOF_TRAIN_END_ISO = "2026-03-01T00:00:00Z"
OOF_START_ISO = "2026-03-04T00:00:00Z"
OOF_END_ISO = "2026-05-28T23:00:00Z"
OOF_FREQ = "3h"

K = 15
SCENARIO_H = 12
ERR_H = 12
PATH_H = 72
SCENARIO_DIM = 96
CARRINGTON_H = 654.48  # 27.27d * 24
POOL_MAX_LOOKBACK_H = 8760  # 365d bound
SMOKE_LOOKBACK_H = 720  # smoke-only bound; full runs use the 365d bound
BLEND_GRID = tuple(round(x, 2) for x in [i * 0.05 for i in range(21)])

SCRATCH = "store/scratch/worker-t1a2"
FIT_NJOBS = 6

# E0408 recipe base params; run sets max_depth=3 plus random_state per fit.
# Columns are existing_all plus 17 strip columns.
BASE_PARAMS = dict(n_estimators=900, learning_rate=0.03, subsample=0.8,
                   colsample_bytree=0.7, min_child_weight=20, reg_lambda=5,
                   tree_method="hist", n_jobs=FIT_NJOBS, device="cpu")
BASE_DEPTH = 3
BASE_COLUMNS = "existing_all+strips17"

# Pinned per-lead convex member weights from dev OOF only. Never refit on
# later windows; shared by every seed of this run (E0408 setting).
WEIGHTS = {
    "1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
            "d45_lvlmean": 0.3946530834632206},
    "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
             "d45_lvlmean": 0.3411415011482895},
    "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
              "d45_lvlmean": 0.3333333333333333},
    "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
              "d45_lvlmean": 0.3333333333333333},
}

# Scenario path source: the seed OOF fit (targets < OOF_TRAIN_END).
# Same model scores query paths, OOF paths, hind paths and every candidate
# path, so all subspaces are comparable; all parameters predate the cutoff.
PATH_SOURCE = "oof-fit"

# Residual definition string for reporting (dv = y minus base prediction).
RESIDUAL_DEF = "dv = y - yhat_base"

AXES_TAG = "[axes: feat=analog/residual-dv, ens=oof-blend]"

SELECT_RULE = ("lb-eval on the official window vs best E0368 3706.19: "
               "3 seeds, seed-mean below best, every seed below best, "
               "seed-mean F1 at least champion, four high cells not worse")
