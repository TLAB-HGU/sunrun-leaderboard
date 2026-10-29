# HSSPEAK25 — PREREG (written BEFORE any gate is read)

Date: 2026-10-20. Author: gjc-main.

## Provenance (why this arm exists, stated honestly)

E1590 (HSSPEAK72, MIN_LEAD=41) is registered and reproducible:
reproduced d_pool **-1.583%** (ci [-2.711, -0.346]), per-fold
F1 -2.33 / F2 -1.23 / F3 -3.30 / F4 0.00 / F5 0.00 / F6 -2.03,
matching the registered -1.483% with the same sign pattern.

E1593's official lb-eval failed **only on cells**:
```
low_low_high  5092.9 > E0472 5037.1   (+55.8)
high_low_high 3894.1 > E0472 3862.7   (+31.4)
low_high_high 4365.1 < 4598.2  (-233.1)  PASS
high_high_high 3916.1 < 4174.4 (-258.3)  PASS
total         3235.4 < 3281.3  (-45.9)   better
```

The two failing cells contain the **highest** share of target>600 km/s
(9.7%, 8.2%) and the **lowest** S (0.029). Their error concentrates at
h25-48 (naiveMSE 12436.6 and 7162.2, the largest of any horizon band).

## Hypothesis (fixed)

HSSPEAK72 injects its Gaussian bump only for h >= 41. The failing cells lose
their error at h25-48. Therefore the bump must start at the **window start**,
h = 25, because the scored window is 25-72 h and an HSS can already peak at
its beginning.

## Design (fixed now, no choice may be revisited after scores are seen)

- Identical to HSSPEAK72 except `MIN_LEAD = 25`.
- `SHRINK = 0.15`, `WIDTH = 12.0`, `HORIZON = 72` unchanged — no evidence exists
  to change them, and changing them without evidence would be fitting.
- Control, training folds, model, features, seeds: byte-identical to E1590's
  `run.py` (copied, only MIN_LEAD differs).

## What was deliberately NOT done

- The 36-combination robustness sweep over MIN_LEAD/SHRINK/WIDTH **was** run and
  is **not** used to pick this value. Its best arm (ML33/SH0.15/WI18, -1.921%)
  is explicitly **not** claimed, because it was chosen by looking at the roll6
  gate = fold-fitting. MIN_LEAD=25 is chosen from the window definition alone.
- No official-window (`lb-eval`) score has been read for this arm.
- Cell labels were NOT used to choose MIN_LEAD.

## Judgement (fixed)

Primary gate, identical to E1590:
`roll6.compare` vs `E1508+E1529` — require `pooled_rel <= -1%`,
`ci95 upper < 0`, `folds_not_worse >= 4`, `f1_cand >= f1_ref`.

- If roll6 passes: run exactly ONE official lb-eval and report it, whatever it says.
- If roll6 fails: record the negative and stop. Do not tune.
- Claiming SOTA additionally requires **all four cell gates** `<= E0472`
  and `total_mse <= 3289.09` and `f1 >= champion`. Total MSE alone is not enough.
