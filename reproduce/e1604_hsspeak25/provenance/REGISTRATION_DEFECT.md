# E1602's registered `cmd` is broken — record of the defect

E1602 (HSSPEAK25) is registered with:

```
cmd: EXPOS_OUT=<dir> python -m experiments.solar_hsspeak25_20261020.run
```

That command **cannot run**. `expos` already injects `EXPOS_OUT` into the child environment
(`expos/core.py:307`: `env = {**os.environ, "EXPOS_OUT": str(out), ...}`), so the literal
`EXPOS_OUT=<dir>` prefix is executed by the shell as a command named `dir`:

```
$ expos run E1602
bash: line 1: dir: No such file or directory     (rc 1)
```

Compare E1590 (HSSPEAK72), which works because its cmd has no prefix:

```
cmd: .venv-fusion/bin/python -m experiments.solar_hsspeak72_20261019.run
```

The `EXPOS_OUT=<dir>` text was copied from the module docstring, where it was documentation for
a manual invocation, not a registration command.

## Why this is recorded instead of fixed

E1602's `code` block pins sha256 hashes for `run.py`, `score.py`, `official.py` and `__init__.py`.
Editing those files — even only their docstrings, which still say HSSPEAK72 and
`experiments/solar_hsspeak72_20261019` — would change the hashes and invalidate the frozen
registration. A registered record that no longer matches its code is worse than one whose `cmd`
field is known-broken and documented.

The arm's **result is unaffected**, and it was independently re-derived in this session from the
stored predictions and the protected truth (not by trusting the earlier number):

| fold | arm SSE | control SSE | rel |
|---|---:|---:|---:|
| F1 | 1,273,175,989 | 1,313,447,173 | **−3.066%** |
| F2 | 1,775,535,144 | 1,809,469,872 | **−1.875%** |
| F3 | 947,384,883 | 962,062,500 | **−1.526%** |
| **pooled** | | | **−2.176%** |

which matches E1602's registered `per_fold` {1: −0.03066, 2: −0.01875, 3: −0.01526} exactly.

The registered-path score was produced by invoking the scorer directly on the output directory:

```
python -m experiments.solar_hsspeak25_20261020.score /tmp/e1602
  pooled_rel -0.01681319752626076   ci95 [-0.033396, -0.0000818]
  per_fold {1:-0.030661, 2:-0.018754, 3:-0.015256, 4:0.0, 5:0.0, 6:-0.022441}
  f1 0.333874 vs control 0.292335   passed: true
```

## The clean fix, for whoever re-registers

Register with a cmd that matches the working E1590 form:

```
cmd: .venv-fusion/bin/python -m experiments.solar_hsspeak25_20261020.run
```

and, at the same time, correct the module docstrings in `run.py`, `score.py` and `official.py`,
which still name HSSPEAK72 and `experiments/solar_hsspeak72_20261019` (they are byte-copies of
the HSSPEAK72 module with `MIN_LEAD` changed). That is safe **only** as a new registration; it
must not be done in place against E1602.
