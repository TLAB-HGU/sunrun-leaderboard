# Provenance after the 2026-11-02 CLI fix

Commit `35cf14bc9` (*calib11f/g/h: --all keeps each period's own round count*) changed
only the CLI: `--rounds` defaults to `None` and `--all` resolves each period from `ROUNDS_BY_PERIOD` instead of
forcing `ROUNDS = 300` on every period. Nothing in the weighting, design, fitting or prediction path changed, so
the forecast is still the output of the current `experiment.py`.

Reproduction with the current file (rounds per period: F1 300 / F2 300 / F3 450):

    /home/t-lab01/sunrun/.venv/bin/python experiment.py --phase weights --period F3 --out /tmp/x
    /home/t-lab01/sunrun/.venv/bin/python experiment.py --phase fit --period F3 --block 0 --out /tmp/x

`/tmp/x/F3_b0.npz` is identical to `results/F3_b0.npz` (max |diff| 0.0, checked 2026-11-02), and
`experiment.py --all` now runs exactly these per-period counts and prints them as a
`{"phase": "rounds", "rounds": {...}}` receipt before fitting.
