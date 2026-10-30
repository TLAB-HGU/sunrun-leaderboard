# Derived v2 datasets

`registry.json` is the executable job inventory. Run each `cmd` with its listed Python environment from the repository workspace. Every builder accepts `--end` and `--output`; the default output is `leaderboard/store/v2/<dataset>`. `--end now` captures a generation snapshot, not a live stream. Read each output `MANIFEST.json` for the actual snapshot, source coverage, hashes, return code, and limitations.

Run `gong.py` after the raw GONG fetch finishes, then run the topology and mixed-source jobs against that receipt inventory. Re-run provisional jobs after raw inventory growth. A recent observation alone does not prove historical coverage. Raw availability is conservatively the maximum of observation + 2 hours and actual HTTP Last-Modified. Recovered HEAD metadata records when it was checked.

The `vendor` tree contains isolated copies of registered extraction kernels; registered originals remain unchanged. Base OCB normalizes timestamps to nanoseconds at selection, Carrington-anchor, and table-lookup boundaries for pandas 3. WSA uses the same numerical kernel with a 120-second process timeout; timeout outputs are explicitly partial. HCS includes SunPy ephemerides and its cache manifest.

`mixed.py chunip` and `mixed.py magconc` require completed v2 CH and SUVI hourly inputs plus the verified Fe195 catalog. They write their own source identity and metadata snapshot. `mixed.py cmechip` segments source images independently and does not require the champion CH table.

Image extraction uses existing frozen weights and fitted preprocessing only. GPU jobs use the actual shared `ch-breakthrough-v2/gpu-leases/gpu{0,1}.lock` files. DINO requires the exact pinned gated checkpoint; unavailable credentials produce `blocked_credentials`, never empty completed embeddings. VJEPA 120h follows a finished frozen VJEPA snapshot; an explicit end must match that snapshot. L72-COND and TAPFN are recorded under their v2 output folders as experiment artifacts requiring consumer reruns, not independent source features.

Verification:

```sh
leaderboard/.venv-fusion/bin/python -m unittest discover -s data-v2/derived -p test_build.py
leaderboard/.venv-huxt/bin/python -m unittest discover -s data-v2/derived -p test_ocb_time_units.py
python3 -m compileall -q data-v2/derived
```

The tests cover causal future-poison behavior with a past-data positive control, Mamba missingness and sequence boundaries, and microsecond-input OCB release/rotation calculations. Scientific output readiness additionally requires successful real generation and source-bound manifests.
