"""D150 band-precursor diagnostic contract: static + synthetic checks (fast, no fits)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d150_oofset/diag_band"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_CONTACT = ["e0147-abl", "d141_e0147", "ablate", "lean.py", "D147", "d147",
                 "d148_enlil", "nn-", "sun-nn", "E0472", "E0474", "E0471",
                 "worker-t1a", "worker-t1b", "worker-t1c", "worker-t1d",
                 "worker-t1e", "worker-t1f", "worker-t1g0", "worker-t1g6",
                 "worker-t1g7", "worker-t2no"]
BANNED_SRC = ["GONG", "CDAW"]
CELL_LABELS = ["low_low_high", "low_high_high", "high_low_high", "high_high_high", "hlh"]

sys.path.insert(0, str(EXP))


def _load(mod_name, alias):
    loc = importlib.util.spec_from_file_location(alias, EXP / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code(names=("spec_band.py", "band_lib.py", "run_diag_band.py")):
    return "\n".join((EXP / n).read_text() for n in names)


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 4, files
    assert {"spec_band.py", "band_lib.py", "run_diag_band.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec_band", "d150band_spec")
    assert SP.DIRECTION == "D150" and SP.OWNER == "worker-t1g8"
    assert SP.KIND == "diag" and SP.REFERENCE == "E0421"
    assert SP.SCRATCH == "store/scratch/worker-t1g8"
    assert int(SP.FIT_NJOBS) <= 6
    assert tuple(int(x) for x in SP.WAV_SCALES) == (3, 6, 12, 24, 48, 96)
    assert tuple((float(lo), float(hi)) for lo, hi in SP.FOURIER_BANDS) == (
        (2.0, 6.0), (6.0, 24.0), (24.0, 72.0), (72.0, 168.0))
    assert tuple(SP.BAND_NAMES) == ("wav_e003", "wav_e006", "wav_e012", "wav_e024",
                                    "wav_e048", "wav_e096", "fft_02_06", "fft_06_24",
                                    "fft_24_72", "fft_72_168")
    assert abs(float(SP.LEVEL_Q) - 0.95) < 1e-12
    assert abs(float(SP.EXTREME_FRAC) - 0.05) < 1e-12
    assert int(SP.MIN_FINITE_FORWARD) == 36
    assert int(SP.FORWARD_H) == 72 and int(SP.TRAIL_H) == 192 and int(SP.FFT_N) == 168
    assert abs(float(SP.NOISE_MSE) - 90.0) < 1e-12
    assert abs(float(SP.R_MIN) - 0.3) < 1e-12 and int(SP.N_MIN) == 20
    assert abs(float(SP.RATIO_HI) - 2.0) < 1e-12 and abs(float(SP.RATIO_LO) - 0.5) < 1e-12
    assert SP.ORIGIN_CEILING_ISO == "2026-03-01T00:00:00Z"
    assert int(SP.SMOKE_WINDOWS) == 1 and int(SP.SMOKE_ORIGINS) == 8
    assert SP.NO_OFFICIAL and SP.NO_BLEND and SP.NO_SELECTION
    assert "[axes:" in SP.AXES_TAG and "diag=band-precursor" in SP.AXES_TAG
    assert "close" in SP.CLOSE_RULE.lower() and "90" in SP.CLOSE_RULE


def test_windows_cover_2023_06_to_2026_02():
    SP = _load("spec_band", "d150band_spec_c")
    BL = _load("band_lib", "d150band_lib_folds")
    assert BL.check_windows(SP.WINDOWS, SP.ORIGIN_CEILING_ISO) is True
    assert [w["name"] for w in SP.WINDOWS] == ["W0", "W1", "W2"]
    first = pd.Timestamp(SP.WINDOWS[0]["start"], tz="UTC")
    last = pd.Timestamp(SP.WINDOWS[-1]["end"], tz="UTC")
    assert first == pd.Timestamp("2023-06-01T00:00:00Z", tz="UTC")
    assert last >= pd.Timestamp("2026-02-26T00:00:00Z", tz="UTC")
    for w in SP.WINDOWS:
        te = pd.Timestamp(w["train_end"], tz="UTC")
        st = pd.Timestamp(w["start"], tz="UTC")
        en = pd.Timestamp(w["end"], tz="UTC")
        assert (st - te) == pd.Timedelta("72h")
        assert en < pd.Timestamp(SP.ORIGIN_CEILING_ISO, tz="UTC")
    for a, b in zip(SP.WINDOWS, SP.WINDOWS[1:]):
        assert pd.Timestamp(b["start"], tz="UTC") > pd.Timestamp(a["end"], tz="UTC")
    assert SP.OOF_FREQ == "3h" and SP.PURGE_H == 72
    BL.check_scales(SP.WAV_SCALES)
    BL.check_fourier_bands(SP.FOURIER_BANDS)
    BL.check_band_names(SP.BAND_NAMES)
    BL.check_level(SP.LEVEL_Q, SP.EXTREME_FRAC, SP.MIN_FINITE_FORWARD)
    BL.check_gate(SP.NOISE_MSE, SP.R_MIN, SP.N_MIN, SP.RATIO_HI, SP.RATIO_LO)
    try:
        BL.check_windows([{"name": "W0", "train_end": "2023-05-29T00:00:00Z",
                           "start": "2023-06-01T00:00:00Z",
                           "end": "2026-03-05T00:00:00Z"}],
                         SP.ORIGIN_CEILING_ISO)
    except ValueError:
        pass
    else:
        raise AssertionError("past-ceiling window must be rejected")
    try:
        BL.check_scales((3, 6, 12))
    except ValueError:
        pass
    else:
        raise AssertionError("short scale grid must be rejected")


def test_band_helpers_synthetic():
    BL = _load("band_lib", "d150band_lib_b")
    # wavelet: flat differenced series gives zero energy; a step gives positive energy
    idx = pd.date_range("2024-01-01T00:00:00Z", "2024-01-15T00:00:00Z", freq="h", tz="UTC")
    dv_flat = pd.Series(np.zeros(len(idx)), index=idx)
    dv_step = pd.Series(np.zeros(len(idx)), index=idx)
    dv_step.loc[dv_step.index >= pd.Timestamp("2024-01-08T00:00:00Z", tz="UTC")] = 5.0
    origins = pd.DatetimeIndex(["2024-01-05T00:00:00Z", "2024-01-09T00:00:00Z"], tz="UTC")
    e_flat = BL.wavelet_energy_frame(origins, dv_flat)
    e_step = BL.wavelet_energy_frame(origins, dv_step)
    assert e_flat["wav_e024"][0] == 0.0
    assert e_step["wav_e024"][1] > 0.0
    # fourier: 12h sine concentrates in the 6-24h period band
    t = np.arange(168)
    sine = np.sin(2 * np.pi * t / 12.0)
    sidx = pd.date_range("2024-02-01T00:00:00Z", periods=168, freq="h", tz="UTC")
    dv_sine = pd.Series(sine, index=sidx)
    fo = BL.fourier_energy_frame(pd.DatetimeIndex([sidx[-1]], tz="UTC"), dv_sine)
    assert fo["fft_06_24"][0] > fo["fft_02_06"][0]
    assert fo["fft_06_24"][0] > fo["fft_72_168"][0]
    assert np.isfinite(fo["fft_06_24"][0])
    # combined frame order
    fr = BL.band_frame(pd.DatetimeIndex([sidx[-1]], tz="UTC"), dv_sine)
    assert list(fr.columns) == ["wav_e003", "wav_e006", "wav_e012", "wav_e024",
                                "wav_e048", "wav_e096", "fft_02_06", "fft_06_24",
                                "fft_24_72", "fft_72_168"]
    # forward stats + onset classification on a synthetic ramp into a high episode
    hrs = pd.date_range("2024-03-01T00:00:00Z", "2024-03-10T00:00:00Z", freq="h", tz="UTC")
    spd = pd.Series(400.0 * np.ones(len(hrs)), index=hrs)
    spd.loc[spd.index >= pd.Timestamp("2024-03-05T00:00:00Z", tz="UTC")] = 800.0
    oo = pd.DatetimeIndex(["2024-03-03T00:00:00Z", "2024-03-07T00:00:00Z"], tz="UTC")
    fwd = BL.forward_stats(oo, spd, forward_h=72, level=600.0)
    assert fwd["base"][0] == 400.0 and fwd["maxf"][0] == 800.0
    assert abs(fwd["rise"][0] - 400.0) < 1e-9 and fwd["share"][0] > 0.05
    cls = BL.classify_onset(fwd["base"], fwd["share"], 600.0, frac=0.05)
    assert bool(cls["onset"][0]) is True
    assert bool(cls["quiet"][0]) is False
    # contrast: onset energies above quiet gives ratio > 1
    en = pd.DataFrame({"wav_e003": np.array([10.0, 1.0]), "fft_02_06": np.array([5.0, 1.0])})
    tab = BL.contrast_table(en, np.array([0.5, 0.0]), np.array([300.0, 10.0]),
                            np.array([True, False]), np.array([False, True]))
    assert tab["wav_e003"]["ratio"] == 10.0
    assert tab["wav_e003"]["n_onset"] == 1 and tab["wav_e003"]["n_quiet"] == 1
    # oracle: separated groups give positive gain
    oc = BL.oracle_group_gain(np.array([800.0, 400.0]), np.array([True, False]),
                              np.array([False, True]))
    assert oc["gain"] > 0 and np.isfinite(oc["gain"])
    c = BL.pearson_corr(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert c["n"] == 3 and abs(float(c["r"]) - 1.0) < 1e-12
    # gate: strong elevated precursor opens; weak contrast closes
    strong = {"b1": {"ratio": 3.0, "r_share": 0.5, "n_share": 50,
                     "r_rise": 0.1, "n_rise": 50}}
    g = BL.decide_gate(strong, {"gain": 500.0}, noise_mse=90.0, r_min=0.3, n_min=20,
                       ratio_hi=2.0, ratio_lo=0.5)
    assert g["decision"] == "open-gate-pass"
    weak = {"b1": {"ratio": 1.2, "r_share": 0.1, "n_share": 50,
                   "r_rise": 0.05, "n_rise": 50}}
    g2 = BL.decide_gate(weak, {"gain": 500.0}, noise_mse=90.0, r_min=0.3, n_min=20,
                        ratio_hi=2.0, ratio_lo=0.5)
    assert g2["decision"] == "close"
    g3 = BL.decide_gate(strong, {"gain": 10.0}, noise_mse=90.0, r_min=0.3, n_min=20,
                        ratio_hi=2.0, ratio_lo=0.5)
    assert g3["decision"] == "close"


def test_poison_synthetic():
    BL = _load("band_lib", "d150band_lib_e")
    idx = pd.date_range("2023-05-01T00:00:00Z", "2023-07-01T00:00:00Z", freq="h", tz="UTC")
    speed = pd.Series(400.0 + np.sin(np.arange(len(idx)) / 24.0), index=idx)
    cut = "2023-05-29T00:00:00Z"
    origins = pd.DatetimeIndex(["2023-05-10T00:00:00Z", "2023-05-20T00:00:00Z",
                                "2023-06-05T00:00:00Z", "2023-06-10T00:00:00Z"], tz="UTC")
    res = BL.band_poison(origins, speed, cut)
    assert res["passed"] is True and res["pre_invariant"] is True
    assert res["n_pre"] == 2 and res["n_post"] == 2
    assert len(res["moved_bands"]) > 0


def test_diag_only_outputs():
    run_src = (EXP / "run_diag_band.py").read_text()
    for w in ("diag_band.json", "causality.json", "meta.json"):
        assert w in run_src, w
    for w in ("predictions_official", "origin_last_input_utc", "pred_kms",
              "SELECT_RULE", "official-score", "official_score", "freeze",
              "submit", "upload", "push"):
        assert w not in run_src, w
    assert "NO_SELECTION" in run_src and "NO_BLEND" in run_src and "NO_OFFICIAL" in run_src


def test_scratch_isolation_and_cpu():
    import re
    SP = _load("spec_band", "d150band_spec_f")
    code = _code().replace("\\", "/")
    assert SP.SCRATCH == "store/scratch/worker-t1g8"
    assert len(set(re.findall(r"worker-[a-z0-9]+", code))) == 1
    assert "worker-t1g8" in code
    assert int(SP.FIT_NJOBS) <= 6
    assert "--gpu" not in code


def test_no_forbidden_leak_or_banned_source():
    code = _code()
    for w in FORBIDDEN:
        assert w not in code, w
    for w in OTHER_CONTACT:
        assert w not in code, w
    low = code.lower()
    for w in BANNED_SRC:
        assert w.lower() not in low, w
    for w in CELL_LABELS:
        assert w not in low, w
    assert "cuda" not in low and "--gpu" not in code
