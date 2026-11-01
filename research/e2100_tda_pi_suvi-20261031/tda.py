"""TDA feature extraction for EUV full-disk frames (arXiv 2610.35549 replication).

Pipeline (paper Sec. 3.2, adapted to GOES-18 SUVI L1b Fe195):
    radiance -> disk crop -> 2x rebin -> clip -> asinh stretch -> [-1,1] -> invert
    -> cubical complex (gudhi) -> H0/H1 persistence diagrams
    -> fixed 14x14 persistence images (Eq. 2, w(p)=p, sigma=0.05)

The paper's absolute constants (clip 7757.31, a = 0.005, applied to AIA data
numbers) do not transfer to SUVI radiance in W m-2 sr-1; the clip is replaced by
the 99.5th percentile of the disk pixels of the same frame. The asinh shape
parameter a = 0.005 is kept because it acts on the [0,1]-normalized image.
"""
from __future__ import annotations

import numpy as np

A = 0.005
GRID = 14
SIGMA = 0.05
REBIN = 2
EDGES = np.linspace(0.0, 1.0, GRID + 1)
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])
CENT_B, CENT_P = np.meshgrid(CENT, CENT, indexing="ij")
PIX = 1.0 / GRID
NFEAT = 2 * GRID * GRID


def read_frame(path: str) -> tuple[np.ndarray, float, float, float]:
    """Return (radiance, rsun_px, crpix1, crpix2) in physics units."""
    import h5netcdf

    with h5netcdf.File(path, "r") as f:
        rad = np.array(f["RAD"][:], dtype=np.float32)
        sf = float(f["RAD"].attrs.get("scale_factor", 1.0))
        ao = float(f["RAD"].attrs.get("add_offset", 0.0))
        rs = float(f["RSUN"][()])
        if not np.isfinite(rs) or rs <= 0:
            diam = float(f["DIAM_SUN"][()])
            rs = 0.5 * diam if np.isfinite(diam) and diam > 0 else float("nan")
        c1 = float(f["CRPIX1"][()])
        c2 = float(f["CRPIX2"][()])
    return rad * sf + ao, rs, c1, c2


def _derive_geometry(rad: np.ndarray) -> tuple[float, float, float]:
    """Fallback when RSUN/CRPIX carry fill values: centroid + area-equivalent radius."""
    hi = float(np.nanmax(rad))
    if not np.isfinite(hi) or hi <= 0:
        raise ValueError("frame has no usable intensity")
    thr = max(0.05 * hi, 1e-3)
    m = np.nan_to_num(rad, nan=0.0) > thr
    if m.sum() < 1000:
        raise ValueError("disk mask too small")
    yy, xx = np.nonzero(m)
    c1 = float(xx.mean()) + 1.0
    c2 = float(yy.mean()) + 1.0
    rs = float(np.sqrt(m.sum() / np.pi))
    return rs, c1, c2


def preprocess(rad: np.ndarray, rs: float, c1: float, c2: float) -> tuple[np.ndarray, np.ndarray]:
    """Disk crop + rebin + asinh normalization + inversion (paper Sec. 3.2)."""
    if not np.isfinite(rs) or rs < 10 or not (np.isfinite(c1) and np.isfinite(c2)):
        rs, c1, c2 = _derive_geometry(rad)
    h, w = rad.shape
    cy, cx = c2 - 1.0, c1 - 1.0
    half = int(np.ceil(rs * 1.05))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        pad = 4 * half
        big = np.full((pad, pad), np.nan, dtype=np.float32)
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(h, y1), min(w, x1)
        big[ys - y0:ys - y0 + (ye - ys), xs - x0:xs - x0 + (xe - xs)] = rad[ys:ye, xs:xe]
        rad = big
        cy, cx = cy - y0, cx - x0
        y0, y1, x0, x1 = 0, pad, 0, pad
    sub = rad[y0:y1, x0:x1]
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    disk = np.hypot(xx - (cx - x0), yy - (cy - y0)) < rs
    sub = np.where(disk, sub, 0.0).astype(np.float32)
    hh, ww = (sub.shape[0] // REBIN) * REBIN, (sub.shape[1] // REBIN) * REBIN
    img = sub[:hh, :ww].reshape(hh // REBIN, REBIN, ww // REBIN, REBIN).mean(axis=(1, 3))
    dk = disk[:hh, :ww].reshape(hh // REBIN, REBIN, ww // REBIN, REBIN).mean(axis=(1, 3)) > 0.5
    hi = float(np.percentile(img[dk], 99.5)) if dk.any() else float(img.max())
    x = np.clip(img, 0.0, hi) / max(hi, 1e-6)
    norm = np.arcsinh(x / A) / np.arcsinh(1.0 / A)      # [0, 1]
    norm = 2.0 * norm - 1.0                              # [-1, 1]
    proc = norm.max() - norm                             # invert: coronal holes become peaks
    return np.ascontiguousarray(proc.astype(np.float64)), dk


def diagrams(proc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from gudhi import CubicalComplex

    cc = CubicalComplex(top_dimensional_cells=proc)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    out = []
    for k in (0, 1):
        dg = np.asarray(cc.persistence_intervals_in_dimension(k), dtype=float)
        out.append(dg[np.isfinite(dg).all(axis=1)] if len(dg) else np.zeros((0, 2)))
    return out[0], out[1]


def persistence_image(dg: np.ndarray) -> np.ndarray:
    """Paper Eq. (2) on a fixed 14x14 grid in (birth, persistence) coordinates."""
    if len(dg) == 0:
        return np.zeros((GRID, GRID))
    b = dg[:, 0]
    p = dg[:, 1] - dg[:, 0]
    ok = np.isfinite(b) & np.isfinite(p) & (p > 0)
    b, p = b[ok], p[ok]
    if len(b) == 0:
        return np.zeros((GRID, GRID))
    w = p
    db = CENT_B.ravel()[None, :] - b[:, None]
    dp = CENT_P.ravel()[None, :] - p[:, None]
    k = np.exp(-(db ** 2 + dp ** 2) / (2 * SIGMA ** 2))
    return ((w[:, None] * k).sum(axis=0) * PIX ** 2).reshape(GRID, GRID)


def features(path: str) -> tuple[np.ndarray, dict]:
    rad, rs, c1, c2 = read_frame(path)
    proc, _ = preprocess(rad, rs, c1, c2)
    d0, d1 = diagrams(proc)
    vec = np.concatenate([persistence_image(d0).ravel(), persistence_image(d1).ravel()]).astype(np.float32)
    meta = {
        "n_h0": int(len(d0)),
        "n_h1": int(len(d1)),
        "max_h1_persistence": float((d1[:, 1] - d1[:, 0]).max()) if len(d1) else 0.0,
        "shape": int(proc.shape[0]),
    }
    return vec, meta
