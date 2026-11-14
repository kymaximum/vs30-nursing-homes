"""Baselines: slope-only (isotonic stand-in for Wald & Allen 2007) and the
operational Wills et al. (2015) hybrid California Vs30 map with its SD grid.
"""
from __future__ import annotations

import zipfile

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .config import CFG, sample_raster, unzip


def operational_rasters():
    """Return (vs30_tif, sd_tif) paths from the USGS/CGS hybrid map release."""
    root = unzip("vs30map", "version_1.0.zip", "vs30map")
    for z in root.rglob("*.zip"):
        try:
            zipfile.ZipFile(z).extractall(root / "grids")
        except zipfile.BadZipFile:
            pass                                  # macOS resource-fork stubs
    tifs = list(root.rglob("*.tif"))
    sd_tif = next(p for p in tifs if p.stem.lower().endswith("_sd"))
    vs30_tif = next(p for p in tifs if p != sd_tif and "7p5c" in p.name)
    return vs30_tif, sd_tif


def operational_at(lon, lat):
    vs30_tif, sd_tif = operational_rasters()
    return np.log(sample_raster(vs30_tif, lon, lat)), sample_raster(sd_tif, lon, lat)


def slope_baseline(X, y, tr, te):
    """Isotonic fit of ln Vs30 on 30 m slope, trained on tr only, predicted at te."""
    s = X[f"slope_{CFG['scales_m'][0]}"].to_numpy()
    m = np.isfinite(s[tr]) & np.isfinite(y[tr])
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s[tr][m], y[tr][m])
    out = np.full(len(te), np.nan)
    ok = np.isfinite(s[te])
    out[ok] = iso.predict(s[te][ok])
    return out
