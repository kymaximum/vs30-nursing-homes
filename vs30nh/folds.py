"""Spatial folds from an empirical variogram, and the Bay Area holdout.

Block width is set from the variogram range of ln Vs30 residuals after
removing the lithology mean, not by convention. The Bay Area (all sites inside
``CFG['bay_wkt']``) is withheld from cross-validation entirely and used as the
regional test for interval coverage and for a final RMSE.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CFG, to_points


def variogram_range(xy, resid, rng, n_bins=25, max_lag_km=120):
    i, j = np.triu_indices(len(xy), 1)
    sub = rng.choice(len(i), size=min(400_000, len(i)), replace=False)
    i, j = i[sub], j[sub]
    h = np.linalg.norm(xy[i] - xy[j], axis=1) / 1000.0
    g = 0.5 * (resid[i] - resid[j]) ** 2
    m = h < max_lag_km
    edges = np.linspace(0, max_lag_km, n_bins + 1)
    b = np.digitize(h[m], edges) - 1
    gamma = np.bincount(b, g[m], n_bins) / np.maximum(np.bincount(b, minlength=n_bins), 1)
    sill = gamma[-5:].mean()
    return float(edges[1:][np.argmax(gamma >= 0.95 * sill)]), edges[1:], gamma


def make_folds(labels, lithology, y, rng):
    """Return dict with xy (metres), fold id, is_bay mask, and variogram pieces."""
    pts_m = to_points(labels).to_crs(CFG["crs_m"])
    xy = np.c_[pts_m.geometry.x, pts_m.geometry.y]
    resid = y - pd.Series(y).groupby(np.asarray(lithology)).transform("mean").to_numpy()
    vrange_km, vg_edges, vg_gamma = variogram_range(xy, resid, rng)

    block_m = max(vrange_km, CFG["min_block_km"]) * 1000
    block_id = np.unique((xy // block_m).astype(int), axis=0, return_inverse=True)[1].ravel()
    fold = pd.Series(rng.permutation(block_id.max() + 1) % CFG["n_folds"])[block_id].to_numpy()

    bay = gpd.GeoSeries.from_wkt([CFG["bay_wkt"]], crs=4326).iloc[0]
    is_bay = to_points(labels).within(bay).to_numpy()
    print(f"range {vrange_km:.0f} km | block {block_m/1000:.0f} km | bay holdout {is_bay.sum()} of {len(y)}")
    print("sites per fold (non-bay):", np.bincount(fold[~is_bay]))
    return dict(xy=xy, fold=fold, is_bay=is_bay, vrange_km=vrange_km,
                block_km=block_m / 1000, vg_edges=vg_edges, vg_gamma=vg_gamma)
