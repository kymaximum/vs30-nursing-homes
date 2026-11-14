"""Siting test (facilities vs uniform background) and SVI-stratified summaries.

These describe the coverage of the measurement record, not the distribution of
risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import CFG, to_points


def boot_ci(v, rng, n=2000):
    s = rng.choice(v, (n, len(v))).mean(1)
    return float(np.mean(v)), *map(float, np.quantile(s, [0.025, 0.975]))


def facility_table(fac, Xf, model, interval, novelty, label_xy, operational_at):
    fac = fac.copy()
    fac["beds"] = pd.to_numeric(fac["beds"], errors="coerce")
    fac["svi_q"] = pd.qcut(fac["svi"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    fac["ln_vs30_op"], fac["sd_op"] = operational_at(fac["lon"], fac["lat"])
    fac["ln_vs30_ml"] = model.predict(Xf)
    fac["lo"], fac["hi"] = interval(Xf)
    fac["novelty"] = novelty(Xf)
    fac_m = to_points(fac).to_crs(CFG["crs_m"])
    fac["d_label_km"] = NearestNeighbors(n_neighbors=1).fit(label_xy).kneighbors(
        np.c_[fac_m.geometry.x, fac_m.geometry.y])[0].ravel() / 1000
    fac["disagree"] = fac["ln_vs30_ml"] - fac["ln_vs30_op"]
    fac["width"] = fac["hi"] - fac["lo"]
    return fac


def background_table(bg, Xb, model, interval, operational_at):
    bg = bg.copy()
    bg["ln_vs30_op"], _ = operational_at(bg["lon"], bg["lat"])
    bg["disagree"] = model.predict(Xb) - bg["ln_vs30_op"]
    lo, hi = interval(Xb)
    bg["width"] = hi - lo
    return bg


def siting_test(fac, bg, rng):
    return pd.DataFrame([
        dict(group=g, metric=m, **dict(zip(["mean", "lo", "hi"], boot_ci(d[m].dropna().to_numpy(), rng))))
        for g, d in [("facilities", fac), ("background", bg)]
        for m in ["disagree", "width"]])


def stratified(fac):
    return (fac.dropna(subset=["svi_q"])
               .groupby("svi_q", observed=True)
               .apply(lambda d: pd.Series(dict(
                   n=len(d), width=d["width"].mean(),
                   d_label_km=d["d_label_km"].mean(), d_label_km_median=d["d_label_km"].median(),
                   novelty=d["novelty"].mean(), d_pga=d["d_PGA"].mean(),
                   excl_zero=d["excludes_zero"].mean())), include_groups=False))
