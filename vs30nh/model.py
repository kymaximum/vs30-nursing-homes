"""Gradient-boosted ln Vs30 model: tuning inside spatial folds, cross-validated
skill against both baselines, feature-set ablation, conformalized quantile
intervals, SHAP attribution and feature-space novelty.
"""
from __future__ import annotations

from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .baselines import slope_baseline
from .config import CAT, CFG, SEED, rmse

PARAM_GRID = [dict(num_leaves=n, learning_rate=lr, min_child_samples=m, n_estimators=600)
              for n, lr, m in product((15, 31, 63), (0.03, 0.06), (10, 25))]

TERRAIN_PFX = ("slope_", "curv_", "tpi_", "rough_")


def fit_lgb(X, y, params, idx, obj="regression", alpha=None, cols=None):
    cols = list(X.columns) if cols is None else cols
    cats = [c for c in CAT if c in cols]
    return lgb.LGBMRegressor(objective=obj, alpha=alpha, verbose=-1,
                             random_state=SEED, **params).fit(
        X.iloc[idx][cols], y[idx], categorical_feature=cats)


def tune(X, y, fold, tr):
    """Pick PARAM_GRID entry by inner spatial CV on the training indices."""
    inner = fold[tr]
    scores = [np.mean([rmse(y[tr][inner == k],
                            fit_lgb(X, y, p, tr[inner != k]).predict(X.iloc[tr[inner == k]]))
                       for k in np.unique(inner)])
              for p in PARAM_GRID]
    return PARAM_GRID[int(np.argmin(scores))]


def cross_validate(X, y, fold, is_bay, operational):
    """Spatially blocked CV on non-Bay sites. Returns (cv table, oof preds, slope oof, best params)."""
    rows, best_by_fold = [], {}
    preds = np.full(len(y), np.nan)
    slope_oof = np.full(len(y), np.nan)
    for k in range(CFG["n_folds"]):
        tr = np.where((fold != k) & ~is_bay)[0]
        te = np.where((fold == k) & ~is_bay)[0]
        best_by_fold[k] = best = tune(X, y, fold, tr)
        preds[te] = fit_lgb(X, y, best, tr).predict(X.iloc[te])
        slope_oof[te] = slope_baseline(X, y, tr, te)
        rows.append(dict(fold=k, n=len(te),
                         gbm=rmse(y[te], preds[te]),
                         slope=rmse(y[te], slope_oof[te]),
                         operational=rmse(y[te], operational[te])))
        print(f"fold {k}: {rows[-1]}")
    return pd.DataFrame(rows), preds, slope_oof, best_by_fold


def ablation(X, y, fold, is_bay, best_by_fold):
    """Does the multiscale stencil add anything over a single-resolution terrain set?"""
    r0 = CFG["scales_m"][0]
    sets = {
        "full": list(X.columns),
        f"terrain@{r0}m only": [c for c in X.columns
                                if not c.startswith(TERRAIN_PFX) or c.endswith(f"_{r0}")],
        "no terrain (elev+geo)": [c for c in X.columns if not c.startswith(TERRAIN_PFX)],
        "terrain only (no geo, no elev)": [c for c in X.columns if c.startswith(TERRAIN_PFX)],
    }
    abl = {}
    for name, cols in sets.items():
        oof = np.full(len(y), np.nan)
        for k in range(CFG["n_folds"]):
            tr = np.where((fold != k) & ~is_bay)[0]
            te = np.where((fold == k) & ~is_bay)[0]
            oof[te] = fit_lgb(X, y, best_by_fold[k], tr, cols=cols).predict(X.iloc[te][cols])
        abl[name] = dict(n_features=len(cols), rmse=rmse(y[~is_bay], oof[~is_bay]),
                         **{f"fold{k}": rmse(y[(fold == k) & ~is_bay], oof[(fold == k) & ~is_bay])
                            for k in range(CFG["n_folds"])})
    return pd.DataFrame(abl).T


class Intervals:
    """Conformalized quantile regression (Romano et al. 2019) on spatial folds.

    Quantile models are fit on non-Bay folds != 0, the conformal correction is
    calibrated on non-Bay fold 0, and coverage is reported on the Bay holdout.
    """

    def __init__(self, X, y, fold, is_bay):
        a = CFG["alpha"]
        tr_all = np.where(~is_bay)[0]
        cal = tr_all[fold[tr_all] == 0]
        fit = tr_all[fold[tr_all] != 0]
        self.best = tune(X, y, fold, fit)
        self.lo_m = fit_lgb(X, y, self.best, fit, "quantile", a / 2)
        self.hi_m = fit_lgb(X, y, self.best, fit, "quantile", 1 - a / 2)
        score = np.maximum(self.lo_m.predict(X.iloc[cal]) - y[cal],
                           y[cal] - self.hi_m.predict(X.iloc[cal]))
        self.q = float(np.quantile(score, min(1.0, np.ceil((len(cal) + 1) * (1 - a)) / len(cal))))
        self.tr_all = tr_all

    def __call__(self, Xi):
        return self.lo_m.predict(Xi) - self.q, self.hi_m.predict(Xi) + self.q


def coverage(lo, hi, yy):
    m = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(yy)
    return float(np.mean((yy[m] >= lo[m]) & (yy[m] <= hi[m]))), float(np.mean(hi[m] - lo[m]))


def calibration_table(X, y, is_bay, interval, operational, operational_sd):
    a = CFG["alpha"]
    bay_idx = np.where(is_bay)[0]
    blo, bhi = interval(X.iloc[bay_idx])
    z = norm.ppf(1 - a / 2)
    olo = operational[bay_idx] - z * operational_sd[bay_idx]
    ohi = operational[bay_idx] + z * operational_sd[bay_idx]
    calib = pd.DataFrame({
        "conformal GBM": coverage(blo, bhi, y[bay_idx]),
        "operational ± z·SD": coverage(olo, ohi, y[bay_idx]),
    }, index=["coverage", "mean width (ln)"]).T
    calib["nominal"] = 1 - a
    return calib


def bay_rmse_table(X, y, fold, is_bay, best, operational):
    """Regional test: point model trained on every non-Bay label, scored on Bay sites."""
    tr_all = np.where(~is_bay)[0]
    bay_idx = np.where(is_bay)[0]
    mdl = fit_lgb(X, y, best, tr_all)
    return pd.Series(dict(
        gbm=rmse(y[bay_idx], mdl.predict(X.iloc[bay_idx])),
        slope=rmse(y[bay_idx], slope_baseline(X, y, tr_all, bay_idx)),
        operational=rmse(y[bay_idx], operational[bay_idx]),
    ), name="bay_rmse")


class Novelty:
    """Distance to the nearest training label in standardised numeric feature space."""

    def __init__(self, X, num):
        self.num = num
        self.med = X.loc[:, num].median()
        Z = X.loc[:, num].fillna(self.med)
        self.scaler = StandardScaler().fit(Z)
        self.nn = NearestNeighbors(n_neighbors=1).fit(self.scaler.transform(Z))

    def __call__(self, df):
        return self.nn.kneighbors(self.scaler.transform(df.loc[:, self.num].fillna(self.med)))[0].ravel()


def shap_values(model, X):
    import shap
    return shap.TreeExplainer(model).shap_values(X)
