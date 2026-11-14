"""Stage 1: labels, facilities, background points, terrain + geology features, spatial folds.

Needs network on first run (USGS 3DEP, ScienceBase, CMS, CDC, Census). Everything is cached
under data/ so later stages and re-runs are offline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from vs30nh import state
from vs30nh.config import SEED
from vs30nh.data import load_all, tracts
from vs30nh.features import build_features
from vs30nh.folds import make_folds

rng = np.random.default_rng(SEED)
labels, fac, bg = load_all(rng)
print(labels.shape, labels["method"].value_counts().head(6).to_dict())
print(fac.shape, f"svi missing {fac['svi'].isna().sum()}", bg.shape)

X, Xf, Xb, y, num = build_features(labels, fac, bg)
print(X.shape, "lithology classes:", list(X["lithology"].cat.categories))
print(f"missing elevation: {X['elev'].isna().mean():.3f}")

folds = make_folds(labels, X["lithology"].to_numpy(), y, rng)
tracts()  # warm the cache so figures can run offline
state.save("stage1", dict(labels=labels, fac=fac, bg=bg, X=X, Xf=Xf, Xb=Xb, y=y, num=num, folds=folds))
