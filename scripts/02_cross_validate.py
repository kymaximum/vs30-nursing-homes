"""Stage 2: baselines, spatially blocked CV, feature-set ablation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from vs30nh import state
from vs30nh.baselines import operational_at
from vs30nh.config import OUT, rmse
from vs30nh.model import ablation, cross_validate

s = state.load("stage1")
X, y, folds, labels = s["X"], s["y"], s["folds"], s["labels"]

operational, operational_sd = operational_at(labels["lon"], labels["lat"])
m = np.isfinite(operational)
print(f"operational map covers {m.mean():.3f} of sites | in-sample rmse {rmse(y[m], operational[m]):.3f} "
      f"| median SD {np.nanmedian(operational_sd):.3f}")

cv, preds, slope_oof, best_by_fold = cross_validate(X, y, folds["fold"], folds["is_bay"], operational)
print(cv.drop(columns=["fold", "n"]).agg(["mean", "std"]).round(3))
abl = ablation(X, y, folds["fold"], folds["is_bay"], best_by_fold)
print(abl.round(3))

cv.to_csv(OUT / "table2_skill.csv", index=False)
abl.to_csv(OUT / "table2c_ablation.csv")
state.save("stage2", dict(operational=operational, operational_sd=operational_sd, cv=cv,
                          preds=preds, slope_oof=slope_oof, best_by_fold=best_by_fold, abl=abl))
