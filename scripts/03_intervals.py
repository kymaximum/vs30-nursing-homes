"""Stage 3: conformal intervals, Bay-holdout calibration and RMSE, final model, SHAP, novelty."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from vs30nh import state
from vs30nh.config import OUT
from vs30nh.model import Intervals, Novelty, bay_rmse_table, calibration_table, fit_lgb, shap_values

s1, s2 = state.load("stage1"), state.load("stage2")
X, y, num, folds = s1["X"], s1["y"], s1["num"], s1["folds"]

interval = Intervals(X, y, folds["fold"], folds["is_bay"])
calib = calibration_table(X, y, folds["is_bay"], interval, s2["operational"], s2["operational_sd"])
print(calib.round(3))
bay_rmse = bay_rmse_table(X, y, folds["fold"], folds["is_bay"], interval.best, s2["operational"])
print(bay_rmse.round(3))

model = fit_lgb(X, y, interval.best, np.arange(len(y)))   # final point model, all labels incl. Bay
sv = shap_values(model, X)
print(pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False).head(10).round(3))
novelty = Novelty(X, num)

calib.to_csv(OUT / "table2b_calibration.csv")
bay_rmse.to_csv(OUT / "table2d_bay_rmse.csv")
state.save("stage3", dict(interval=interval, model=model, shap=sv, novelty=novelty, calib=calib,
                          bay_rmse=bay_rmse))
