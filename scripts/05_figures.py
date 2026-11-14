"""Stage 5: the five paper figures, written to outputs/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vs30nh import state
from vs30nh.config import OUT
from vs30nh.data import tracts
from vs30nh.figures import (fig1_folds_variogram, fig2_predicted_vs_measured, fig3_shap_summary,
                            fig4_facility_map, fig5_siting_and_knowledge, fig6_scenario_hazard)

s1, s2, s3, s4 = (state.load(f"stage{i}") for i in (1, 2, 3, 4))
T = tracts()
fig1_folds_variogram(s1["labels"], T, s1["folds"])
fig2_predicted_vs_measured(s1["y"], s1["folds"]["is_bay"], s2["slope_oof"], s2["operational"], s2["preds"])
fig3_shap_summary(s3["shap"], s1["X"])
fig4_facility_map(s4["fac"], T, s4["trace_xy"])
fig5_siting_and_knowledge(s4["fac"], s4["bg"], s4["strat"])
fig6_scenario_hazard(s4["fac"], T, s4["trace_xy"])
print("wrote", sorted(p.name for p in OUT.iterdir()))
