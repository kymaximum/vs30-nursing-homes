"""Stage 4: facility predictions, HayWired scenario differences, siting test, SVI strata."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from vs30nh import state
from vs30nh.baselines import operational_at
from vs30nh.config import OUT, SEED
from vs30nh.scenario import hazard_table, rupture_surface, scenario_differences, summarise
from vs30nh.siting import background_table, facility_table, siting_test, stratified

s1, s3 = state.load("stage1"), state.load("stage3")
rng = np.random.default_rng(SEED)

fac = facility_table(s1["fac"], s1["Xf"], s3["model"], s3["interval"], s3["novelty"],
                     s1["folds"]["xy"], operational_at)
fac = scenario_differences(fac)
print(summarise(fac).round(3))
by_band, exceed = hazard_table(fac)
print(by_band.round(3)); print(exceed)
bg = background_table(s1["bg"], s1["Xb"], s3["model"], s3["interval"], operational_at)

siting = siting_test(fac, bg, rng)
strat = stratified(fac)
print(siting.round(3).to_string(index=False))
print(strat.round(3))

_, trace_xy, _ = rupture_surface()
summarise(fac).to_csv(OUT / "table3a_scenario.csv")
by_band.to_csv(OUT / "table4a_dsha_by_distance.csv")
exceed.to_csv(OUT / "table4b_dsha_exceedance.csv")
siting.to_csv(OUT / "table3b_siting.csv", index=False)
strat.to_csv(OUT / "table3_stratified.csv")
fac.to_parquet(OUT / "facility_results.parquet")
state.save("stage4", dict(fac=fac, bg=bg, siting=siting, strat=strat, trace_xy=trace_xy))
