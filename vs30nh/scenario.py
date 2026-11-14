"""HayWired M7.0 scenario ground motion at facilities.

BSSA14 (Boore et al. 2014) is evaluated twice per facility, changing only Vs30.
Source and path terms are identical and cancel, so the difference is the site
effect alone; the nonlinear site term is included natively because both
evaluations share a reference-rock motion. Facilities with no operational Vs30
(outside the raster) get NaN rather than a default-soil difference.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .config import CFG


def rupture_surface():
    from openquake.baselib import __version__ as oq_version
    from openquake.hazardlib.geo import Line, Point
    from openquake.hazardlib.geo.surface import SimpleFaultSurface
    assert tuple(int(v) for v in oq_version.split(".")[:2]) >= (3, 19), oq_version

    trace_xy = np.array(CFG["hayward_trace"], float)
    trace_km = gpd.GeoSeries([LineString(trace_xy)], crs=4326).to_crs(CFG["crs_m"]).length.iloc[0] / 1000
    assert 70 < trace_km < 95, f"Hayward trace is {trace_km:.0f} km; HayWired rupture is ~83 km"
    print(f"Hayward trace: {len(trace_xy)} vertices, {trace_km:.0f} km")
    R = CFG["rupture"]
    surface = SimpleFaultSurface.from_fault_data(
        Line([Point(float(x), float(yy)) for x, yy in trace_xy]),
        R["usd"], R["lsd"], R["dip"], R["spacing"])
    return surface, trace_xy, trace_km


def scenario_differences(fac: pd.DataFrame) -> pd.DataFrame:
    """Add rjb, near, d_<IMT>, d_<IMT>_lo/_hi and excludes_zero columns to ``fac``."""
    from openquake.hazardlib.contexts import simple_cmaker
    from openquake.hazardlib.geo import Mesh
    from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014

    surface, _, _ = rupture_surface()
    rjb = surface.get_joyner_boore_distance(Mesh(fac["lon"].to_numpy(), fac["lat"].to_numpy()))
    cmaker = simple_cmaker([BooreEtAl2014()], list(CFG["imts"]))
    R = CFG["rupture"]

    def gmm(ln_vs30):
        """(mean ln GM, total sigma), each n_imt x n_sites, for the given ln Vs30."""
        ctx = cmaker.new_ctx(len(fac))
        ctx["mag"], ctx["rake"] = R["mag"], R["rake"]
        ctx["rjb"] = rjb
        ctx["vs30"] = np.clip(np.exp(np.nan_to_num(ln_vs30, nan=np.log(360.0))), 90, 1500)
        out = cmaker.get_mean_stds([ctx])[0]       # [mean, sig, tau, phi]
        return out[0], out[1]

    op_ok = np.isfinite(fac["ln_vs30_op"].to_numpy())
    (mean_op, sig_op), (mean_ml, sig_ml) = gmm(fac["ln_vs30_op"].to_numpy()), gmm(fac["ln_vs30_ml"].to_numpy())
    (mean_lo, _), (mean_hi, _) = gmm(fac["lo"].to_numpy()), gmm(fac["hi"].to_numpy())
    for i, imt in enumerate(CFG["imts"]):
        # deterministic scenario hazard: median and 84th-percentile ground motion (g) at each site
        fac[f"{imt}_op"] = np.where(op_ok, np.exp(mean_op[i]), np.nan)
        fac[f"{imt}_op_p84"] = np.where(op_ok, np.exp(mean_op[i] + sig_op[i]), np.nan)
        fac[f"{imt}_ml"] = np.exp(mean_ml[i])
        fac[f"{imt}_ml_p84"] = np.exp(mean_ml[i] + sig_ml[i])
        fac[f"sig_{imt}"] = sig_op[i]
        fac[f"d_{imt}"] = np.where(op_ok, mean_ml[i] - mean_op[i], np.nan)
        fac[f"d_{imt}_lo"] = np.where(op_ok, np.minimum(mean_lo[i], mean_hi[i]) - mean_op[i], np.nan)
        fac[f"d_{imt}_hi"] = np.where(op_ok, np.maximum(mean_lo[i], mean_hi[i]) - mean_op[i], np.nan)
    fac["rjb"] = rjb
    fac["near"] = fac["rjb"] <= CFG["near_km"]
    fac["excludes_zero"] = ((fac["d_PGA_lo"] > 0) | (fac["d_PGA_hi"] < 0)).where(op_ok)
    return fac


def hazard_table(fac: pd.DataFrame, thresholds=(0.1, 0.2, 0.4)) -> pd.DataFrame:
    """DSHA summary for the footprint: median PGA by distance band and counts of
    facilities whose median scenario PGA exceeds each threshold, under each Vs30 map."""
    nf = fac[fac["near"] & fac["PGA_op"].notna()].copy()
    nf["band"] = pd.cut(nf["rjb"], [0, 10, 25, 50, 100, CFG["near_km"]], include_lowest=True)
    by_band = nf.groupby("band", observed=True).agg(
        n=("PGA_op", "size"), beds=("beds", "sum"),
        pga_op_median=("PGA_op", "median"), pga_ml_median=("PGA_ml", "median"),
        pga_op_p84_median=("PGA_op_p84", "median"), sigma=("sig_PGA", "median"))
    exceed = pd.DataFrame({
        f"n>{t:g}g (operational)": [(nf["PGA_op"] > t).sum()] for t in thresholds
    } | {f"n>{t:g}g (model)": [(nf["PGA_ml"] > t).sum()] for t in thresholds}, index=["footprint"]).T
    return by_band, exceed


def summarise(fac: pd.DataFrame) -> pd.DataFrame:
    """Describe d_<IMT> for all facilities and for those in the footprint."""
    op_ok = fac["d_PGA"].notna()
    rows = []
    for label, m in [("all", op_ok), ("footprint", op_ok & fac["near"])]:
        d = fac.loc[m, [f"d_{i}" for i in CFG["imts"]]].describe().T
        d["excludes_zero"] = fac.loc[m, "excludes_zero"].mean()
        d.insert(0, "subset", label)
        rows.append(d)
    return pd.concat(rows)
