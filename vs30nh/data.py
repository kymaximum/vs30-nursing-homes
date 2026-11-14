"""Labels (measured Vs30), nursing facilities, and uniform background points.

Measured Vs30 comes from the Yong et al. USGS compilation (2020 update),
restricted to California by a spatial join against TIGER tracts. CMS ships
facility coordinates with a geocoding footnote, which is used as a quality
filter. Background points are uniform over California and serve as the control
group for the siting test.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CFG, cached, col, fetch, to_points, unzip

FOOTNOTE_OK = {"", "nan", "none"}

_TRACTS = None
_CA_POLY = None


def tracts() -> gpd.GeoDataFrame:
    global _TRACTS, _CA_POLY
    if _TRACTS is None:
        shp = next(unzip("tracts", "tracts_ca.zip", "tracts").rglob("*.shp"))
        _TRACTS = gpd.read_file(shp).to_crs(4326)
        _CA_POLY = _TRACTS.union_all()
    return _TRACTS


def ca_poly():
    tracts()
    return _CA_POLY


def build_labels() -> pd.DataFrame:
    df = pd.read_csv(fetch("vs30", "vs30_compilation_2020.csv"), encoding="utf-8-sig")
    df = df.rename(columns={col(df, "ID"): "site_id", col(df, "LAT"): "lat", col(df, "LON"): "lon",
                            col(df, "VS30"): "vs30", col(df, "METHOD"): "method"})
    df["vs30"] = pd.to_numeric(df["vs30"], errors="coerce")
    df = df[df["vs30"].between(80, 2500)].dropna(subset=["lon", "lat"])
    df = df[to_points(df).within(ca_poly()).to_numpy()]
    df["ln_vs30"] = np.log(df["vs30"])
    return (df.groupby("site_id")
              .agg(lon=("lon", "mean"), lat=("lat", "mean"), ln_vs30=("ln_vs30", "mean"),
                   method=("method", "first"), n_meas=("vs30", "size"))
              .reset_index())


def build_facilities() -> pd.DataFrame:
    df = pd.read_csv(fetch("cms", "cms_nh.csv"), low_memory=False)
    df.columns = df.columns.str.lower().str.replace(r"\W+", "_", regex=True).str.strip("_")
    df = df[df[col(df, "state")].astype(str).str.strip().eq("CA")]

    c_id = (col(df, "ccn", required=False) or col(df, "provider_number", required=False)
            or col(df, "provider_id"))
    names = {c_id: "provider_id",
             col(df, "latitude"): "lat", col(df, "longitude"): "lon",
             col(df, "certified_beds"): "beds",
             col(df, "overall_rating"): "rating",
             col(df, "ownership_type"): "ownership"}
    df = df.rename(columns=names)
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    c_fn = col(df, "geocoding_footnote", required=False)
    footnote = (df[c_fn].astype("string").fillna("").str.strip().str.lower()
                if c_fn else pd.Series("", index=df.index, dtype="string"))
    keep = footnote.isin(FOOTNOTE_OK) & df[["lat", "lon"]].notna().all(axis=1)
    print(f"facilities: dropped {(~keep).sum()} of {len(df)} on coordinate quality")
    if keep.sum() == 0:
        raise RuntimeError("no facilities survived; check the resolved CMS columns")

    out = df.loc[keep, ["provider_id", "lat", "lon", "beds", "rating", "ownership"]].reset_index(drop=True)
    svi = pd.read_csv(fetch("svi", "svi_ca.csv"), dtype={"FIPS": str})
    j = gpd.sjoin(to_points(out), tracts()[["GEOID", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out["svi"] = j["GEOID"].map(svi.set_index("FIPS")["RPL_THEMES"]).replace(-999, np.nan).to_numpy()
    return out


def sample_background(n: int, rng: np.random.Generator) -> pd.DataFrame:
    minx, miny, maxx, maxy = ca_poly().bounds
    pts = gpd.GeoSeries(gpd.points_from_xy(
        rng.uniform(minx, maxx, 4 * n), rng.uniform(miny, maxy, 4 * n)), crs=4326)
    keep = pts[pts.within(ca_poly())].iloc[:n]
    return pd.DataFrame({"lon": keep.geometry.x.values, "lat": keep.geometry.y.values})


def load_all(rng: np.random.Generator):
    labels = cached("labels", build_labels)
    fac = cached("facilities", build_facilities)
    bg = cached("background", lambda: sample_background(len(fac), rng))
    return labels, fac, bg
