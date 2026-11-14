"""Terrain (3DEP stencil) and geology (SGMC) features.

Elevation is sampled on a nine-point stencil at four radii instead of
rasterising the state. Lithology and age enter as native categoricals with
rare classes pooled on the labels and the other tables aligned to the same
categories.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CAT, CFG, DATA, cached, col, to_points, unzip
from .data import ca_poly

# ---------------------------------------------------------------- terrain
OFF = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)], float)


def stencil_coords(lon, lat, r_m):
    dlat = r_m / 111320.0
    dlon = dlat / np.cos(np.radians(lat))
    return (lon[:, None] + OFF[:, 0] * dlon[:, None],
            lat[:, None] + OFF[:, 1] * dlat)


def terrain_at(lon, lat, r_m):
    import py3dep
    gx, gy = stencil_coords(lon, lat, r_m)
    z = np.asarray(py3dep.elevation_bycoords(
        list(zip(gx.ravel(), gy.ravel())), crs=4326, source="tep"), float)
    z = z.reshape(gx.shape)
    dzdx = (z[:, 1] - z[:, 2]) / (2 * r_m)
    dzdy = (z[:, 3] - z[:, 4]) / (2 * r_m)
    lap = (z[:, 1] + z[:, 2] + z[:, 3] + z[:, 4] - 4 * z[:, 0]) / r_m ** 2
    return pd.DataFrame({
        f"slope_{r_m}": np.degrees(np.arctan(np.hypot(dzdx, dzdy))),
        f"curv_{r_m}": lap,
        f"tpi_{r_m}": z[:, 0] - z[:, 1:].mean(1),
        f"rough_{r_m}": z[:, 1:].std(1),
        "elev": z[:, 0],
    })


def terrain_table(df: pd.DataFrame) -> pd.DataFrame:
    parts = [terrain_at(df["lon"].to_numpy(), df["lat"].to_numpy(), r) for r in CFG["scales_m"]]
    out = pd.concat([p.drop(columns="elev") for p in parts], axis=1)
    out["elev"] = parts[0]["elev"]
    return out.set_index(df.index)


# ---------------------------------------------------------------- geology
def pool_rare(s: pd.Series, min_n: int) -> pd.Series:
    keep = s.value_counts()[lambda v: v >= min_n].index
    return s.where(s.isin(keep), "other").astype("category")


def load_sgmc() -> gpd.GeoDataFrame:
    import pyogrio
    if (DATA / "sgmc_ca.zip").exists():
        d = unzip("sgmc", "sgmc_ca.zip", "sgmc_ca")
        src = next(p for p in d.rglob("*.shp") if "poly" in p.name.lower() or "geol" in p.name.lower())
        layer = None
    else:
        src = next(unzip("sgmc", "sgmc_geodatabase.zip", "sgmc").rglob("*.gdb"))
        layer = "SGMC_Geology"
    layer_crs = pyogrio.read_info(src, layer=layer)["crs"]
    mask = gpd.GeoSeries([ca_poly()], crs=4326).to_crs(layer_crs).iloc[0]
    g = gpd.read_file(src, layer=layer, mask=mask).to_crs(4326)
    print(f"SGMC: {len(g)} polygons in the California mask")
    lith = col(g, "GENERALIZ")
    age = col(g, "MIN_MAX_AGE", required=False) or col(g, "AGE")
    return g[[lith, age, "geometry"]].rename(columns={lith: "lithology", age: "age"})


def geo_table(df: pd.DataFrame, sgmc: gpd.GeoDataFrame, contacts: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = to_points(df)
    g = gpd.sjoin(pts, sgmc, how="left", predicate="within").drop(columns="index_right")
    g = g[~g.index.duplicated()]
    near = gpd.sjoin_nearest(pts.to_crs(CFG["crs_m"])[["geometry"]], contacts,
                             how="left", distance_col="d_contact")
    near = near[~near.index.duplicated()]
    return pd.DataFrame({
        "lithology": g["lithology"].fillna("unmapped").astype(str),
        "age": g["age"].fillna("unmapped").astype(str),
        "d_contact": near["d_contact"].to_numpy(),
    }, index=df.index)


# ---------------------------------------------------------------- assembly
def build_features(labels, fac, bg):
    """Return (X, Xf, Xb, y, NUM): feature tables aligned on categories."""
    terrain = cached("terrain_labels", lambda: terrain_table(labels))
    terrain_fac = cached("terrain_fac", lambda: terrain_table(fac))
    terrain_bg = cached("terrain_bg", lambda: terrain_table(bg))

    def _geo():
        sgmc = load_sgmc()
        contacts = gpd.GeoDataFrame(geometry=sgmc.to_crs(CFG["crs_m"]).boundary, crs=CFG["crs_m"])
        return sgmc, contacts

    _ctx = {}

    def geo_for(df):
        if not _ctx:
            _ctx["sgmc"], _ctx["contacts"] = _geo()
        return geo_table(df, _ctx["sgmc"], _ctx["contacts"])

    geo = cached("geo2_labels", lambda: geo_for(labels))
    geo_fac = cached("geo2_fac", lambda: geo_for(fac))
    geo_bg = cached("geo2_bg", lambda: geo_for(bg))

    for c in CAT:                               # pool on the labels, align the others
        geo[c] = pool_rare(geo[c], CFG["rare_class_min_n"])
        for other in (geo_fac, geo_bg):
            other[c] = pd.Categorical(other[c].where(other[c].isin(geo[c].cat.categories), "other"),
                                      categories=geo[c].cat.categories)

    X = pd.concat([terrain, geo], axis=1)
    Xf = pd.concat([terrain_fac, geo_fac], axis=1)
    Xb = pd.concat([terrain_bg, geo_bg], axis=1)
    y = labels["ln_vs30"].to_numpy()
    num = [c for c in X.columns if c not in CAT]
    return X, Xf, Xb, y, num
