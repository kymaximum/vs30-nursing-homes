"""Configuration, paths, and small I/O helpers shared by every stage.

All inputs are public downloads. ``DATA`` holds raw downloads and cached
intermediate tables (git-ignored); ``OUT`` holds figures and tables that the
paper uses (tracked).
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
OUT = ROOT / "outputs"
for _p in (DATA, CACHE, OUT):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 11

CFG = dict(
    crs_m="EPSG:3310",                       # California Albers, metres
    scales_m=(30, 90, 300, 1000),            # terrain stencil radii
    n_folds=5,
    alpha=0.10,                              # conformal miscoverage (90% PI)
    imts=("PGA", "SA(1.0)"),
    # HayWired M7.0 rupture: the Hayward fault from San Pablo Bay to Fremont
    # (Detweiler & Hudnut 2017). Surface trace as (lon, lat) vertices NW -> SE.
    # Hard-coded because the Quaternary Fault database segments do not snap and
    # a linemerge collapses to a short fragment.
    hayward_trace=[(-122.395, 38.020), (-122.300, 37.905), (-122.220, 37.820),
                   (-122.120, 37.710), (-122.020, 37.600), (-121.930, 37.490)],
    rupture=dict(mag=7.0, rake=180.0, dip=90.0, usd=0.0, lsd=13.0, spacing=1.0),
    near_km=150.0,                           # facilities reported as "in the footprint"
    bay_wkt="POLYGON((-122.9 37.2,-121.5 37.2,-121.5 38.4,-122.9 38.4,-122.9 37.2))",
    min_block_km=25.0,
    rare_class_min_n=30,
    urls=dict(
        vs30="https://www.sciencebase.gov/catalog/file/get/5f44290282ce4c3d1222da63?f=__disk__01%2Ff6%2F97%2F01f6979edcca8e11d4df3eb5b08b6b11a43d3bb8",
        vs30map="https://www.sciencebase.gov/catalog/file/get/5a5fa029e4b06e28e9bfc43a?f=__disk__2b%2F15%2Fc5%2F2b15c57ae4bc092da9b3d8af7a2a32d8304f8a30",
        cms="https://data.cms.gov/provider-data/sites/default/files/resources/0970c657767e40f8521187d02ba8d0ed_1792181744/NH_ProviderInfo_Oct2026.csv",
        svi="https://svi.cdc.gov/Documents/Data/2022/csv/states/California.csv",
        tracts="https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_06_tract.zip",
        sgmc="https://www.sciencebase.gov/catalog/file/get/5888bf4fe4b05ccb964bab9d?f=__disk__24%2Ff6%2Fe1%2F24f6e139c181fd9fe43df2aaf7f50b1c5b3b6297",
    ),
)

CAT = ["lithology", "age"]                   # categorical feature names


def fetch(key: str, name: str) -> Path:
    """Download ``CFG['urls'][key]`` to ``DATA/name`` once and return the path."""
    p = DATA / name
    if not p.exists():
        print(f"downloading {key} -> {p}")
        urlretrieve(CFG["urls"][key], p)
    return p


def cached(name: str, build):
    """Build a DataFrame once and cache it as parquet under ``DATA/cache``."""
    import pandas as pd
    p = CACHE / f"{name}.parquet"
    legacy = DATA / f"{name}.parquet"          # layout used by the original notebook
    if not p.exists() and legacy.exists():
        p = legacy
    if p.exists():
        return pd.read_parquet(p)
    df = build()
    df.to_parquet(p)
    return df


def unzip(key: str, name: str, dest: str) -> Path:
    d = DATA / dest
    if not d.exists():
        zipfile.ZipFile(fetch(key, name)).extractall(d)
    return d


def sample_raster(path, lon, lat):
    """Sample a raster at lon/lat (EPSG:4326); nodata, zeros and negatives -> NaN."""
    import rasterio
    with rasterio.open(path) as src:
        vals = np.array([v[0] for v in src.sample(zip(lon, lat))], float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    vals[~np.isfinite(vals) | (vals <= 0)] = np.nan
    return vals


def to_points(df):
    import geopandas as gpd
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)


def col(df, *needles: str, required: bool = True):
    """First column whose upper-cased name contains every needle."""
    for c in df.columns:
        if all(n.upper() in c.upper() for n in needles):
            return c
    if required:
        raise KeyError(f"no column matching {needles}; have {list(df.columns)}")
    return None


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
