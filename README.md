# Where the ground is least known

Data-driven V<sub>S30</sub> prediction and its propagation to scenario shaking at
California skilled nursing facilities. Final project for EPS 55 (Introduction to
Earthquake Seismology), Harvard College, fall 2026.

## What this does

Ground motion at a site depends as much on the shallow subsurface as on the
earthquake, and the standard descriptor of that subsurface, V<sub>S30</sub>, is
measured at only a few thousand places and inferred everywhere else. This is a
deterministic seismic hazard analysis (Baker 2021) for a HayWired M7.0 Hayward
Fault scenario at every CMS-listed nursing home in California, with the site
term treated as the uncertain input: a gradient-boosted model predicts ln
V<sub>S30</sub> from open terrain and geologic-map features, is evaluated against
the topographic-slope proxy and the operational California map under spatially
blocked cross-validation, is wrapped in conformal prediction intervals, and both
estimates are propagated through the Boore et al. (2014) ground-motion model to
give median and 84th-percentile shaking at each facility.

The short version of what it finds: the learned model matches the operational
map in accuracy and does not beat its published uncertainty; in the Bay Area
footprint it raises predicted PGA at three facilities in four, resolves a
change at one in three, and resolves nothing near the fault, where nonlinear
soil response makes the choice of V<sub>S30</sub> map nearly irrelevant;
facilities are sited on common ground, not uncertain ground; and facilities in
more socially vulnerable tracts are farther from the measurements that anchor
every V<sub>S30</sub> map, though the model's own uncertainty does not register
that gap.

## Layout

```
vs30nh/        library code, one module per stage
  config.py    configuration, paths, download/cache helpers
  data.py      measured Vs30 labels, CMS facilities, background points
  features.py  3DEP terrain stencil, SGMC lithology/age
  folds.py     empirical variogram, spatial blocks, Bay Area holdout
  baselines.py slope proxy and the operational Wills/Thompson map
  model.py     LightGBM, nested tuning, CV, ablation, conformal intervals, SHAP
  scenario.py  Hayward rupture surface, BSSA14 median and sigma, DSHA tables
  siting.py    siting test and SVI-stratified summaries
  figures.py   the six paper figures
scripts/       numbered stages; run_all.py runs them in order
outputs/       figures and tables (tracked)
notebooks/     run_pipeline.ipynb (runs the stages and shows results); v4_vs30.ipynb (original)
tests/         offline smoke tests
```

## Running it

```
pip install -r requirements.txt
python scripts/run_all.py
```

or open `notebooks/run_pipeline.ipynb`, which runs the same stages and displays
the tables and figures. If you have a `data/` folder from an earlier notebook
run, copy it to the repository root; its `*.parquet` caches are picked up
as is and nothing is re-downloaded.

The first run downloads everything it needs (USGS V<sub>S30</sub> compilation,
3DEP elevation by point query, USGS State Geologic Map Compilation, the
Wills/Thompson V<sub>S30</sub> grid and its SD grid, CMS Provider Information,
CDC/ATSDR SVI, Census tracts) into `data/`, which is git-ignored. The 3DEP
queries are the slow part; every intermediate table is cached, so later runs
and the later stages are offline. Each stage writes its state to
`data/cache/stageN.joblib`, so a single stage can be re-run on its own.

All inputs are public and need no account. Nothing in `data/` is redistributed
here.

## Reproducibility notes

Fold assignment and the variogram subsample use `numpy.random.default_rng(11)`
in the order they appear in `scripts/01_features.py`; the bootstrap in the
siting test uses a fresh generator with the same seed, so its confidence
intervals differ from the notebook's in the third decimal. Everything else is
deterministic given the cached data.

The Hayward rupture trace is six hard-coded vertices (72 km) rather than a
merge of Quaternary Fault database segments, which do not snap and collapse to
a short fragment under `linemerge`. `tests/test_smoke.py` checks the trace
length.

## Citation

See `CITATION.cff`.
