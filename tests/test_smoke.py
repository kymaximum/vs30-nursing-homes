"""Offline checks that need no downloads: config sanity and the rupture trace length."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

from vs30nh.config import CFG


def test_hayward_trace_length_km():
    pts = CFG["hayward_trace"]
    L = 0.0
    for (a, b), (c, d) in zip(pts, pts[1:]):
        L += math.hypot((c - a) * 111.32 * math.cos(math.radians((b + d) / 2)), (d - b) * 111.32)
    assert 70 < L < 95, L


def test_trace_runs_nw_to_se():
    lats = [p[1] for p in CFG["hayward_trace"]]
    assert lats == sorted(lats, reverse=True)


def test_config_keys():
    for k in ("crs_m", "scales_m", "n_folds", "alpha", "imts", "rupture", "near_km", "bay_wkt"):
        assert k in CFG
