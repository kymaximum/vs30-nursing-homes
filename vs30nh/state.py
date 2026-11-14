"""Stage-to-stage state, persisted as joblib pickles under ``data/cache``."""
from __future__ import annotations

import joblib

from .config import CACHE


def save(name: str, obj) -> None:
    joblib.dump(obj, CACHE / f"{name}.joblib")


def load(name: str):
    p = CACHE / f"{name}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run the earlier stage first (see scripts/run_all.py)")
    return joblib.load(p)
