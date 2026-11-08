"""Projected ownership for Classic (starting model; retrained from contest results once slates
are ingested - the NBA lesson: mid-priced value plays get under-owned by naive models).

Per position, weight = projection^a * value^b (value = points per $1k); each position is
normalized to the field's roster spots for it (QB 100%, RB 200% + its FLEX share, WR 300% + FLEX,
TE 100% + FLEX, DST 100%; FLEX 100% split RB 35 / WR 50 / TE 15), capped, then tilted toward
cheaper players until the field's average lineup salary fits (target_salary).
Smaller slates concentrate ownership automatically (fewer players share the same totals).
Research overrides (Ownership) replace the model for those players.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .settings import get as S

SPOTS = {"QB": 100.0, "RB": 235.0, "WR": 350.0, "TE": 115.0, "DST": 100.0}
EXP = {"QB": (2.0, 1.5), "RB": (1.6, 2.0), "WR": (1.6, 1.8), "TE": (1.6, 1.8), "DST": (1.2, 1.5)}


def _normalize(w: np.ndarray, total: float, cap: float) -> np.ndarray:
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        return w
    own = w / w.sum() * total
    for _ in range(30):
        over = own > cap
        if not over.any():
            break
        excess = (own[over] - cap).sum()
        own[over] = cap
        free = ~over & (own > 0)
        if not free.any():
            break
        own[free] += excess * own[free] / own[free].sum()
    return own


def project(df: pd.DataFrame) -> np.ndarray:
    """df: Projection, Salary, pos -> ownership % per row."""
    proj = df["Projection"].clip(lower=0).to_numpy(float)
    sal = df["Salary"].to_numpy(float)
    pos = df["pos"].to_numpy(object)
    value = proj / np.maximum(sal / 1000, 0.2)
    cap = float(S("ownership.cap"))
    target = float(S("ownership.target_salary"))

    def own(lam):
        out = np.zeros(len(df))
        for p, total in SPOTS.items():
            m = pos == p
            if not m.any():
                continue
            a, b = EXP[p]
            w = proj[m] ** a * np.minimum(value[m], 6) ** b * np.exp(-lam * sal[m] / 1000)
            out[m] = _normalize(w, total, cap if p != "DST" else min(cap, 40))
        return out

    o = own(0.0)
    if (o * sal).sum() / 100 > target:
        lo, hi = 0.0, 2.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if (own(mid) * sal).sum() / 100 > target:
                lo = mid
            else:
                hi = mid
        o = own(hi)
    return np.maximum(o, (proj > 0.5) * 0.1)
