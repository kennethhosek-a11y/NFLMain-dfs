"""Offline tests for the history database helpers and the slate-size backtest pieces."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.backtest import slate_size as B
from research.history import build_db as H


def test_slate_tags():
    assert H.slate_tag("Sunday", "13:00", "2026-11-08") == "early"
    assert H.slate_tag("Sunday", "16:25", "2026-11-08") == "afternoon"
    assert H.slate_tag("Sunday", "09:30", "2026-11-08") == "morning"
    assert H.slate_tag("Sunday", "20:20", "2026-11-08") == "night"
    assert H.slate_tag("Thursday", "12:30", "2026-11-26") == "thanksgiving"
    assert H.slate_tag("Thursday", "20:20", "2026-11-26") == "thanksgiving"
    assert H.slate_tag("Thursday", "20:15", "2026-11-05") == "thu"
    assert H.slate_tag("Monday", "20:15", "2026-11-09") == "mon"


def test_dk_points_from_nflverse_columns():
    row = {c: 0 for c in ["passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds",
                          "receptions", "receiving_yards", "receiving_tds", "fumbles_lost_total",
                          "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
                          "special_teams_tds", "fumble_recovery_tds"]}
    qb = dict(row, passing_yards=305, passing_tds=3, passing_interceptions=1, rushing_yards=20)
    wr = dict(row, receptions=9, receiving_yards=112, receiving_tds=1, receiving_2pt_conversions=1)
    pts = H.dk_points(pd.DataFrame([qb, wr]))
    assert pts.iloc[0] == pytest.approx(12.2 + 12 - 1 + 3 + 2.0)
    assert pts.iloc[1] == pytest.approx(9 + 11.2 + 6 + 3 + 2)


def test_synthetic_salaries_are_dk_shaped():
    real = pd.DataFrame({"pos": ["WR"] * 200, "trail": np.linspace(2, 25, 200), "implied": 22.0,
                         "salary": np.linspace(3000, 9500, 200)})
    fit = H.fit_pricing(real)
    s = H.synth_salary(pd.Series([2.0, 12.0, 30.0, np.nan]), pd.Series(["WR"] * 4), fit, implied=[22, 22, 22, 22])
    assert (s % 100 == 0).all() and s.between(3000, 10000).all()
    assert s.iloc[0] < s.iloc[1] < s.iloc[2] or s.iloc[2] == 10000


def test_gate_percentiles_and_weights():
    x = pd.DataFrame({"pos": ["WR"] * 10, "proj": np.linspace(3, 20, 10), "salary": [5000] * 10})
    pct = B.gate_pct(x, 150.0)
    assert pct.min() == 0 and pct.max() < 1 and np.all(np.diff(pct) >= 0)
    assert B.weight(2026) > B.weight(2025) > B.weight(2023) == 1.0


def test_portfolio_pick_respects_cap():
    rng = np.random.default_rng(0)
    lus = np.array([np.sort(rng.choice(30, 9, replace=False)) for _ in range(200)])
    sc = rng.random((200, 50))
    ch = B.pick(lus, sc, 20, 0.5)
    assert len(ch) == 20
    assert np.bincount(lus[ch].ravel()).max() <= 10
