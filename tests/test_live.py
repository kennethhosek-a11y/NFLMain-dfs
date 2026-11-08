"""Offline tests for nflmain.live (fake ESPN payloads shaped like the real ones)."""
from datetime import date

import numpy as np
import pandas as pd

from nflmain import live, sources

SCOREBOARD = {"events": [
    {"id": "1", "date": "2026-11-08T18:00Z",
     "status": {"period": 3, "displayClock": "7:30", "type": {"state": "in"}},
     "competitions": [{"competitors": [
         {"homeAway": "home", "score": "21", "team": {"abbreviation": "BUF"}},
         {"homeAway": "away", "score": "7", "team": {"abbreviation": "BAL"}}]}]},
    {"id": "2", "date": "2026-11-08T21:25Z",
     "status": {"period": 0, "displayClock": "0:00", "type": {"state": "pre"}},
     "competitions": [{"competitors": [
         {"homeAway": "home", "score": "0", "team": {"abbreviation": "SEA"}},
         {"homeAway": "away", "score": "0", "team": {"abbreviation": "ARI"}}]}]},
]}


def _cat(name, keys, rows):
    return {"name": name, "keys": keys,
            "athletes": [{"athlete": {"displayName": n}, "stats": s} for n, s in rows]}


PASS_K = ["completions/passingAttempts", "passingYards", "yardsPerPassAttempt", "passingTouchdowns",
          "interceptions", "sacks-sackYardsLost", "adjQBR", "QBRating"]
RUSH_K = ["rushingAttempts", "rushingYards", "yardsPerRushAttempt", "rushingTouchdowns", "longRushing"]
REC_K = ["receptions", "receivingYards", "yardsPerReception", "receivingTouchdowns", "longReception",
         "receivingTargets"]
FUM_K = ["fumbles", "fumblesLost", "fumblesRecovered"]
SUMMARY = {"boxscore": {"players": [
    {"team": {"abbreviation": "BUF"}, "statistics": [
        _cat("passing", PASS_K, [("Josh Allen", ["21/34", "310", "9.1", "2", "1", "1-9", "78", "96"])]),
        _cat("rushing", RUSH_K, [("Josh Allen", ["7", "30", "4.3", "1", "14"])]),
        _cat("receiving", REC_K, [("Dalton Kincaid", ["6", "105", "17.5", "1", "40", "8"])]),
    ]},
    {"team": {"abbreviation": "BAL"}, "statistics": [
        _cat("passing", PASS_K, [("Joe Fagnano", ["10/21", "91", "4.3", "0", "2", "3-15", "7", "40"])]),
        _cat("fumbles", FUM_K, [("Joe Fagnano", ["1", "1", "0"])]),
    ]},
]}, "scoringPlays": []}


class _Resp:
    def __init__(self, d):
        self.d = d

    def json(self):
        return self.d


def _fake_get(url, params=None, timeout=60, tries=3):
    return _Resp(SCOREBOARD if url.endswith("scoreboard") else SUMMARY)


def test_game_status(monkeypatch):
    monkeypatch.setattr(sources, "_get", _fake_get)
    g = live.game_status(date(2026, 11, 8))
    assert [x["state"] for x in g] == ["in", "pre"]
    assert g[0]["home"] == "BUF" and g[0]["away_score"] == 7
    assert abs(g[0]["frac_left"] - (900 + 450) / 3600) < 1e-9
    assert g[1]["frac_left"] == 1.0


def test_live_points(monkeypatch):
    monkeypatch.setattr(sources, "_get", _fake_get)
    df = live.live_points(date(2026, 11, 8))
    pts = dict(zip(df["name_key"], df["dk_pts"]))
    # 310*.04 + 2*4 - 1 + 3 (300 bonus) + 3.0 + 6
    assert abs(pts["josh allen"] - (12.4 + 8 - 1 + 3 + 3 + 6)) < 1e-6
    assert abs(pts["dalton kincaid"] - (6 + 10.5 + 6 + 3)) < 1e-6
    assert df.loc[df["name_key"] == "josh allen", "pos_hint"].iloc[0] == "QB"
    # BUF defense mid-game: 3 sacks + 2 INT*2 + 1 fumble*2 = 9, plus the PA tier scaled by time played
    frac = (900 + 450) / 3600
    assert abs(pts["buf dst"] - (9 + 4 * (1 - frac))) < 1e-6   # 7 allowed = the 7-13 tier (4 pts)
    assert set(df["espn_id"]) == {"1"}          # the pre-game isn't fetched


def test_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no network")
    monkeypatch.setattr(sources, "_get", boom)
    df = live.live_points(date(2026, 11, 8))
    assert list(df.columns) == live.COLS and df.empty


def test_condition():
    pool = pd.DataFrame({"name_key": ["a", "b", "c", "d", "e"], "team": ["BUF", "BUF", "KC", "KC", "SEA"],
                         "pos": ["QB", "WR", "RB", "WR", "WR"]})
    fp = np.full((5, 4), 10.0)
    lv = pd.DataFrame([
        {"name": "A", "name_key": "a", "team": "BUF", "pos_hint": "QB", "dk_pts": 12.0, "espn_id": "1",
         "state": "in", "frac_left": 0.25},
        {"name": "C", "name_key": "c", "team": "KC", "pos_hint": "", "dk_pts": 20.0, "espn_id": "3",
         "state": "post", "frac_left": 0.0},
    ], columns=live.COLS)
    started = np.array([True, True, True, True, False])
    out = live.condition(fp, pool, lv, started)
    assert np.allclose(out[0], 12 + 2.5)      # in progress: actual + remaining share
    assert np.allclose(out[1], 2.5)           # same game, no stats yet: remaining share only
    assert np.allclose(out[2], 20)            # final
    assert np.allclose(out[3], 0)             # final, never recorded a stat
    assert np.allclose(out[4], 10)            # not started: untouched
    assert np.allclose(fp, 10)                # input not modified
