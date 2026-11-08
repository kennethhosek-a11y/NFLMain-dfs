"""Synthetic 3-game Classic slate (offline): two 1:00 games and one 4:25 game."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflmain import project
from nflmain.names import norm

GAMES = [("AAA", "BBB", "01:00PM", 47.0, -3.0), ("CCC", "DDD", "01:00PM", 44.0, 2.5),
         ("EEE", "FFF", "04:25PM", 50.0, -6.5)]
DAY = "2026-11-08"
SHAPE = [("QB", 6500, 0, 4), ("RB", 7800, 3, 17), ("RB", 5000, 2, 7), ("RB", 4000, 1, 3),
         ("WR", 8000, 9, 0), ("WR", 6200, 7, 0), ("WR", 4500, 5, 0), ("WR", 3500, 3, 0),
         ("TE", 5500, 6, 0), ("TE", 2800, 2, 0)]


def roster():
    out = {}
    for a, h, *_ in GAMES:
        for t in (a, h):
            out[t] = [(f"{t.title()} {p}{k}", p, s + 100 * (ord(t[0]) - 65), tg, c)
                      for k, (p, s, tg, c) in enumerate(SHAPE)]
    return out


def salary_csv() -> str:
    rows = ["Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame,Status"]
    pid = 40000000
    for a, h, kick, *_ in GAMES:
        for t in (a, h):
            for name, pos, sal, *_ in roster()[t] + [(f"{t.title()}s", "DST", 3000 + 50 * (ord(t[0]) - 65), 0, 0)]:
                pid += 1
                rp = pos if pos in ("QB", "DST") else f"{pos}/FLEX"
                rows.append(f"{pos},{name} ({pid}),{name},{pid},{rp},{sal},{a}@{h} 11/08/2026 {kick} ET,{t},10.0,")
    return "\n".join(rows) + "\n"


def weekly_stats() -> pd.DataFrame:
    rows = []
    opp = {}
    for a, h, *_ in GAMES:
        opp[a], opp[h] = h, a
    for w in range(1, 7):
        for t, plist in roster().items():
            for name, pos, sal, tgt, car in plist:
                r = dict(player_id=f"id_{name}", player_name=name, player_display_name=name, position=pos,
                         season=2026, week=w, season_type="REG", game_id=f"g{w}_{min(t, opp[t])}", team=t,
                         opponent_team=opp[t])
                rec = int(round(tgt * 0.68))
                qb = pos == "QB" and name.endswith("0")
                r.update(attempts=33 if qb else 0, completions=22 if qb else 0, passing_yards=240 if qb else 0,
                         passing_tds=1.5 if qb else 0, passing_interceptions=0.7 if qb else 0,
                         sacks_suffered=2 if qb else 0, carries=car, rushing_yards=car * 4.2,
                         rushing_tds=0.3 * (car > 10), targets=tgt, receptions=rec, receiving_yards=rec * 11,
                         receiving_tds=0.3 * (tgt > 5), receiving_air_yards=tgt * 8, fumbles_lost_total=0.05,
                         def_sacks=0.4)
                rows.append(r)
    return pd.DataFrame(rows).assign(name_key=lambda d: d["player_display_name"].map(norm))


def data() -> dict:
    s = weekly_stats()
    return {"stats": s, "snaps": None, "rz": None, "th": project.team_history(s),
            "use": project.player_usage(s, None, None)}


def lines(_d=None):
    out = []
    for a, h, kick, total, sp in GAMES:
        hh = int(kick[:2]) + 12
        out.append({"away": a, "home": h, "kickoff": f"{DAY}T{hh:02d}:{kick[3:5]}:00-05:00", "total": total,
                    "spread_home": sp, "line_source": "test", "indoor": True})
    return out


def patch_network(monkeypatch, tmp_path):
    """No network: ESPN lines/weather/injuries are faked; overrides/state/output go to tmp."""
    from nflmain import overrides, pipeline, slate, sources
    monkeypatch.setattr(sources, "espn_games", lambda d: lines())
    monkeypatch.setattr(sources, "weather", lambda *a, **k: {"indoor": True, "wind": 0, "precip": 0})
    monkeypatch.setattr(overrides, "DIR", tmp_path / "overrides")
    monkeypatch.setattr(pipeline, "OUT", tmp_path / "output")
    monkeypatch.setattr(pipeline, "HIST", tmp_path / "history")
    monkeypatch.setattr(slate, "STATE", tmp_path / "state")
    empty = pd.DataFrame(columns=["name", "team_name", "team", "pos", "espn_status", "comment", "updated",
                                  "name_key", "status"])
    return empty
