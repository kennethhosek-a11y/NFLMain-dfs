"""Live DraftKings points for games that have started (ESPN), for late swap and standings.

- game_status(day): every game on the date with its state (pre / in / post), score, and how much
  of regulation is left (frac_left: 1 before kickoff, 0 when final, 0.02 in overtime).
- live_points(day): one row per player (and one per defense) with DK Classic points so far.
- condition(fp, pool, live, started): replaces the simulated points of players whose game has
  started with what has actually happened, plus the remaining share of their simulated points.

ESPN's box score has no positions, so pos_hint is a guess from the stat category (passing -> QB,
otherwise "" for skill players; DST for defenses). Two-point conversions aren't in the box score
and are left out (worth 2 points, rare). Any network or parsing problem returns an empty frame.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from . import scoring, sources
from .names import norm
from .teams import canon

COLS = ["name", "name_key", "team", "pos_hint", "dk_pts", "espn_id", "state", "frac_left"]


def _num(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _frac_left(state: str, period: int, clock: str) -> float:
    """Share of regulation (4 x 15 min) still to play."""
    if state == "pre":
        return 1.0
    if state == "post":
        return 0.0
    if period > 4:
        return 0.02
    try:
        m, s = str(clock).split(":")
        secs = int(m) * 60 + int(float(s))
    except ValueError:
        secs = 0
    left = (4 - max(period, 1)) * 900 + secs
    return float(np.clip(left / 3600, 0.0, 1.0))


def game_status(day: date) -> list[dict]:
    r = sources._get(f"{sources.ESPN}/scoreboard", params={"dates": f"{day:%Y%m%d}"}, timeout=30)
    out = []
    for e in r.json().get("events", []):
        c = e["competitions"][0]
        st = e.get("status", {})
        state = st.get("type", {}).get("state", "pre")
        period = int(st.get("period") or 0)
        clock = st.get("displayClock", "")
        g = {"espn_id": str(e["id"]), "state": state, "period": period, "clock": clock,
             "kickoff": datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(sources.ET).isoformat(),
             "frac_left": _frac_left(state, period, clock)}
        for t in c["competitors"]:
            side = t["homeAway"]
            g[side] = canon(t["team"]["abbreviation"])
            g[f"{side}_score"] = int(_num(t.get("score")))
        out.append(g)
    return out


def _stat_rows(box: dict) -> dict:
    """{team: {category: [(name, {key: value})]}} from the summary's boxscore.players."""
    out = {}
    for t in box.get("players", []):
        team = canon(t["team"]["abbreviation"])
        cats = out.setdefault(team, {})
        for s in t.get("statistics", []):
            keys = s.get("keys", [])
            rows = []
            for a in s.get("athletes", []):
                vals = {}
                for k, v in zip(keys, a.get("stats", [])):
                    if "/" in k and "/" in str(v):            # "completions/passingAttempts": "10/21"
                        for kk, vv in zip(k.split("/"), str(v).split("/")):
                            vals[kk] = _num(vv)
                    elif "-" in k and "-" in str(v):          # "sacks-sackYardsLost": "3-15"
                        for kk, vv in zip(k.split("-"), str(v).split("-")):
                            vals[kk] = _num(vv)
                    else:
                        vals[k] = _num(v)
                rows.append((a["athlete"].get("displayName", ""), vals))
            cats[s.get("name", "")] = rows
    return out


def _game_points(summary: dict, g: dict) -> list[dict]:
    stats = _stat_rows(summary.get("boxscore", {}))
    rows = []
    for team, cats in stats.items():
        opp = g["home"] if team == g["away"] else g["away"]
        players: dict[str, dict] = {}

        def add(name, key, v):
            players.setdefault(name, {})[key] = players.get(name, {}).get(key, 0.0) + v
        for name, v in cats.get("passing", []):
            add(name, "pass_yd", v.get("passingYards", 0)); add(name, "pass_td", v.get("passingTouchdowns", 0))
            add(name, "ints", v.get("interceptions", 0)); add(name, "_qb", 1)
        for name, v in cats.get("rushing", []):
            add(name, "rush_yd", v.get("rushingYards", 0)); add(name, "rush_td", v.get("rushingTouchdowns", 0))
        for name, v in cats.get("receiving", []):
            add(name, "rec", v.get("receptions", 0)); add(name, "rec_yd", v.get("receivingYards", 0))
            add(name, "rec_td", v.get("receivingTouchdowns", 0))
        for name, v in cats.get("fumbles", []):
            add(name, "fum_lost", v.get("fumblesLost", 0))
        ret_td = 0.0
        for cat, k in (("kickReturns", "kickReturnTouchdowns"), ("puntReturns", "puntReturnTouchdowns")):
            for name, v in cats.get(cat, []):
                add(name, "ret_td", v.get(k, 0))
                ret_td += v.get(k, 0)
        for name, p in players.items():
            qb = p.pop("_qb", 0) and p.get("pass_yd", 0) + p.get("pass_td", 0) > 0
            if not any(p.get(k, 0) for k in p):
                pts = 0.0
            else:
                pts = float(scoring.offense(**{k: p.get(k, 0) for k in
                                               ("pass_yd", "pass_td", "ints", "rush_yd", "rush_td", "rec",
                                                "rec_yd", "rec_td", "fum_lost", "ret_td")}))
            rows.append({"name": name, "name_key": norm(name), "team": team, "pos_hint": "QB" if qb else "",
                         "dk_pts": pts})
        # defense: takeaways come from the OTHER team's box (their INTs thrown, fumbles lost, sacks taken)
        o = stats.get(opp, {})
        sacks = sum(v.get("sacks", 0) for _, v in o.get("passing", []))
        ints = sum(v.get("interceptions", 0) for _, v in o.get("passing", []))
        fum = sum(v.get("fumblesLost", 0) for _, v in o.get("fumbles", []))
        d_td = max(sum(v.get("defensiveTouchdowns", 0) for _, v in cats.get("defensive", [])),
                   sum(v.get("interceptionTouchdowns", 0) for _, v in cats.get("interceptions", [])))
        safeties = sum(1 for sp in summary.get("scoringPlays", [])
                       if "safety" in str(sp.get("type", {}).get("text", "")).lower()
                       and canon(sp.get("team", {}).get("abbreviation", "")) == team)
        # points allowed: the opponent's score (DK excludes TDs the offense gave up on turnovers;
        # close enough live)
        pa = g["home_score"] if team == g["away"] else g["away_score"]
        dst = scoring.dst(sacks=sacks, ints=ints, fum_rec=fum, td=d_td + ret_td, safeties=safeties, pa=pa)
        if g["state"] == "in":                       # points-allowed tier isn't settled until the end
            dst = dst - scoring.points_allowed(pa) + scoring.points_allowed(pa) * (1 - g["frac_left"])
        rows.append({"name": f"{team} DST", "name_key": f"{team.lower()} dst", "team": team, "pos_hint": "DST",
                     "dk_pts": float(dst)})
    for r in rows:
        r.update(espn_id=g["espn_id"], state=g["state"], frac_left=g["frac_left"])
    return rows


def live_points(day: date) -> pd.DataFrame:
    try:
        games = [g for g in game_status(day) if g["state"] in ("in", "post")]
        rows = []
        for g in games:
            s = sources._get(f"{sources.ESPN}/summary", params={"event": g["espn_id"]}, timeout=30).json()
            rows += _game_points(s, g)
        return pd.DataFrame(rows, columns=COLS)
    except Exception as e:  # noqa: BLE001
        print(f"[live] ESPN live scores unavailable ({e.__class__.__name__}: {e})")
        return pd.DataFrame(columns=COLS)


def condition(fp: np.ndarray, pool: pd.DataFrame, live: pd.DataFrame, started: np.ndarray) -> np.ndarray:
    """Started players: final games -> actual points every sim; in progress -> actual + the
    remaining share of the simulated points. No live row: final -> 0 (didn't play)."""
    out = np.array(fp, dtype=np.float32, copy=True)
    if live is None or not len(live):
        live = pd.DataFrame(columns=COLS)
    by_kt = {(k, t): r for k, t, r in zip(live["name_key"], live["team"], live.to_dict("records"))}
    by_k = {}
    for k, r in zip(live["name_key"], live.to_dict("records")):
        by_k.setdefault(k, r)
    # game state per team (for players with no live row)
    team_state = {}
    for r in live.to_dict("records"):
        team_state.setdefault(r["team"], (r["state"], float(r["frac_left"])))
    keys = pool["name_key"].to_numpy()
    teams = pool["team"].to_numpy()
    for i in np.flatnonzero(np.asarray(started, dtype=bool)):
        r = by_kt.get((keys[i], teams[i])) or by_k.get(keys[i])
        if r is not None:
            if r["state"] == "post":
                out[i] = r["dk_pts"]
            else:
                out[i] = r["dk_pts"] + out[i] * float(r["frac_left"])
            continue
        state, left = team_state.get(teams[i], ("in", 1.0))
        out[i] = 0.0 if state == "post" else out[i] * left
    return out
