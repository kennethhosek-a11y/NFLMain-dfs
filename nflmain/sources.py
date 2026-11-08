"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
Free data sources (shared by Showdown and, later, Classic).

- nflverse (GitHub releases): weekly player stats, snap counts, play-by-play (red-zone usage)
- ESPN: game lines (spread / total via the scoreboard), venue + weather, injury report
- National Weather Service (api.weather.gov): hourly wind / rain at outdoor stadiums

Downloads are cached under data/cache/ (not committed) and refreshed when older than max_age.
No sportsbook-props source is used here (see project.PROPS_ENABLED).
"""
from __future__ import annotations

import io
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .names import norm
from .teams import STADIUMS, canon

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "nflmain/1.0"}
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


def season_of(d: date) -> int:
    return d.year if d.month >= 3 else d.year - 1


def _get(url, params=None, timeout=60, tries=3) -> requests.Response:
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def _cached(name: str, url: str, max_age_h: float) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / name
    if f.exists() and time.time() - f.stat().st_mtime < max_age_h * 3600:
        return f
    try:
        r = _get(url, timeout=180)
        f.write_bytes(r.content)
    except Exception as e:  # noqa: BLE001
        if not f.exists():
            raise
        print(f"[data] couldn't refresh {name} ({e}); using the cached copy")
    return f


# ------------------------------------------------------------------ nflverse

def weekly_stats(season: int, max_age_h: float = 12) -> pd.DataFrame:
    f = _cached(f"stats_player_week_{season}.csv",
                f"{NFLVERSE}/stats_player/stats_player_week_{season}.csv", max_age_h)
    s = pd.read_csv(f, low_memory=False)
    s = s[s["season_type"] == "REG"].copy() if "season_type" in s else s
    s["team"] = s["team"].map(canon)
    s["opponent_team"] = s["opponent_team"].map(canon)
    s["name_key"] = s["player_display_name"].map(norm)
    return s


def snap_counts(season: int, max_age_h: float = 12) -> pd.DataFrame:
    f = _cached(f"snap_counts_{season}.csv", f"{NFLVERSE}/snap_counts/snap_counts_{season}.csv", max_age_h)
    s = pd.read_csv(f)
    s = s[s["game_type"] == "REG"].copy()
    s["team"] = s["team"].map(canon)
    s["name_key"] = s["player"].map(norm)
    return s


def redzone_usage(season: int, max_age_h: float = 12) -> pd.DataFrame:
    """Per player-game red-zone (inside the 20) targets and carries, plus team totals.
    Keyed by nflverse player id (matches weekly_stats.player_id)."""
    out = CACHE / f"rz_{season}.csv"
    f = _cached(f"pbp_{season}.parquet", f"{NFLVERSE}/pbp/play_by_play_{season}.parquet", max_age_h)
    if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
        return pd.read_csv(out)
    p = pd.read_parquet(f, columns=["game_id", "week", "season_type", "posteam", "yardline_100",
                                    "play_type", "receiver_player_id", "rusher_player_id",
                                    "two_point_attempt"])
    p = p[(p["season_type"] == "REG") & (p["yardline_100"] <= 20) & (p["two_point_attempt"] != 1)
          & p["play_type"].isin(["pass", "run"])]
    p["posteam"] = p["posteam"].map(canon)
    tgt = p[p["play_type"] == "pass"].groupby(["game_id", "week", "posteam", "receiver_player_id"]).size()
    car = p[p["play_type"] == "run"].groupby(["game_id", "week", "posteam", "rusher_player_id"]).size()
    tgt = tgt.rename("rz_tgt").reset_index().rename(columns={"receiver_player_id": "player_id"})
    car = car.rename("rz_car").reset_index().rename(columns={"rusher_player_id": "player_id"})
    rz = tgt.merge(car, on=["game_id", "week", "posteam", "player_id"], how="outer").fillna(0)
    team = p.groupby(["game_id", "posteam"])["play_type"].value_counts().unstack(fill_value=0)
    team = team.rename(columns={"pass": "team_rz_pass", "run": "team_rz_run"}).reset_index()
    rz = rz.merge(team, on=["game_id", "posteam"], how="left").rename(columns={"posteam": "team"})
    rz.to_csv(out, index=False)
    return rz


# ------------------------------------------------------------------ ESPN

def espn_games(d: date) -> list[dict]:
    """Games on an Eastern date: teams, kickoff (ET), spread (home view), total, venue, weather."""
    r = _get(f"{ESPN}/scoreboard", params={"dates": f"{d:%Y%m%d}"}, timeout=30)
    out = []
    for e in r.json().get("events", []):
        c = e["competitions"][0]
        teams = {t["homeAway"]: canon(t["team"]["abbreviation"]) for t in c["competitors"]}
        kick = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(ET)
        g = {"espn_id": e["id"], "away": teams["away"], "home": teams["home"],
             "kickoff": kick.isoformat(), "status": e.get("status", {}).get("type", {}).get("name", ""),
             "spread_home": None, "total": None, "line_source": "",
             "indoor": bool(c.get("venue", {}).get("indoor")),
             "venue": c.get("venue", {}).get("fullName", ""),
             "weather": (e.get("weather") or {}).get("displayValue", ""),
             "temp": (e.get("weather") or {}).get("temperature")}
        for o in c.get("odds", []) or []:
            if o.get("overUnder") is None:
                continue
            g["total"] = float(o["overUnder"])
            sp = abs(float(o.get("spread") or 0))
            home_fav = bool((o.get("homeTeamOdds") or {}).get("favorite"))
            g["spread_home"] = -sp if home_fav else sp
            g["line_source"] = f"ESPN ({(o.get('provider') or {}).get('name', '?')}): {o.get('details', '')}"
            break
        out.append(g)
    # ESPN "dates" is a UTC-ish window: keep games whose ET kickoff is on that date
    return [g for g in out if g["kickoff"][:10] == f"{d:%Y-%m-%d}"] or out


def espn_injuries() -> pd.DataFrame:
    r = _get(f"{ESPN}/injuries", timeout=30)
    rows = []
    names = {}
    for t in r.json().get("injuries", []):
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            team = a.get("team", {}).get("abbreviation") or names.get(t.get("displayName"))
            rows.append({"name": a.get("displayName", ""), "team_name": t.get("displayName", ""),
                         "team": canon(team) if team else "",
                         "pos": (a.get("position") or {}).get("abbreviation", ""),
                         "espn_status": i.get("status", ""), "comment": i.get("shortComment", ""),
                         "updated": i.get("date", "")})
    df = pd.DataFrame(rows, columns=["name", "team_name", "team", "pos", "espn_status", "comment", "updated"])
    df["name_key"] = df["name"].map(norm)
    df["status"] = df["espn_status"].map(_espn_status).fillna("")
    return df


def _espn_status(s: str) -> str:
    s = str(s).lower()
    if s.startswith("out") or "reserve" in s or "suspend" in s or "pup" in s:
        return "OUT"
    if s.startswith("doubt"):
        return "D"
    if s.startswith("question"):
        return "Q"
    return ""


# ------------------------------------------------------------------ weather (NWS)

def weather(home: str, kickoff_iso: str, indoor: bool) -> dict:
    """Wind (mph), precip chance (%) and temp at kickoff for outdoor games; {} if unknown."""
    if indoor or home not in STADIUMS or STADIUMS[home][2]:
        return {"indoor": True, "wind": 0.0, "precip": 0.0, "temp": None, "forecast": "Indoors"}
    lat, lon, _ = STADIUMS[home]
    try:
        pts = _get(f"https://api.weather.gov/points/{lat},{lon}", timeout=20).json()
        hourly = _get(pts["properties"]["forecastHourly"], timeout=20).json()["properties"]["periods"]
    except Exception as e:  # noqa: BLE001
        print(f"[weather] NWS forecast unavailable ({e})")
        return {}
    kick = datetime.fromisoformat(kickoff_iso)
    best = None
    for p in hourly:
        t = datetime.fromisoformat(p["startTime"])
        if t <= kick + timedelta(hours=1) and (best is None or t > datetime.fromisoformat(best["startTime"])):
            best = p
    if best is None:
        return {}
    wind = float(str(best.get("windSpeed", "0")).split()[0].split("to")[0] or 0)
    return {"indoor": False, "wind": wind, "temp": best.get("temperature"),
            "precip": float((best.get("probabilityOfPrecipitation") or {}).get("value") or 0),
            "forecast": best.get("shortForecast", "")}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
