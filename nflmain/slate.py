"""DraftKings NFL Classic: lobby, slates (draft groups), contests and salary files.

Classic roster: QB, RB, RB, WR, WR, WR, TE, FLEX (RB/WR/TE), DST; $50,000 cap; players from at
least 2 games. Each player locks at his own game's kickoff.

Slate types (each contest in the DKEntries file maps to its DK draft group = its slate):
  early        1:00 games only               -> one lock window, no late swap
  main         1:00 + 4:00 games             -> late swap on the 4:00 players
  afternoon    4:00 games only (incl. Turbo) -> small slate, heavier stacks, light/no gate
  thanksgiving Thursday Thanksgiving slate   -> three windows (~12:30, ~4:30, ~8:20)
  other        primetime / Sun-Mon / Mon-Thu -> not built by this system (reported)
"""
from __future__ import annotations

import io
import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .names import norm
from .teams import canon

UA = {"User-Agent": "nflmain/1.0"}
CLASSIC = 21          # DK contestTypeId: NFL Classic
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "state"
CAP = 50000
SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
SLOT_POS = [{"QB"}, {"RB"}, {"RB"}, {"WR"}, {"WR"}, {"WR"}, {"TE"}, {"RB", "WR", "TE"}, {"DST"}]


def lobby() -> dict:
    r = requests.get("https://www.draftkings.com/lobby/getcontests?sport=NFL", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def classify(suffix: str | None, kickoffs: list[datetime]) -> str:
    """Slate type from DK's draft-group suffix and the games' kickoff times (ET)."""
    s = (suffix or "").lower()
    if any(k in s for k in ("primetime", "sun-mon", "mon-thu", "showdown", "night")):
        return "other"
    if not kickoffs:
        return "other"
    d = kickoffs[0].date()
    if d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28:      # Thanksgiving Thursday
        return "thanksgiving"
    if "early" in s:
        return "early"
    if "afternoon" in s or "turbo" in s and min(k.hour for k in kickoffs) >= 15:
        return "afternoon"
    hours = sorted({k.hour for k in kickoffs})
    if d.weekday() != 6:                    # not Sunday and not Thanksgiving: TNF/MNF/Saturday stacks
        return "other" if len(kickoffs) <= 2 else "main"
    early = any(h < 15 for h in hours)
    late = any(15 <= h < 19 for h in hours)
    night = any(h >= 19 for h in hours)
    if night and len(set(kickoffs)) <= 2 and not early:
        return "other"
    if early and late:
        return "main"
    if early:
        return "early"
    if late:
        return "afternoon"
    return "other"


def classic_groups(lb: dict | None = None, day: str | None = None) -> list[dict]:
    """Every NFL Classic draft group in the lobby: id, start, games, suffix, slate type, contests."""
    lb = lb or lobby()
    n_contests = {}
    for c in lb.get("Contests", []):
        n_contests[c.get("dg")] = n_contests.get(c.get("dg"), 0) + 1
    out = []
    for g in lb.get("DraftGroups", []):
        if g.get("ContestTypeId") != CLASSIC:
            continue
        start = datetime.fromisoformat(g["StartDateEst"].split(".")[0]).replace(tzinfo=ET)
        if day and f"{start:%Y-%m-%d}" != day:
            continue
        suffix = (g.get("ContestStartTimeSuffix") or "").strip()
        games = g.get("GameCount") or 0
        # the lobby has no per-game times: the start and the suffix decide; refined from the salary file
        kick = [start] if "early" in suffix.lower() or "afternoon" in suffix.lower() else \
            [start, start.replace(hour=16, minute=25)] if start.hour < 15 and games >= 8 and start.weekday() == 6 else [start]
        out.append({"draft_group": g["DraftGroupId"], "start": start.isoformat(), "date": f"{start:%Y-%m-%d}",
                    "games": games, "suffix": suffix or "(Main)", "featured": g.get("DraftGroupTag") == "Featured",
                    "slate_type": classify(suffix, kick), "contests": n_contests.get(g["DraftGroupId"], 0)})
    return sorted(out, key=lambda x: (x["start"], -x["games"]))


def contests_for(draft_group: int, lb: dict | None = None) -> pd.DataFrame:
    lb = lb or lobby()
    rows = [{"Contest ID": str(c["id"]), "Contest Name": c["n"], "Entry Fee": c.get("a"),
             "Max Entries": c.get("m"), "Max Per User": c.get("mec"), "Prizes": c.get("po")}
            for c in lb.get("Contests", []) if c.get("dg") == draft_group]
    return pd.DataFrame(rows).sort_values("Prizes", ascending=False) if rows else pd.DataFrame(rows)


def fetch_csv(draft_group: int) -> str:
    r = requests.get("https://www.draftkings.com/lineup/getavailableplayerscsv",
                     params={"contestTypeId": CLASSIC, "draftGroupId": draft_group}, headers=UA, timeout=30)
    r.raise_for_status()
    return r.content.decode("utf-8-sig")


def read_salaries(src) -> pd.DataFrame:
    """A DKSalaries.csv (path or text). Also handles a DKEntries.csv (salary block on the right)."""
    text = str(src)
    if "\n" not in text and Path(text).exists():
        text = Path(text).read_text(encoding="utf-8-sig")
    text = text.lstrip("﻿")
    first = text.splitlines()[0] if text else ""
    if first.startswith("Position,Name + ID"):
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    else:
        import csv
        rows = list(csv.reader(io.StringIO(text)))
        df = None
        for r, row in enumerate(rows):
            if "Name + ID" in row:
                c0 = row.index("Position")
                block = [x[c0:c0 + 9] for x in rows[r + 1:] if len(x) >= c0 + 9]
                df = pd.DataFrame(block, columns=["Position", "Name + ID", "Name", "ID", "Roster Position",
                                                  "Salary", "Game Info", "TeamAbbrev", "AvgPointsPerGame"])
                df = df[df["Name"] != ""].reset_index(drop=True)
                break
        if df is None:
            raise ValueError("doesn't look like a DraftKings salary file")
    if "Status" not in df:
        df["Status"] = ""
    return df


def kickoff_of(game_info: str) -> datetime | None:
    """'DET@MIA 11/08/2026 01:00PM ET' -> datetime ET. Started games show no time on DK."""
    txt = str(game_info)
    m = re.search(r"(\d{2}/\d{2}/\d{4}) (\d{1,2}:\d{2}[AP]M)", txt)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%m/%d/%Y %I:%M%p").replace(tzinfo=ET)


GAME_RE = r"^\w+@\w+ \d\d/\d\d/\d{4} \d\d:\d\d[AP]M"
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "state"


def _patch_game_info(df: pd.DataFrame) -> pd.DataFrame:
    """Once a game starts, DK's salary file shows 'In-Progress' / 'Final' instead of
    'AWAY@HOME date time ET' for its players (found live 11/08/2026, 1:00 games). Every good Game
    Info seen is remembered by player ID (data/state/game_info.json); started games are filled
    back in from it, or from the build's baseline files."""
    gi = df["Game Info"].astype(str)
    good = gi.str.match(GAME_RE)
    f = STATE_DIR / "game_info.json"
    try:
        known = json.loads(f.read_text()) if f.exists() else {}
    except Exception:  # noqa: BLE001
        known = {}
    if good.any():
        known.update(dict(zip(df.loc[good, "ID"].astype(str), gi[good])))
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(known))
        except OSError:
            pass
    if good.all():
        return df
    for b in sorted(STATE_DIR.glob("baseline_*.csv")):
        try:
            bd = pd.read_csv(b, dtype=str, usecols=["ID", "Game Info"])
        except Exception:  # noqa: BLE001
            continue
        bd = bd[bd["Game Info"].astype(str).str.match(GAME_RE)]
        for i, g in zip(bd["ID"], bd["Game Info"]):
            known.setdefault(str(i), g)
    df = df.copy()
    fill = df["ID"].astype(str).map(known)
    m = ~good & fill.notna()
    df.loc[m, "Game Info"] = fill[m]
    if (~good & fill.isna()).any():
        print(f"[slate] {int((~good & fill.isna()).sum())} players in started games with no saved game info")
    return df


def load_classic(src) -> pd.DataFrame:
    """One row per player: Name, ID, Position, Salary, team, opp, home, game, kickoff, status, name_key."""
    df = read_salaries(src)
    df["Salary"] = df["Salary"].astype(int)
    df = _patch_game_info(df)
    p = df.copy()
    p["team"] = p["TeamAbbrev"].map(canon)
    gm = p["Game Info"].str.extract(r"^(\w+)@(\w+)")
    away, home = gm[0].map(canon), gm[1].map(canon)
    p["opp"] = [h if t == a else a for t, a, h in zip(p["team"], away, home)]
    p["home"] = p["team"] == home
    p["game"] = away + "@" + home
    p["kickoff"] = p["Game Info"].map(lambda s: (kickoff_of(s).isoformat() if kickoff_of(s) else ""))
    p["status"] = p["Status"].str.upper().str.strip().replace(
        {"O": "OUT", "IR": "OUT", "PUP": "OUT", "NA": "OUT", "SUSP": "OUT"})
    p["name_key"] = p["Name"].map(norm)
    p.loc[p["Position"] == "DST", "name_key"] = p["team"].str.lower() + " dst"
    p["AvgPointsPerGame"] = pd.to_numeric(p["AvgPointsPerGame"], errors="coerce").fillna(0)
    return p.reset_index(drop=True)


def slate_type_of(pool: pd.DataFrame, suffix: str | None = None) -> str:
    kicks = sorted({datetime.fromisoformat(k) for k in pool["kickoff"] if k})
    return classify(suffix, kicks)


def windows(pool: pd.DataFrame) -> list[str]:
    """Distinct lock times (ET iso), earliest first."""
    return sorted({k for k in pool["kickoff"] if k})


# ---------------------------------------------------------------- pins (DK drops locked slates)
def _pin_file(day: str) -> Path:
    return STATE / f"slates_{day}.json"


def pins(day: str) -> dict:
    f = _pin_file(day)
    return json.loads(f.read_text()) if f.exists() else {"groups": {}, "contests": {}}


def pin(day: str, groups: list[dict] | None = None, contests: dict | None = None) -> None:
    """Remember draft groups and contest -> draft group for the day (the lobby drops them at lock)."""
    STATE.mkdir(parents=True, exist_ok=True)
    cur = pins(day)
    for g in groups or []:
        cur["groups"][str(g["draft_group"])] = g
    cur["contests"].update({str(k): int(v) for k, v in (contests or {}).items()})
    _pin_file(day).write_text(json.dumps(cur, indent=1))


def find_group(day: str, slate_type: str | None = None, draft_group: int | None = None) -> dict:
    """The draft group to build: by id, or the biggest one of a slate type (default main)."""
    pinned = pins(day)["groups"]
    if draft_group and str(draft_group) in pinned:
        return pinned[str(draft_group)]
    try:
        groups = classic_groups(day=day)
    except Exception:  # noqa: BLE001
        groups = list(pinned.values())
    if draft_group:
        groups = [g for g in groups if g["draft_group"] == int(draft_group)]
    else:
        want = slate_type or "main"
        groups = [g for g in groups if g["slate_type"] == want]
        if not groups:
            groups = [g for g in pinned.values() if g["slate_type"] == want]
    if not groups:
        raise RuntimeError(f"No DK Classic slate found for {slate_type or draft_group} on {day}")
    g = max(groups, key=lambda g: (g.get("featured", False), g["contests"], g["games"]))
    pin(day, [g])
    return g


def contest_groups(contest_ids: list[str], day: str) -> dict[str, int]:
    """Contest id -> draft group: pinned first, then the lobby, then DK's contest API."""
    known = pins(day)["contests"]
    out = {c: known[c] for c in contest_ids if c in known}
    need = [c for c in contest_ids if c not in out]
    if need:
        try:
            lb = lobby()
            for c in lb.get("Contests", []):
                if str(c.get("id")) in need:
                    out[str(c["id"])] = int(c["dg"])
        except Exception as e:  # noqa: BLE001
            print(f"[slate] lobby unavailable ({e})")
    for c in [c for c in contest_ids if c not in out]:
        try:
            from .contests import details
            d = details(c)
            if d.get("draftGroupId"):
                out[c] = int(d["draftGroupId"])
        except Exception as e:  # noqa: BLE001
            print(f"[slate] contest {c}: draft group unknown ({e})")
    pin(day, contests=out)
    return out


def eligibility(pos: np.ndarray) -> np.ndarray:
    """(n, 9) bool: can player i fill slot s."""
    pos = np.asarray(pos, dtype=object)
    return np.array([[p in sp for sp in SLOT_POS] for p in pos], dtype=bool).reshape(len(pos), len(SLOTS))
