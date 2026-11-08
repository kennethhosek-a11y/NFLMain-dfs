"""DraftKings contest downloads ("Export Lineups to CSV" / contest-standings-<id>.zip): saved for good,
plus the field's habits measured from them.

Ownership recommendation #1 (research/ownership): DK keeps these files for only 10 days. Every
Sunday night Kenny downloads the file for every main-slate contest we entered, plus the Milly
Maker, and uploads it to the weekly chat. `python -m nflmain results <files...>` then:
  - keeps the raw file:             data/contests/<date>/<contest id>.csv.gz (every entry's full lineup)
  - keeps each player's %Drafted:   data/contests/<date>/<contest id>_ownership.csv
  - measures the field's habits:    data/contests/field_habits.csv (one row per contest): QB stack
    rate, double / triple stacks, bring-backs, FLEX position split, salary used, duplicate lineups,
    top-owned players - the numbers the ownership model and simulated field are fitted to.

A DK contest file has columns: Rank, EntryId, EntryName, TimeRemaining, Points, Lineup, (blank),
Player, Roster Position, %Drafted, FPTS. Lineup is one text field such as
"DST Bills FLEX Jahmyr Gibbs QB Josh Allen RB ... TE ... WR ... WR ... WR ...".
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .names import norm

ROOT = Path(__file__).resolve().parent.parent
CONTESTS = ROOT / "data" / "contests"
SLOTS = ("QB", "RB", "WR", "TE", "FLEX", "DST")
_TOKEN = re.compile(r"(?:^|\s)(QB|RB|WR|TE|FLEX|DST)\s")


def read_download(path) -> tuple[str, pd.DataFrame]:
    """(contest id, raw table) from a DK contest download (.csv or .zip)."""
    path = Path(path)
    m = re.search(r"(\d{6,})", path.name)
    cid = m.group(1) if m else path.stem
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            text = z.read(name).decode("utf-8-sig")
            m = re.search(r"(\d{6,})", name) or m
            cid = m.group(1) if m else cid
    else:
        text = path.read_text(encoding="utf-8-sig")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if "Lineup" not in df or "EntryId" not in df:
        raise ValueError(f"{path.name}: not a DraftKings contest download (no Lineup / EntryId columns)")
    return cid, df


def parse_lineup(text: str) -> list[tuple[str, str]]:
    """'QB Josh Allen RB A B RB C D ...' -> [(slot, name), ...] (empty if the entry was blank)."""
    text = " " + str(text).strip() + " "
    hits = list(_TOKEN.finditer(text))
    out = []
    for k, h in enumerate(hits):
        end = hits[k + 1].start() if k + 1 < len(hits) else len(text)
        out.append((h.group(1), text[h.end():end].strip()))
    return out


def ownership_table(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's %Drafted and actual points (DK lists them on the right of the file)."""
    if "Player" not in df:
        return pd.DataFrame(columns=["Player", "pos", "own", "fpts"])
    o = df[df["Player"].astype(str).str.strip() != ""][["Player", "Roster Position", "%Drafted", "FPTS"]].copy()
    o.columns = ["Player", "pos", "own", "fpts"]
    o["own"] = pd.to_numeric(o["own"].astype(str).str.rstrip("%"), errors="coerce")
    o["fpts"] = pd.to_numeric(o["fpts"], errors="coerce")
    o["name_key"] = o["Player"].map(norm)
    return o.drop_duplicates("Player").sort_values("own", ascending=False).reset_index(drop=True)


def field_habits(df: pd.DataFrame, own: pd.DataFrame, salaries: pd.DataFrame | None = None) -> dict:
    """How this contest's field built lineups. `salaries` (name_key, team, pos, salary) is the
    slate's saved DK salary file (data/research/history/dk_salaries/), for teams and salary used."""
    df = df[df["EntryId"].astype(str).str.strip() != ""]          # rows of the player list only
    lus = [parse_lineup(x) for x in df["Lineup"]]
    lus = [lu for lu in lus if len(lu) == 9]
    n = len(lus)
    out = {"entries": int(len(df)), "lineups_parsed": n}
    if not n:
        return out
    keys = pd.Series(["|".join(sorted(nm for _, nm in lu)) for lu in lus])
    vc = keys.value_counts()
    out["unique_lineups"] = int(len(vc))
    out["share_duplicated"] = round(float((keys.map(vc) > 1).mean()), 4)
    out["most_copies"] = int(vc.iloc[0])
    pos_of = dict(zip(own["Player"], own["pos"])) if len(own) else {}
    flex = pd.Series([next((pos_of.get(nm, "?") for s, nm in lu if s == "FLEX"), "?") for lu in lus])
    for p in ("RB", "WR", "TE"):
        out[f"flex_{p}"] = round(float((flex == p).mean()), 4)
    top = own.head(10)
    out["top_owned"] = "; ".join(f"{r.Player} {r.own:.1f}%" for r in top.itertuples())
    if salaries is not None and len(salaries):
        s = salaries.drop_duplicates("name_key").set_index("name_key")
        team = s["team"].to_dict()
        sal = s["salary"].to_dict()
        opp = {}
        if "game" in s:
            for k, g in s["game"].items():
                a, h = str(g).split("@") if "@" in str(g) else ("", "")
                opp[k] = h if team.get(k) == a else a

        def key(nm, slot):
            k = norm(nm)
            if slot == "DST" and k not in team:          # DK names defenses by nickname
                k = next((kk for kk in team if kk.endswith(" dst") and s.loc[kk, "Name"] == nm), k) \
                    if "Name" in s else k
            return k
        stacks = []
        sal_used = []
        for lu in lus:
            ks = [(slot, key(nm, slot), pos_of.get(nm, slot)) for slot, nm in lu]
            sal_used.append(sum(sal.get(k, np.nan) for _, k, _ in ks))
            qb = [k for slot, k, _ in ks if slot == "QB"]
            if not qb or qb[0] not in team:
                stacks.append((np.nan, np.nan))
                continue
            t, o = team[qb[0]], opp.get(qb[0], "")
            mates = sum(1 for slot, k, p in ks if slot != "QB" and team.get(k) == t and p in ("WR", "TE"))
            bb = sum(1 for slot, k, p in ks if team.get(k) == o and p in ("RB", "WR", "TE"))
            stacks.append((mates, bb))
        st = pd.DataFrame(stacks, columns=["mates", "bb"]).dropna()
        if len(st):
            out["qb_stacked"] = round(float((st["mates"] >= 1).mean()), 4)
            out["double_stack"] = round(float((st["mates"] == 2).mean()), 4)
            out["triple_stack"] = round(float((st["mates"] >= 3).mean()), 4)
            out["bring_back"] = round(float((st["bb"] >= 1).mean()), 4)
        su = pd.Series(sal_used).dropna()
        if len(su):
            out["avg_salary"] = round(float(su.mean()), 0)
            out["share_full_50k"] = round(float((su >= 50000).mean()), 4)
    return out


def save(path, day: str, salaries: pd.DataFrame | None = None, name: str = "") -> dict:
    """Keep a contest download for good and add its field habits to data/contests/field_habits.csv."""
    cid, df = read_download(path)
    d = CONTESTS / day
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{cid}.csv.gz", index=False)
    own = ownership_table(df)
    own.to_csv(d / f"{cid}_ownership.csv", index=False)
    h = {"date": day, "contest_id": cid, "contest": name, **field_habits(df, own, salaries)}
    f = CONTESTS / "field_habits.csv"
    old = pd.read_csv(f, dtype={"contest_id": str}) if f.exists() else pd.DataFrame()
    if len(old):
        old = old[~((old["date"] == day) & (old["contest_id"] == cid))]
    pd.concat([old, pd.DataFrame([h])], ignore_index=True).to_csv(f, index=False)
    return h


def saved_salaries(day: str) -> pd.DataFrame | None:
    d = ROOT / "data" / "research" / "history" / "dk_salaries"
    files = sorted(d.glob(f"{day}_*.csv.gz")) if d.exists() else []
    return pd.concat([pd.read_csv(f) for f in files]) if files else None
