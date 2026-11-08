"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
Research overrides: data/overrides/<YYYY-MM-DD>.csv, applied on top of the model for every slate
that day (an optional Slate column, e.g. "afternoon", limits a row to one slate type).

One row per player, only the columns research has an opinion on:
  Name           player name as on DK (DSTs: the DK name, e.g. "Jaguars")
  ProjectionAdj  DK points added to the FLEX projection (negative to cut). The player's whole
                 simulated distribution is scaled, so correlations are kept.
  Status         OUT / Q / D, ACTIVE (clears a tag), or START (this QB starts for his team)
  Ownership      projected ownership %   (replaces the model's)
  Slate          optional: early / main / afternoon / thanksgiving (blank = every slate)
  Note           free text, carried into the output and the workbook

No file = no change. Every applied change is printed and kept in the projections
(ResearchAdj, ResearchNote, ProjSource "+research") so results show whether research helped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .names import norm

DIR = Path(__file__).resolve().parent.parent / "data" / "overrides"
COLS = ["Name", "ProjectionAdj", "Status", "Ownership", "Note", "Slate", "MaxExposure"]
STATUS = {"OUT": "OUT", "O": "OUT", "Q": "Q", "D": "D", "DOUBTFUL": "D", "GTD": "Q", "ACTIVE": "",
          "IN": "", "START": "START", "": None}


def path_for(day: str, directory: Path | None = None) -> Path:
    return (directory or DIR) / f"{day}.csv"


def load(day: str, slate_type: str | None = None, directory: Path | None = None,
         quiet: bool = False) -> pd.DataFrame | None:
    f = path_for(day, directory)
    if not f.exists():
        return None
    ov = pd.read_csv(f, dtype=str, keep_default_na=False)
    if "Name" not in ov:
        print(f"[research] {f.name} has no Name column - ignored")
        return None
    for c in COLS:
        if c not in ov:
            ov[c] = ""
    for c in ("ProjectionAdj", "Ownership", "MaxExposure"):
        ov[c] = pd.to_numeric(ov[c].str.strip(), errors="coerce")
    ov.loc[ov["MaxExposure"] > 1, "MaxExposure"] /= 100          # 45 or 0.45 both mean 45%
    ov["Status"] = ov["Status"].str.strip().str.upper()
    ov["Note"] = ov["Note"].str.strip()
    ov["Slate"] = ov["Slate"].str.strip().str.lower()
    if slate_type:
        ov = ov[(ov["Slate"] == "") | (ov["Slate"] == slate_type)]
    ov["name_key"] = ov["Name"].map(norm)
    ov = ov[ov["name_key"] != ""].drop_duplicates("name_key", keep="last")
    if not quiet:
        print(f"[research] {len(ov)} override row(s) from {f.name}")
    return ov.set_index("name_key")


def _keys(df: pd.DataFrame) -> pd.Series:
    """Match on player name, and DSTs also on their DK name ('Jaguars')."""
    return df["Name"].map(norm)


def apply_status(pool: pd.DataFrame, ov: pd.DataFrame | None) -> pd.DataFrame:
    pool = pool.copy()
    pool["qb_override"] = False
    if ov is None:
        return pool
    keys = _keys(pool)
    missing = sorted(set(ov.index) - set(keys))
    if missing:
        print("[research] not in this player pool (ignored): " + ", ".join(ov.loc[missing, "Name"]))
    for k, r in ov.iterrows():
        if k not in set(keys):
            continue
        new = STATUS.get(r["Status"], "bad")
        if new == "bad":
            print(f"[research] {r['Name']}: unknown Status '{r['Status']}' - ignored")
            continue
        if new is None:
            continue
        hit = keys == k
        if new == "START":
            pool.loc[hit, "qb_override"] = True
            pool.loc[hit, "status"] = ""
            print(f"[research] {r['Name']}: starting QB")
            continue
        old = pool.loc[hit, "status"].iloc[0]
        if old != new:
            print(f"[research] {r['Name']}: status {old or 'active'} -> {new or 'active'}")
        pool.loc[hit, "status"] = new
    return pool


def apply_projection(proj: pd.DataFrame, fp: np.ndarray, ov: pd.DataFrame | None) -> np.ndarray:
    """Scale each adjusted player's simulated points so his mean moves by ProjectionAdj."""
    proj["ResearchAdj"] = 0.0
    proj["ResearchNote"] = ""
    if ov is None:
        return fp
    fp = fp.copy()
    keys = _keys(proj)
    for k, r in ov.iterrows():
        rows = np.flatnonzero((keys == k).values)
        if not len(rows):
            continue
        i = rows[0]
        if r["Note"]:
            proj.iat[i, proj.columns.get_loc("ResearchNote")] = r["Note"]
        if STATUS.get(r["Status"]) is not None:
            proj.iat[i, proj.columns.get_loc("ProjSource")] += "+research"
        if pd.isna(r["ProjectionAdj"]):
            continue
        m0 = float(fp[i].mean())
        if m0 <= 0.05:
            print(f"[research] {r['Name']}: projected 0 - points override ignored")
            continue
        m1 = max(m0 + float(r["ProjectionAdj"]), 0.0)
        fp[i] *= m1 / m0
        proj.iat[i, proj.columns.get_loc("ResearchAdj")] = m1 - m0
        if "+research" not in proj.iat[i, proj.columns.get_loc("ProjSource")]:
            proj.iat[i, proj.columns.get_loc("ProjSource")] += "+research"
        print(f"[research] {r['Name']}: {r['ProjectionAdj']:+.1f} pts -> projection {m0:.1f} -> {m1:.1f}")
    return fp


def apply_ownership(proj: pd.DataFrame, own: np.ndarray, ov: pd.DataFrame | None) -> np.ndarray:
    own = own.copy()
    if ov is None:
        return own
    keys = _keys(proj)
    for k, r in ov.iterrows():
        rows = np.flatnonzero((keys == k).values)
        if not len(rows) or proj["Projection"].values[rows[0]] <= 0 or pd.isna(r["Ownership"]):
            continue
        i = rows[0]
        print(f"[research] {r['Name']}: ownership {own[i]:.1f}% -> {float(r['Ownership']):.1f}%")
        own[i] = float(np.clip(r["Ownership"], 0, 100))
    return own


def exposure_exceptions(pool: pd.DataFrame, ov: pd.DataFrame | None, quiet: bool = False) -> dict:
    """Kenny's rule (11/08/2026): players at or under rules.cheap_salary are capped at
    rules.cheap_max_exposure UNLESS research argues a serious mispricing. An exception is a row in
    the overrides file with MaxExposure (e.g. 0.45) AND a Note making the case; without a Note it is
    ignored. Returns {pool row index: max share}."""
    out = {}
    if ov is None or "MaxExposure" not in ov:
        return out
    keys = _keys(pool).to_numpy()
    for k, r in ov.iterrows():
        if pd.isna(r["MaxExposure"]):
            continue
        rows = np.flatnonzero(keys == k)
        if not len(rows):
            continue
        if not str(r["Note"]).strip():
            print(f"[research] {r['Name']}: MaxExposure ignored - an exception needs a Note arguing the mispricing")
            continue
        out[int(rows[0])] = float(np.clip(r["MaxExposure"], 0.0, 1.0))
        if not quiet:
            print(f"[research] exposure exception: {r['Name']} up to {r['MaxExposure']:.0%} - {r['Note']}")
    return out
