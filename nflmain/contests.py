"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
DK contests: entries files (DKEntries.csv), contest details (size, payouts), and scoring
candidate lineups against a simulated field.

Classic entries files: Entry ID, Contest Name, Contest ID, Entry Fee, QB, RB, RB, WR, WR, WR, TE,
FLEX, DST (then DK's instructions and the salary list to the right). One file can hold contests
from several slates (Early / Main / Afternoon); each contest is built on its own slate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "nflmain/1.0"}
META = Path(__file__).resolve().parent.parent / "data" / "contests_meta"
SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
SLOT_COLS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
NCOL = 4 + len(SLOTS)


@dataclass
class Contest:
    id: str
    name: str
    fee: float
    size: int
    payouts: list = field(default_factory=list)   # (min_pos, max_pos, cash)
    entry_ids: list = field(default_factory=list)
    max_per_user: int | None = None
    prize_pool: float = 0.0
    payout_source: str = "DK"

    @property
    def large(self) -> bool:
        from .settings import get
        return self.size >= int(get("contest.large_field"))

    def pay_table(self) -> np.ndarray:
        """Prize for each finishing position 1..size (generic top-heavy table if unknown)."""
        n = max(int(self.size), 2)
        pay = np.zeros(n + 1)
        if self.payouts:
            for a, b, cash in self.payouts:
                pay[a - 1:min(b, n)] = cash
        else:
            pool = self.prize_pool or 0.85 * n * self.fee
            paid = max(int(0.22 * n), 1)
            w = 1 / np.arange(1, paid + 1) ** 1.1
            w = np.maximum(w / w.sum() * pool, 1.5 * self.fee * (paid > 1))
            pay[:paid] = w / w.sum() * pool
        return pay

    @property
    def cash_line(self) -> int:
        p = self.pay_table()
        nz = np.flatnonzero(p > 0)
        return int(nz[-1] + 1) if len(nz) else 1


def details(contest_id: str) -> dict:
    """Saved copies in data/contests_meta/ first (DK sometimes blocks cloud machines)."""
    f = META / f"{contest_id}.json"
    if f.exists():
        return json.loads(f.read_text()).get("contestDetail", {})
    r = requests.get(f"https://api.draftkings.com/contests/v1/contests/{contest_id}",
                     params={"format": "json"}, headers=UA, timeout=30)
    r.raise_for_status()
    META.mkdir(parents=True, exist_ok=True)
    f.write_text(r.text)
    return r.json().get("contestDetail", {})


def lobby_info(cid: str) -> dict | None:
    """Size, fee and prize pool from the DK lobby (no payout table) - fallback when the
    contest-details API refuses this machine."""
    try:
        from .slate import lobby
        for c in lobby().get("Contests", []):
            if str(c.get("id")) == str(cid):
                return c
    except Exception:  # noqa: BLE001
        return None
    return None


def contest_from_id(cid: str, entry_ids=None) -> Contest:
    try:
        d = details(str(cid))
    except Exception as e:  # noqa: BLE001
        lb = lobby_info(cid)
        if lb is None:
            raise
        print(f"[contest] DK payout table unavailable ({e.__class__.__name__}); using lobby size/fee/prizes "
              f"with an estimated top-heavy payout table for {lb.get('n')}")
        return Contest(str(cid), lb.get("n", str(cid)), float(lb.get("a") or 0), int(lb.get("m") or 0),
                       entry_ids=list(entry_ids or []), max_per_user=lb.get("mec"),
                       prize_pool=float(lb.get("po") or 0), payout_source="estimated")
    c = Contest(str(cid), d.get("name", str(cid)), float(d.get("entryFee") or 0), int(d.get("maximumEntries") or 0),
                entry_ids=list(entry_ids or []), max_per_user=d.get("maximumEntriesPerUser"))
    for p in d.get("payoutSummary", []):
        cash = sum(float(x.get("value") or 0) for x in p.get("payoutDescriptions", []))
        c.payouts.append((int(p["minPosition"]), int(p["maxPosition"]), cash))
    return c


def read_entries(path) -> pd.DataFrame:
    """Rows of a DKEntries.csv (only real entries). DK's file has the salary list to the right, so
    rows have different widths: read with the csv module and keep the first 13 columns."""
    import csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [(r + [""] * NCOL)[:NCOL] for r in csv.reader(fh)]
    head = [str(x).strip() for x in rows[0]]
    if head[4:13] != SLOTS:
        raise ValueError(f"{path}: not an NFL Classic entries file (columns {head[4:13]})")
    d = pd.DataFrame(rows[1:], columns=["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + SLOT_COLS, dtype=str)
    d = d[d["Entry ID"].str.strip().str.isdigit()].reset_index(drop=True)
    d["Contest ID"] = d["Contest ID"].str.strip()
    return d


def entry_ids_in(entries: pd.DataFrame) -> list[list[str | None]]:
    """Player IDs per entry in slot order (None where empty), from cells like 'Name (12345678)' or '12345678'."""
    import re
    out = []
    for _, r in entries.iterrows():
        lu = []
        for c in SLOT_COLS:
            m = re.search(r"(\d{6,})", str(r.get(c, "")))
            lu.append(m.group(1) if m else None)
        out.append(lu)
    return out


def load_contests(entries: pd.DataFrame, field_size: int | None = None) -> list[Contest]:
    out = []
    for cid, g in entries.groupby("Contest ID", sort=False):
        fee = float(str(g["Entry Fee"].iloc[0]).replace("$", "").replace(",", "") or 0)
        try:
            c = contest_from_id(cid, g["Entry ID"].tolist())
        except Exception as e:  # noqa: BLE001
            print(f"[contest] couldn't fetch details for {g['Contest Name'].iloc[0]} ({e}); generic payouts")
            c = Contest(cid, g["Contest Name"].iloc[0], fee, field_size or 10000, entry_ids=g["Entry ID"].tolist())
        if field_size:
            c.size = field_size
        out.append(c)
    return out


def write_entries(entries: pd.DataFrame, picks: dict, path) -> None:
    """picks: Entry ID -> list of 9 DK player IDs in slot order (DK accepts IDs on upload).
    Entries without a pick keep what they had."""
    d = entries.copy()
    for r in d.index:
        lu = picks.get(str(d.at[r, "Entry ID"]).strip())
        if lu is not None:
            d.loc[r, SLOT_COLS] = [str(x) for x in lu]
    d = d[["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + SLOT_COLS]
    d.columns = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + SLOTS
    d.to_csv(path, index=False)


# ------------------------------------------------------------------ contest simulation

def field_dupe_rate(cand_keys: list, field_keys: list, cand_q: np.ndarray, field_q: np.ndarray) -> np.ndarray:
    """P(a random field lineup is identical to each candidate): counted in the field sample,
    smoothed with an ownership-product estimate fitted on that sample (rare lineups)."""
    from collections import Counter
    cnt = Counter(field_keys)
    M = len(field_keys)
    emp = np.array([cnt.get(k, 0) for k in cand_keys], dtype=float) / M
    fq = np.array([cnt[k] for k in field_keys], dtype=float) / M
    mask = fq * M >= 3
    k = np.median(fq[mask] / np.maximum(field_q[mask], 1e-30)) if mask.sum() >= 10 else 0.0
    analytic = k * cand_q
    n_seen = np.array([cnt.get(kk, 0) for kk in cand_keys])
    return np.where(n_seen >= 3, emp, np.maximum(analytic, emp * 0.5))


def evaluate(cand: np.ndarray, field_lus: np.ndarray, fp: np.ndarray, contest: Contest,
             dupe_rate: np.ndarray, chunk: int = 500) -> dict:
    """cand: (L, S) our lineup scores; field_lus: (M, 9) simulated field lineups; fp: (n, S)
    player points. Per lineup per sim: payout, top-1% finish, cash, share of the field beaten."""
    from .lineups import scores
    L, S = cand.shape
    M = len(field_lus)
    N = max(contest.size, 2)
    pay = contest.pay_table()
    cum = np.r_[0, np.cumsum(pay)]
    dup = dupe_rate * (N - 1)                                  # expected copies of each lineup
    top_cut = max(N * 0.01, 1)
    cash_line = contest.cash_line
    out = {k: np.zeros((L, S), dtype=np.float32) for k in ("pay", "beat")}
    out["top1"] = np.zeros((L, S), dtype=bool)
    out["cash"] = np.zeros((L, S), dtype=bool)
    for a in range(0, S, chunk):
        b = min(a + chunk, S)
        F = np.sort(scores(field_lus, fp[:, a:b]), axis=0)               # (M, s)
        span = float(max(F.max(), cand[:, a:b].max()) - min(F.min(), cand[:, a:b].min()) + 10)
        off = np.arange(b - a) * span
        flatF = (F - F.min() + off[None, :]).T.ravel()
        q = (cand[:, a:b] - F.min() + off[None, :])
        below = np.searchsorted(flatF, q.ravel(), side="left").reshape(L, b - a) - np.arange(b - a)[None, :] * M
        frac_above = 1 - below / M
        above = frac_above * (N - 1 - dup[:, None])
        pos = 1 + np.rint(above).astype(np.int64)
        d = np.rint(dup).astype(np.int64)[:, None]
        hi = np.minimum(pos + d, N)
        out["pay"][:, a:b] = (cum[hi] - cum[pos - 1]) / (hi - pos + 1)
        out["top1"][:, a:b] = pos <= top_cut
        out["cash"][:, a:b] = pos <= cash_line
        out["beat"][:, a:b] = 1 - frac_above
    return out
