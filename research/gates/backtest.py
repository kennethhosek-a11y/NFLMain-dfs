"""Ceiling-gate research by slate size (mirrors the NBA research/gate study).

Question: how big a ceiling-gate cut is safe for each slate size? The NBA finding: a flat 25% cut
threw out 15-22% of the players in each night's actual best lineup, so the cut was scaled by slate
size (0% at 2 games ... 15% at 8+).

This version (11/07/2026) uses the model's own simulated slates as "truth": for slates of 2, 3, 4,
7 and 11 games (games sampled from a real DK slate), each held-out simulated slate's best lineup is
found, and we measure how many of its players each gate cut would have thrown out. It shows how
the trade-off moves with slate size, but it trusts the model; the real-data version (actual DK
points and salaries, week by week - Kenny's Showdown database has 406 games with salaries and
actual points) is the next step and decides the settings. See README.md.

    python -m research.gates.backtest --date 2026-11-08 --slate main
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nflmain import lineups as L
from nflmain import pipeline, settings, slate

CUTS = (0.0, 0.05, 0.10, 0.15, 0.25)
SIZES = (2, 3, 4, 7, 11)


def gate_mask(df: pd.DataFrame, fp: np.ndarray, pace: float, cut: float) -> np.ndarray:
    hit = (fp / np.maximum(df["Salary"].to_numpy()[:, None] / 1000, 0.2) >= pace).mean(axis=1)
    out = np.zeros(len(df), bool)
    if cut <= 0:
        return out
    for pos in ("QB", "RB", "WR", "TE", "DST"):
        m = (df["pos"] == pos).to_numpy() & (df["Projection"].to_numpy() >= 3)
        if m.sum() >= 4:
            out |= m & (hit < np.quantile(hit[m], cut))
    return out


def study(p, n_slates: int = 6, n_truth: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    games = sorted(p.df["game"].unique())
    rows = []
    for size in [s for s in SIZES if s <= len(games)]:
        for k in range(n_slates if size < len(games) else 1):
            pick = list(rng.choice(games, size=size, replace=False))
            sub = p.df[p.df["game"].isin(pick) & (p.df["Projection"] > 0.3)]
            idx = np.flatnonzero(p.df.index.isin(sub.index))
            fp = p.fp[idx]
            df = sub.reset_index(drop=True)
            P = L.Pool.from_df(df)
            rules = L.Rules(stack_min=0, stack_max=8, bringback=0, game_max=8, max_per_team=8, max_vs_own_dst=8, min_salary=0, rb_per_team=3, no_qb_team_max=8, qb_own_rb=2, hard=False)
            opt = L.Optimizer(P, rules)
            best = opt.solve(df["Projection"].to_numpy())
            if best is None:
                continue
            pace = df["Projection"].to_numpy()[best].sum() / 50 + 0.1
            truth = rng.choice(fp.shape[1], size=n_truth, replace=False)
            rest = np.setdiff1d(np.arange(fp.shape[1]), truth)
            masks = {c: gate_mask(df, fp[:, rest], pace, c) for c in CUTS}
            for s in truth:
                lu = opt.solve(fp[:, s])
                if lu is None:
                    continue
                for c, m in masks.items():
                    rows.append({"games": size, "cut": c, "thrown_out": m[lu].mean()})
    r = pd.DataFrame(rows)
    return r.groupby(["games", "cut"])["thrown_out"].mean().unstack().mul(100).round(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--slate", default="main")
    a = ap.parse_args()
    settings.use(a.date)
    p = pipeline.run(a.slate, a.date, n_sims=4000, quiet=True, save=False)
    t = study(p)
    print("% of each simulated slate's best-lineup players the gate would throw out (rows: games on slate,"
          " columns: share cut)")
    print(t.to_string())


if __name__ == "__main__":
    main()
