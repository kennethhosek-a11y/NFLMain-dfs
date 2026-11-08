"""Backtest (Kenny 11/08 list, item 7): let the QB have up to 3 of his own pass-catchers instead of
the hard max of 2. Same method as research/backtest/rules.py (every real 2021-2026 slate, 20
lineups from 30 correlated simulated slates, scored on what really happened: best lineup as % of
the real optimal and the 3rd-best). Every other hard rule stays on (no DST vs own QB, $49,500).
"Triples allowed" lets the picker use them freely; it does not force any.

    python -m research.backtest.stack3 [--workers 4] [--worlds 30]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nflmain import lineups as L

from . import rules as R
from . import slate_size as B


def _rules(stack_max: int) -> L.Rules:
    r = L.Rules(**R.BASE)                 # hard rules on
    r.stack_max = stack_max               # research only: past the hard max of 2
    return r


def _job(args):
    sl, x, seed, k = args
    rng = np.random.default_rng(seed)
    act = x["dk_pts"].to_numpy()
    P = R.make_pool(x)
    free = L.Rules(stack_min=0, stack_max=8, bringback=0, max_per_team=9, game_max=9, max_vs_own_dst=8,
                   min_salary=0, rb_per_team=3, no_qb_team_max=8, qb_own_rb=2, hard=False)
    best = L.Optimizer(P, free).solve(act)
    if best is None:
        return []
    optimal = float(act[best].sum())
    q = [i for i in best if P.pos[i] == "QB"][0]
    mates = sum(1 for i in best if P.team[i] == P.team[q] and P.pos[i] in ("WR", "TE"))
    fp = B.worlds(x, k, rng)
    rows = []
    for name, sm in (("baseline (QB + 1-2)", 2), ("QB + up to 3 allowed", 3)):
        opt = L.Optimizer(P, _rules(sm))
        lus = [opt.solve(fp[:, s], time_limit=3) for s in range(k)]
        lus = [l for l in lus if l is not None]
        if not lus:
            continue
        lus = np.unique(np.array([np.sort(l) for l in lus]), axis=0)
        ch = R.pick(lus, L.scores(lus, fp), R.N)
        a = np.sort(act[lus[ch]].sum(axis=1))[::-1]
        trip = float(np.mean([sum(1 for i in lus[j] if P.team[i] == P.team[[p for p in lus[j] if P.pos[p] == "QB"][0]]
                                   and P.pos[i] in ("WR", "TE")) >= 3 for j in ch]))
        rows.append({"season": sl["season"], "week": sl["week"], "slate": sl["slate"], "n": sl["n"], "rule": name,
                     "best_pct": float(a[0] / optimal), "third_pct": float(a[min(2, len(a) - 1)] / optimal),
                     "triple_share": trip, "optimal_mates": mates})
    return rows


def main(argv=None):
    from nflmain.parallel import pmap
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    g, allp = B.load()
    allp["tgt"] = R.trailing_tgt(allp)
    sls = B.slates(g)
    if a.limit:
        sls = sls[-a.limit:]
    jobs = [(sl, B.pool_for(sl, allp).assign(tgt=lambda d: d["tgt"] if "tgt" in d else 0.0), 7 + i, a.worlds)
            for i, sl in enumerate(sls)]
    res = pmap(_job, jobs, workers=a.workers) if a.workers > 1 else [_job(j) for j in jobs]
    r = pd.DataFrame([x for rr in res for x in rr])
    R.RES.mkdir(parents=True, exist_ok=True)
    r.to_csv(R.RES / "stack3_raw.csv", index=False)
    w = np.where(r["season"] == 2026, 3.0, np.where(r["season"] == 2025, 1.5, 1.0))
    r = r.assign(w=w)
    base = r[r.rule.str.startswith("baseline")].set_index(["season", "week", "slate"])
    alt = r[~r.rule.str.startswith("baseline")].set_index(["season", "week", "slate"])
    j = alt.join(base[["best_pct", "third_pct"]], rsuffix="_base", how="inner")
    for col in ("best_pct", "third_pct"):
        d = (j[col] - j[col + "_base"]) * 100
        ww = j["w"]
        m = float((d * ww).sum() / ww.sum())
        se = float(np.sqrt(((d - m) ** 2 * ww).sum() / ww.sum() / max(len(d) - 1, 1)))
        d26 = d[j.index.get_level_values("season") == 2026]
        print(f"{col}: {m:+.2f} +/- {se:.2f} (weighted, {len(d)} slates); 2026 only {d26.mean():+.2f} ({len(d26)} slates)")
    print(f"triples used in {100 * j['triple_share'].mean():.0f}% of picked lineups; real best lineup had QB+3+: "
          f"{100 * (r.drop_duplicates(['season', 'week', 'slate'])['optimal_mates'] >= 3).mean():.0f}% of slates")


if __name__ == "__main__":
    main()
