"""Slate-size backtests on the history database (mirrors research/gate in Kenny's NBA repo).

Every completed Sunday 2021-2026 gives real DK slates: Early (1:00 games), Main (1:00 + 4:00),
Afternoon (4:00 games), plus Thanksgiving. Projection = each player's trailing DK average (games
before this one only) scaled by his team's closing Vegas total, as the real system scales usage.
Pool = players who actually played (as if inactives were known, like the NBA test). Salaries:
real DK (2021, and 2026 from Week 9), else synthetic DK-style (research/history).
2026 weeks count WEIGHT_2026 x (default 3), 2025 weeks 1.5x, older 1x, because they reflect this
season's rosters and roles.

  gate       how many of each slate's real best-lineup players a ceiling-gate cut would throw out
  stacks     how the real best lineups are built, by slate size (QB stacks, bring-backs, game stacks)
  portfolio  20-lineup portfolios built under each setting (gate cut, stack rule, exposure cap),
             scored on the real results: best lineup as % of the real optimal, hit rates

    python -m research.backtest.slate_size gate|stacks|portfolio [--workers 4]
Results: research/backtest/results/*.csv and RESULTS.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nflmain import lineups as L

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "research" / "history"
RES = Path(__file__).resolve().parent / "results"
WEIGHT_2026, WEIGHT_2025 = 3.0, 1.5
CV = {"QB": 0.584, "RB": 0.703, "WR": 0.725, "TE": 0.741, "DST": 0.744}   # fitted 11/07/2026 (`cv`)
GATE_CUTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def weight(season: int) -> float:
    return WEIGHT_2026 if season == 2026 else WEIGHT_2025 if season == 2025 else 1.0


def size_bucket(n: int) -> str:
    return "2-3" if n <= 3 else "4-5" if n <= 5 else "6-9" if n <= 9 else "10+"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_csv(DB / "games.csv.gz")
    p = pd.read_csv(DB / "player_week.csv.gz", low_memory=False)
    d = pd.read_csv(DB / "dst_week.csv.gz")
    p = p.rename(columns={"position": "pos"})
    d = d.assign(name=d["team"] + " DST", player_id=d["team"] + "_DST")
    cols = ["season", "week", "game_id", "player_id", "name", "pos", "team", "opp", "dk_pts", "trail", "implied",
            "salary", "sal_source"]
    allp = pd.concat([p[cols], d[cols]], ignore_index=True)
    return g, allp


def slates(g: pd.DataFrame) -> list[dict]:
    """Real DK slates from the schedule: early / main / afternoon each Sunday, plus Thanksgiving."""
    out = []
    wk = g.groupby(["season", "week"])["completed"].all()
    for (s, w), done in wk.items():
        if not done:
            continue
        gw = g[(g["season"] == s) & (g["week"] == w)]
        for name, tags in (("early", ["early"]), ("main", ["early", "afternoon"]), ("afternoon", ["afternoon"]),
                           ("thanksgiving", ["thanksgiving"])):
            ids = gw.loc[gw["slate"].isin(tags), "game_id"].tolist()
            if len(ids) >= 2:
                out.append({"season": int(s), "week": int(w), "slate": name, "games": ids, "n": len(ids)})
    return out


def pool_for(sl: dict, allp: pd.DataFrame) -> pd.DataFrame:
    x = allp[allp["game_id"].isin(sl["games"])].copy()
    x = x[x["salary"].notna()]
    # projection: trailing DK average scaled by the team's Vegas total (defense: by the opponent's)
    imp = x["implied"].fillna(22.0)
    opp_imp = x.groupby("game_id")["implied"].transform("sum").fillna(44.0) - imp
    base = x["trail"].fillna(x["pos"].map({"QB": 12, "RB": 4, "WR": 4, "TE": 3, "DST": 6}))
    x["proj"] = np.where(x["pos"] == "DST", base * (opp_imp / 22.0) ** -0.8, base * (imp / 22.0) ** 0.6)
    x["game"] = x["game_id"]
    x["kickoff"] = ""
    return x[x["proj"] > 0.5].reset_index(drop=True)


def optimizer(x: pd.DataFrame, stack_min=0, bringback=0, game_max=8, eligible=None):
    P = _pool(x)
    return _opt(P, x, stack_min, bringback, game_max, eligible)


def _pool(x):
    """Player arrays for the optimizer (lineups.Pool)."""
    make = getattr(L, "Pool")
    return make(x["salary"].to_numpy(float), x["pos"].to_numpy(object), x["team"].to_numpy(object),
               x["opp"].to_numpy(object), x["game"].to_numpy(object), x["kickoff"].to_numpy(object))


def _opt(P, x, stack_min, bringback, game_max, eligible):
    rules = L.Rules(stack_min=stack_min, stack_max=3, bringback=bringback, max_per_team=8, game_max=game_max,
                    no_offense_vs_dst=stack_min > 0)
    return L.Optimizer(P, rules, eligible), P


def p_hit(proj, cv, need):
    s2 = np.log1p(cv ** 2)
    mu = np.log(np.maximum(proj, 0.5)) - s2 / 2
    return norm.sf((np.log(np.maximum(need, 0.5)) - mu) / np.sqrt(s2))


def gate_pct(x: pd.DataFrame, best_proj: float) -> np.ndarray:
    """Percentile of each player's chance to reach winning pace among playable players at his
    position (the production gate's yardstick: pace = best projected lineup pts/$1k + 0.1)."""
    pace = best_proj / 50 + 0.1
    ph = p_hit(x["proj"].to_numpy(), x["pos"].map(CV).to_numpy(), pace * x["salary"].to_numpy() / 1000)
    pct = np.ones(len(x))
    for pos in ("QB", "RB", "WR", "TE", "DST"):
        m = (x["pos"] == pos).to_numpy() & (x["proj"].to_numpy() >= 3)
        if m.sum() >= 4:
            r = np.sort(ph[m])
            pct[m] = np.searchsorted(r, ph[m], side="left") / len(r)
    return pct


# ------------------------------------------------------------------ gate + stacks (one pass)

def _slate_job(args):
    sl, x = args
    opt, P = optimizer(x)
    best_p = opt.solve(x["proj"].to_numpy())
    best_a = opt.solve(x["dk_pts"].to_numpy())
    if best_p is None or best_a is None:
        return None
    pct = gate_pct(x, float(x["proj"].to_numpy()[best_p].sum()))
    pos, team, opp, game = P.pos, P.team, P.opp, P.game
    lu = best_a
    qb = lu[pos[lu] == "QB"][0]
    mates = int(sum(1 for i in lu if team[i] == team[qb] and pos[i] in ("WR", "TE")))
    rb_mates = int(sum(1 for i in lu if team[i] == team[qb] and pos[i] == "RB"))
    bb = int(sum(1 for i in lu if team[i] == opp[qb] and pos[i] in ("RB", "WR", "TE")))
    dst = lu[pos[lu] == "DST"][0]
    vs_dst = int(sum(1 for i in lu if opp[i] == team[dst] and pos[i] != "DST"))
    g_counts = pd.Series(game[lu]).value_counts()
    # points given up by forcing each stack rule (vs the unconstrained real optimal)
    act = x["dk_pts"].to_numpy()
    lost = {}
    for name, kw in (("qb1", dict(stack_min=1)), ("qb2", dict(stack_min=2)), ("qb1_bb", dict(stack_min=1, bringback=2))):
        o2, _ = optimizer(x, **kw)
        l2 = o2.solve(act)
        lost[f"pts_lost_{name}"] = float(act[lu].sum() - act[l2].sum()) if l2 is not None else np.nan
    row = {**{k: sl[k] for k in ("season", "week", "slate", "n")}, "optimal": float(act[lu].sum()),
           "proj_best": float(x["proj"].to_numpy()[best_p].sum()), "qb_mates": mates, "qb_rb_mates": rb_mates,
           "bringback": bb, "vs_own_dst": vs_dst, "max_game": int(g_counts.max()),
           "games_used": int(len(g_counts)), **lost}
    spots = [{"season": sl["season"], "slate": sl["slate"], "n": sl["n"], "pos": pos[i], "pct": float(pct[i]),
              "salary": float(P.sal[i])} for i in lu]
    return row, spots


def run_slates(workers: int):
    from nflmain.parallel import pmap
    g, allp = load()
    sls = slates(g)
    jobs = [(sl, pool_for(sl, allp)) for sl in sls]
    res = pmap(_slate_job, jobs, workers=workers) if workers > 1 else [_slate_job(j) for j in jobs]
    rows, spots = [], []
    for r in res:
        if r is None:
            continue
        rows.append(r[0])
        spots += r[1]
    return pd.DataFrame(rows), pd.DataFrame(spots)


def summarize_gate(spots: pd.DataFrame) -> pd.DataFrame:
    spots = spots.assign(w=spots["season"].map(weight), bucket=spots["n"].map(size_bucket))
    out = []
    for (sl, b), gp in spots.groupby(["slate", "bucket"]):
        r = {"slate": sl, "games": b, "slates": int(len(gp) / 9), "weighted": round(gp["w"].sum() / 9, 1)}
        for c in GATE_CUTS[1:]:
            lost = np.average(gp["pct"] < c, weights=gp["w"])
            r[f"{int(c * 100)}% lost"] = round(100 * lost, 1)
            r[f"{int(c * 100)}% lift"] = round(1 - lost / c, 2)     # 0 = no better than a random cut
        out.append(r)
    return pd.DataFrame(out)


def summarize_stacks(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.assign(w=rows["season"].map(weight))
    out = []
    for sl, gp in rows.groupby("slate"):
        w = gp["w"]
        av = lambda s: round(float(np.average(s, weights=w)), 3)
        out.append({"slate": sl, "slates": len(gp), "avg games": round(gp["n"].mean(), 1),
                    "optimal pts": round(av(gp["optimal"]), 1),
                    "QB + 0 catchers": av(gp["qb_mates"] == 0), "QB + 1": av(gp["qb_mates"] == 1),
                    "QB + 2": av(gp["qb_mates"] == 2), "QB + 3+": av(gp["qb_mates"] >= 3),
                    "has bring-back": av(gp["bringback"] >= 1), "QB + own RB": av(gp["qb_rb_mates"] >= 1),
                    "hitter vs own DST": av(gp["vs_own_dst"] >= 1),
                    "most from 1 game (avg)": av(gp["max_game"]), "5+ from 1 game": av(gp["max_game"] >= 5),
                    "pts lost forcing QB+1": av(gp["pts_lost_qb1"].fillna(0)),
                    "pts lost forcing QB+2": av(gp["pts_lost_qb2"].fillna(0)),
                    "pts lost forcing QB+1+bring-back": av(gp["pts_lost_qb1_bb"].fillna(0))})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ portfolio test

PORTFOLIO_N = 20
STACK_RULES = {"none": dict(stack_min=0, bringback=0), "QB+1": dict(stack_min=1, bringback=0),
               "QB+2": dict(stack_min=2, bringback=0), "QB+1+BB": dict(stack_min=1, bringback=2)}
CAPS = [0.3, 0.4, 0.5, 0.7, 1.0]


def worlds(x: pd.DataFrame, k: int, rng) -> np.ndarray:
    """Correlated simulated outcomes around the projection: player noise (position CV) x a team
    factor (teammates rise together) x a game factor (both sides of a shootout); DST moves against
    the opposing offense. A light stand-in for the full game simulator, fast enough for 300 slates."""
    n = len(x)
    teams = x["team"].to_numpy(object)
    games_ = x["game"].to_numpy(object)
    tu, ti = np.unique(teams, return_inverse=True)
    gu, gi = np.unique(games_, return_inverse=True)
    tf = rng.normal(0, 0.22, (len(tu), k))
    gf = rng.normal(0, 0.15, (len(gu), k))
    cv = x["pos"].map(CV).to_numpy()
    s2 = np.log1p(cv ** 2)[:, None]
    z = rng.standard_normal((n, k)) * np.sqrt(s2)
    shared = tf[ti] + gf[gi]
    is_dst = (x["pos"] == "DST").to_numpy()
    opp_idx = np.array([np.flatnonzero(tu == o)[0] if (tu == o).any() else 0 for o in x["opp"]])
    shared[is_dst] = -tf[opp_idx[is_dst]] * 0.8
    fp = x["proj"].to_numpy()[:, None] * np.exp(z + shared - s2 / 2 - 0.5 * (0.22 ** 2 + 0.15 ** 2))
    return fp.astype(np.float32)


def pick(lus: np.ndarray, sc: np.ndarray, n: int, cap: float) -> list[int]:
    """Greedy portfolio: highest ceiling (90th pct in the worlds) first, under the exposure cap, no
    repeats, lineups differing by 2+ players."""
    order = np.argsort(-np.percentile(sc, 90, axis=1))
    lim = max(1, int(np.floor(cap * n + 1e-9)))
    cnt: dict = {}
    out = []
    for j in order:
        if any(cnt.get(int(p), 0) >= lim for p in lus[j]):
            continue
        if any(len(set(lus[j]) & set(lus[o])) > 7 for o in out):
            continue
        out.append(int(j))
        for p in lus[j]:
            cnt[int(p)] = cnt.get(int(p), 0) + 1
        if len(out) == n:
            break
    return out


def _portfolio_job(args):
    sl, x, seed, k = args
    rng = np.random.default_rng(seed)
    act = x["dk_pts"].to_numpy()
    o0, _ = optimizer(x)
    best_a = o0.solve(act)
    best_p = o0.solve(x["proj"].to_numpy())
    if best_a is None or best_p is None:
        return []
    optimal = float(act[best_a].sum())
    pct = gate_pct(x, float(x["proj"].to_numpy()[best_p].sum()))
    fp = worlds(x, k, rng)
    rows = []
    variants = [("gate", c, "QB+1", 0.5) for c in GATE_CUTS] + \
               [("stack", 0.10 if sl["n"] >= 6 else 0.05, s, 0.5) for s in STACK_RULES if s != "QB+1"] + \
               [("cap", 0.10 if sl["n"] >= 6 else 0.05, "QB+1", c) for c in CAPS if c != 0.5]
    cache = {}
    for kind, cut, stack, cap in variants:
        key = (cut, stack)
        if key not in cache:
            elig = (pct >= cut).astype(float)
            opt, _ = optimizer(x, eligible=elig, game_max=8, **STACK_RULES[stack])
            lus = [opt.solve(fp[:, s], time_limit=3) for s in range(k)]
            lus = np.unique(np.array([np.sort(l) for l in lus if l is not None]), axis=0)
            cache[key] = (lus, L.scores(lus, fp) if len(lus) else None)
        lus, sc = cache[key]
        if not len(lus):
            continue
        ch = pick(lus, sc, PORTFOLIO_N, cap)
        a = act[lus[ch]].sum(axis=1)
        rows.append({"season": sl["season"], "week": sl["week"], "slate": sl["slate"], "n": sl["n"], "test": kind,
                     "gate": cut, "stack": stack, "cap": cap, "best_pct": float(a.max() / optimal),
                     "avg_pct": float(a.mean() / optimal), "hit85": float(a.max() >= 0.85 * optimal),
                     "hit80": float(a.max() >= 0.80 * optimal), "lineups": len(ch)})
    return rows


def run_portfolio(workers: int, k: int = 60, seed: int = 3, only=None):
    from nflmain.parallel import pmap
    g, allp = load()
    sls = [s for s in slates(g) if only is None or s["slate"] in only]
    jobs = [(sl, pool_for(sl, allp), seed + i, k) for i, sl in enumerate(sls)]
    res = pmap(_portfolio_job, jobs, workers=workers) if workers > 1 else [_portfolio_job(j) for j in jobs]
    return pd.DataFrame([r for rr in res for r in rr])


def summarize_portfolio(r: pd.DataFrame) -> pd.DataFrame:
    r = r.assign(w=r["season"].map(weight))
    out = []
    for (sl, test, gate, stack, cap), gp in r.groupby(["slate", "test", "gate", "stack", "cap"]):
        w = gp["w"]
        out.append({"slate": sl, "test": test, "gate": gate, "stack": stack, "cap": cap, "slates": len(gp),
                    "best lineup % of optimal": round(100 * np.average(gp["best_pct"], weights=w), 1),
                    "avg lineup %": round(100 * np.average(gp["avg_pct"], weights=w), 1),
                    "hit 80%": round(100 * np.average(gp["hit80"], weights=w), 1),
                    "hit 85%": round(100 * np.average(gp["hit85"], weights=w), 1),
                    "2026 only: best %": round(100 * gp.loc[gp["season"] == 2026, "best_pct"].mean(), 1)
                    if (gp["season"] == 2026).any() else np.nan})
    return pd.DataFrame(out)


def fit_cv() -> dict:
    """Per position: SD / mean of DK points among players with a solid trailing average."""
    _, allp = load()
    x = allp[allp["trail"] >= 6]
    out = {}
    for pos, gp in x.groupby("pos"):
        ratio = gp["dk_pts"] / gp["trail"]
        out[pos] = round(float(ratio.std()), 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gate", "stacks", "portfolio", "cv"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=60)
    ap.add_argument("--slates", nargs="*", help="limit portfolio test to these slate types")
    a = ap.parse_args(argv)
    RES.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if a.what == "cv":
        print(fit_cv())
        return
    if a.what in ("gate", "stacks"):
        rows, spots = run_slates(a.workers)
        rows.to_csv(RES / "slates.csv", index=False)
        spots.to_csv(RES / "optimal_spots.csv", index=False)
        sg, ss = summarize_gate(spots), summarize_stacks(rows)
        sg.to_csv(RES / "gate_by_slate_size.csv", index=False)
        ss.to_csv(RES / "stacks_by_slate.csv", index=False)
        print(sg.to_string(index=False))
        print()
        print(ss.T.to_string())
    else:
        r = run_portfolio(a.workers, a.worlds, only=a.slates)
        r.to_csv(RES / "portfolio_raw.csv", index=False)
        s = summarize_portfolio(r)
        s.to_csv(RES / "portfolio_by_setting.csv", index=False)
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
