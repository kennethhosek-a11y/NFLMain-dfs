"""Backtest of the Moderate / Opinion rules in Kenny's lineup-rules workbook
(research/rules/NFL_DFS_Lineup_Rules_2026-11-08.xlsx), one rule at a time, on every real
2021-2026 slate in the history database (2026 weeks x3, 2025 x1.5).

For every slate:
  1. "Real best lineup" check: does the slate's actual best lineup follow the rule, and how many
     points does forcing the rule cost (best lineup under the rule vs without, on actual points)?
  2. Portfolio check: 20 lineups built from 30 correlated simulated slates (research/backtest/
     slate_size.worlds) with the rule ON vs the baseline (hard rules only), scored on what really
     happened: best lineup as % of the real optimal, and the 3rd-best (depth).
Verdict per rule: "held up" = the paired difference vs baseline is not negative beyond 2 standard
errors on either the weighted set or 2026 alone; "helps" = positive beyond 2 SE; "hurts" =
negative beyond 2 SE. Rules about ownership (M12-M14 soft parts, M4's chalk credit, SS6) can't be
tested without ownership history (paid) - they are listed as untested.

    python -m research.backtest.rules [--workers 4] [--worlds 30]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflmain import lineups as L
from nflmain.portfolio import Extra

from . import slate_size as B

RES = Path(__file__).resolve().parent / "results"
N = 20
SHORT = ("afternoon", "thanksgiving")

# baseline: hard rules only (QB + 1-2 catchers, no DST vs own QB, $49,500), everything else off
BASE = dict(stack_min=1, stack_max=2, bringback=1, max_per_team=8, game_max=8, max_vs_own_dst=8,
            rb_per_team=3, no_qb_team_max=8, qb_own_rb=2)
# lineup-level rules (their own candidates): name -> (workbook id, evidence, Rules changes, slate types)
LINEUP = {
    "M6 at most 1 hitter vs own DST": ("M6", "Opinion", dict(max_vs_own_dst=1), None),
    "M6 none vs own DST (old rule)": ("M6", "Opinion", dict(max_vs_own_dst=0), None),
    "M7 QB + own RB only if receiver": ("M7", "Moderate", dict(qb_own_rb=1), None),
    "M8 max 1 RB per team": ("M8", "Opinion", dict(rb_per_team=1), None),
    "M8 max 2 from a team without its QB": ("M8", "Opinion", dict(no_qb_team_max=2), None),
    "M8 max 4 per team": ("M8", "Opinion", dict(max_per_team=4), None),
    "M8 all three team limits": ("M8", "Opinion", dict(rb_per_team=1, no_qb_team_max=2, max_per_team=4), None),
    "M9 stacks ONLY from the 2 highest-total games (what M9 warns against)": ("M9", "Moderate", dict(top_games=2), None),
    "SS1+SS2 short slates: 5 per team, 2 RBs, QB+RB ok (vs M7+M8 on)": ("SS1/SS2", "Opinion",
        dict(max_per_team=5, rb_per_team=2, qb_own_rb=2, no_qb_team_max=2), SHORT),
    "SS3 short slates: up to 2 hitters vs own DST (vs M6)": ("SS3", "Opinion", dict(max_vs_own_dst=2), SHORT),
    "SS4 short slates: $48,500 floor (vs $49,500)": ("SS4", "Opinion", dict(floor=48500), SHORT),
}
# reference for the short-slate rows: the main rules M6-M8 on
SHORT_REF = dict(max_vs_own_dst=1, rb_per_team=1, no_qb_team_max=2, max_per_team=4, qb_own_rb=1)
# portfolio-level rules (baseline candidates, different picking)
PORT = {
    "M4 ~50% of lineups with exactly 1 bring-back": ("M4", "Moderate", dict(bands={"bringback": (0.4, 0.6)})),
    "M11 TE in FLEX at most 10%": ("M11", "Moderate", dict(bands={"te_flex": (0.0, 0.1)})),
    "M15 QB max 25%, DST max 15%": ("M15", "Opinion", dict(pos_caps={"QB": 0.25, "DST": 0.15})),
    "M16 lineups differ by 2+ players": ("M16", "Opinion", dict(min_unique=2)),
    "M16 lineups differ by 3+ players": ("M16", "Opinion", dict(min_unique=3)),
    "SS5 short slates: exposure up to 60%": ("SS5", "Opinion", dict(cap=0.6)),
}


def rules_for(kw: dict) -> L.Rules:
    kw = dict(kw)
    floor = kw.pop("floor", None)
    kw.pop("top_games", None)
    r = L.Rules(**{**BASE, **kw})
    if floor:
        r = L.Rules(**{**BASE, **kw}, hard=False, min_salary=floor)
        r.stack_min, r.stack_max = 1, 2
    return r


def trailing_tgt(allp: pd.DataFrame) -> pd.Series:
    p = pd.read_csv(B.DB / "player_week.csv.gz", usecols=["season", "week", "player_id", "target_share"], low_memory=False)
    p = p.sort_values(["player_id", "season", "week"])
    p["tgt"] = p.groupby("player_id")["target_share"].transform(lambda s: s.shift(1).ewm(halflife=4, min_periods=1).mean())
    return allp.merge(p[["season", "week", "player_id", "tgt"]], on=["season", "week", "player_id"], how="left")["tgt"].fillna(0).to_numpy()


def make_pool(x: pd.DataFrame):
    P = B._pool(x)
    P.tgt = x["tgt"].to_numpy(float)
    return P


def pick(lus, sc, n, cap=0.5, extra: Extra | None = None):
    """Greedy by ceiling (90th pct in the worlds), with the exposure cap, per-position caps,
    lineup differences and shape bands (lower bounds filled first)."""
    order = list(np.argsort(-np.percentile(sc, 90, axis=1)))
    capn = max(1, int(np.floor(cap * n + 1e-9)))
    pos = extra.pos if extra is not None else None
    capv = {}
    if extra is not None:
        for p_, f in extra.pos_caps.items():
            capv[p_] = max(2, int(np.floor(f * n + 1e-9)))
    mu = extra.min_unique if extra is not None else 1
    bands = extra.band_counts(n) if extra is not None else {}
    out, cnt = [], {}

    def fits(j):
        for p in lus[j]:
            lim = min(capn, capv.get(pos[p], capn)) if pos is not None else capn
            if cnt.get(int(p), 0) + 1 > lim:
                return False
        if mu > 1 and any(len(set(lus[j]) & set(lus[o])) > 9 - mu for o in out):
            return False
        for k, (lo, hi) in bands.items():
            if extra.props[k][j] and sum(extra.props[k][o] for o in out) + 1 > hi:
                return False
        return True

    def add(j):
        out.append(int(j))
        for p in lus[j]:
            cnt[int(p)] = cnt.get(int(p), 0) + 1

    for k, (lo, hi) in bands.items():           # reserve the band minimums first
        for j in order:
            if sum(extra.props[k][o] for o in out) >= lo:
                break
            if extra.props[k][j] and j not in out and fits(j):
                add(j)
    for j in order:
        if len(out) >= n:
            break
        if j not in out and fits(j):
            add(j)
    return out


def follows(name: str, lu, P, top_games=None, receiving=0.12) -> bool:
    """Does lineup lu obey just this one rule (ignoring every other rule)?"""
    pos, team, opp, game = P.pos, P.team, P.opp, P.game
    q = lu[pos[lu] == "QB"][0]
    d = lu[pos[lu] == "DST"][0]
    vs_dst = sum(1 for i in lu if opp[i] == team[d] and pos[i] in ("RB", "WR", "TE"))
    rbs = [i for i in lu if pos[i] == "RB"]
    rb_team = max(pd.Series(team[rbs]).value_counts()) if rbs else 0
    per_team = int(pd.Series(team[lu]).value_counts().max())
    no_qb = max([sum(1 for i in lu if team[i] == t and pos[i] in ("RB", "WR", "TE")) for t in set(team[lu]) - {team[q]}] or [0])
    own_rb = [i for i in rbs if team[i] == team[q]]
    if name.startswith("M6 at most 1"):
        return vs_dst <= 1
    if name.startswith("M6 none"):
        return vs_dst == 0
    if name.startswith("M7"):
        return all(P.tgt[i] >= receiving for i in own_rb)
    if name.startswith("M8 max 1 RB"):
        return rb_team <= 1
    if name.startswith("M8 max 2 from"):
        return no_qb <= 2
    if name.startswith("M8 max 4"):
        return per_team <= 4
    if name.startswith("M8 all"):
        return rb_team <= 1 and no_qb <= 2 and per_team <= 4
    if name.startswith("M9"):
        return game[q] in top_games
    if name.startswith("SS1"):
        return per_team > 4 or rb_team > 1 or bool(own_rb)          # uses a relaxation
    if name.startswith("SS3"):
        return vs_dst == 2
    if name.startswith("SS4"):
        return P.sal[lu].sum() < 49500
    return np.nan


def _job(args):
    sl, x, seed, k = args
    rng = np.random.default_rng(seed)
    act = x["dk_pts"].to_numpy()
    P = make_pool(x)
    free = L.Rules(stack_min=0, stack_max=8, bringback=0, max_per_team=9, game_max=9, max_vs_own_dst=8,
                   min_salary=0, rb_per_team=3, no_qb_team_max=8, qb_own_rb=2, hard=False)
    best = L.Optimizer(P, free).solve(act)
    if best is None:
        return []
    optimal = float(act[best].sum())
    fp = B.worlds(x, k, rng)
    tot = x.groupby("game")["implied"].sum().sort_values(ascending=False)
    rows = []

    def build(kw):
        r = rules_for(kw)
        elig = np.ones(len(x))
        if kw.get("top_games"):
            topg = set(tot.index[:kw["top_games"]])
            elig[(P.pos == "QB") & ~np.isin(P.game, list(topg))] = 0
        opt = L.Optimizer(P, r, elig)
        lus = [opt.solve(fp[:, s], time_limit=3) for s in range(k)]
        lus = [l for l in lus if l is not None]
        if not lus:
            return None, None, r
        lus = np.unique(np.array([np.sort(l) for l in lus]), axis=0)
        # actual best lineup under this rule (cost of forcing it)
        a = opt.solve(act, time_limit=5)
        return lus, (float(act[a].sum()) if a is not None else np.nan), r

    def score(name, lus, ch, best_rule, compliant, ref_best=None):
        a = act[lus[ch]].sum(axis=1)
        s = np.sort(a)[::-1]
        rows.append({"season": sl["season"], "week": sl["week"], "slate": sl["slate"], "n": sl["n"], "rule": name,
                     "best_pct": float(s[0] / optimal), "third_pct": float(s[min(2, len(s) - 1)] / optimal),
                     "pts_lost_forcing": ((ref_best if ref_best is not None else base_best) - best_rule)
                     if best_rule == best_rule else np.nan,
                     "optimal_follows": compliant})

    base_best = np.nan
    base_lus, base_best, base_r = build({})
    if base_lus is None:
        return []
    sc = L.scores(base_lus, fp)
    score("baseline", base_lus, pick(base_lus, sc, N), base_best, True)
    short = sl["slate"] in SHORT
    if short:
        ref_lus, ref_best, _ = build(SHORT_REF)
        if ref_lus is not None:
            score("short-slate reference (M6-M8 on)", ref_lus, pick(ref_lus, L.scores(ref_lus, fp), N), ref_best, np.nan)
    else:
        ref_best = None
    for name, (_, _, kw, only) in LINEUP.items():
        if only and sl["slate"] not in only:
            continue
        if only:
            kw = {**SHORT_REF, **kw}
        lus, b, r = build(kw)
        if lus is None:
            continue
        fol = follows(name, best, P, set(tot.index[:2]))
        score(name, lus, pick(lus, L.scores(lus, fp), N), b, fol, ref_best if only else None)
    props = L.shapes(base_lus, P)
    bprops = L.shapes(best[None, :], P)
    for name, (rid, _, kw) in PORT.items():
        if rid == "SS5" and not short:
            continue
        ext = Extra(P.pos, props, kw.get("bands", {}), kw.get("pos_caps", {}), kw.get("min_unique", 1))
        ch = pick(base_lus, sc, N, kw.get("cap", 0.5), ext)
        pf = {"M4": bool(bprops["bringback"][0]), "M11": not bprops["te_flex"][0]}.get(rid, np.nan)
        score(name, base_lus, ch, base_best, pf)
    return rows


def summarize(r: pd.DataFrame) -> pd.DataFrame:
    r = r.assign(w=r["season"].map(B.weight))
    base = r[r["rule"] == "baseline"].set_index(["season", "week", "slate"])
    sref = r[r["rule"] == "short-slate reference (M6-M8 on)"].set_index(["season", "week", "slate"])
    meta = {**{k: (v[0], v[1]) for k, v in LINEUP.items()}, **{k: (v[0], v[1]) for k, v in PORT.items()}}
    out = []
    for name, g in r[~r["rule"].isin(["baseline", "short-slate reference (M6-M8 on)"])].groupby("rule"):
        g = g.set_index(["season", "week", "slate"])
        ref = sref if name.startswith(("SS1", "SS3", "SS4")) else base
        j = g.join(ref[["best_pct", "third_pct"]], rsuffix="_ref", how="inner")
        row = {"rule": name, "id": meta[name][0], "evidence": meta[name][1], "slates": len(j),
               "real best lineup follows it": round(100 * float(np.nanmean(g["optimal_follows"].astype(float))), 0)
               if g["optimal_follows"].notna().any() else np.nan,
               "pts lost forcing it": round(float(np.nanmean(g["pts_lost_forcing"])), 1)}
        for label, d, w in (("all", j, j["w"]), ("2026", j[j.index.get_level_values(0) == 2026], None)):
            if not len(d):
                continue
            for m in ("best_pct", "third_pct"):
                diff = 100 * (d[m] - d[m + "_ref"])
                ww = d["w"] if w is not None else np.ones(len(d))
                mean = float(np.average(diff, weights=ww))
                se = float(np.sqrt(np.average((diff - mean) ** 2, weights=ww) / max(len(d) - 1, 1)))
                row[f"{label}: {'best' if m == 'best_pct' else '3rd best'} vs ref (pts of %)"] = round(mean, 2)
                row[f"{label}: {'best' if m == 'best_pct' else '3rd best'} SE"] = round(se, 2)
        m, s = row.get("all: best vs ref (pts of %)", 0), row.get("all: best SE", 1)
        m26, s26 = row.get("2026: best vs ref (pts of %)", 0), row.get("2026: best SE", 1)
        if len(j) < 10:
            row["verdict"] = "too few slates to tell"
        else:
            row["verdict"] = ("hurts" if (m < -2 * s or (m26 < -2 * s26 and s26 > 0)) else
                              "helps" if m > 2 * s else "held up (no harm)")
        out.append(row)
    return pd.DataFrame(out).sort_values("id")


def main(argv=None):
    from nflmain.parallel import pmap
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="first N slates only (testing)")
    a = ap.parse_args(argv)
    g, allp = B.load()
    allp["tgt"] = trailing_tgt(allp)
    sls = B.slates(g)
    if a.limit:
        sls = sls[-a.limit:]
    jobs = [(sl, B.pool_for(sl, allp).assign(tgt=lambda d: d["tgt"] if "tgt" in d else 0.0), 7 + i, a.worlds)
            for i, sl in enumerate(sls)]
    res = pmap(_job, jobs, workers=a.workers) if a.workers > 1 else [_job(j) for j in jobs]
    r = pd.DataFrame([x for rr in res for x in rr])
    RES.mkdir(parents=True, exist_ok=True)
    r.to_csv(RES / "rules_raw.csv", index=False)
    s = summarize(r)
    s.to_csv(RES / "rules_by_rule.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
