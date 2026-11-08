"""Late swap and news re-evaluation: re-optimize only the open (unlocked) spots of DK entries.

Lock rule: a player whose game has kicked off (at or after its listed kickoff, or DK shows no time
for it any more) can never be moved. Locked players stay in their DK slot.

For every entry: keep the locked players; generate fills for the open slots (best fill under the
remaining salary in many simulated slates - conditioned on live scores for games already played);
score every option against the contest's simulated field and payouts; keep only options that beat
the current lineup on an INDEPENDENT set of simulations (paired, z bar adjusted for the number
tested); then pick one option per entry jointly under the 50% exposure cap. Entries holding a
player who won't play (OUT / inactive / projected 0) must change.

Game theory comes from the objective itself: the field is conditioned on the same live scores, so
an entry that is behind needs outcomes the field doesn't have (leans away from chalk) and an entry
that is ahead protects with what the field has (leans into chalk). Each change is explained with
the ownership of the players going out and coming in.
Option generation runs in spawned worker processes (nflmain/parallel.py).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from . import lineups as L
from .contests import evaluate, field_dupe_rate
from .parallel import pmap_init
from .settings import get as S
from .slate import CAP, ET, SLOT_POS, eligibility

_W: dict = {}


def locked_mask(pool: pd.DataFrame, now: datetime | None = None) -> np.ndarray:
    now = now or datetime.now(ET)
    k = pool["kickoff"].astype(str).to_numpy()
    out = np.zeros(len(pool), bool)
    for i, s in enumerate(k):
        out[i] = (not s) or datetime.fromisoformat(s) <= now
    return out


def _fill(points, open_slots, allowed, elig, sal, budget, team, game, opp, pos, team_room, game_room,
          dst_team=None, no_vs_dst=True):
    """Best players for the open slots (max points) among `allowed`, salary <= budget, team/game
    room counting the locked players, never a hitter facing our DST."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    players = np.flatnonzero(allowed)
    pairs = np.array([(i, s) for s in open_slots for i in players if elig[i, s]])
    if not len(pairs):
        return None
    m = len(pairs)
    A, lo, hi = [], [], []
    for s in open_slots:
        A.append((pairs[:, 1] == s).astype(float)); lo.append(1); hi.append(1)
    for i in np.unique(pairs[:, 0]):
        A.append((pairs[:, 0] == i).astype(float)); lo.append(0); hi.append(1)
    A.append(sal[pairs[:, 0]].astype(float)); lo.append(0); hi.append(budget)
    for t, room in team_room.items():
        row = (team[pairs[:, 0]] == t).astype(float)
        if row.any():
            A.append(row); lo.append(0); hi.append(max(room, 0))
    for g, room in game_room.items():
        row = (game[pairs[:, 0]] == g).astype(float)
        if row.any():
            A.append(row); lo.append(0); hi.append(max(room, 0))
    if no_vs_dst:
        off = np.isin(pos[pairs[:, 0]], ["QB", "RB", "WR", "TE"])
        if dst_team is not None:                       # locked DST: nobody facing it
            row = (off & (opp[pairs[:, 0]] == dst_team)).astype(float)
            if row.any():
                A.append(row); lo.append(0); hi.append(0)
        else:                                          # open DST: DST t and hitters facing t exclusive
            for k in np.flatnonzero(pos[pairs[:, 0]] == "DST"):
                t = team[pairs[k, 0]]
                row = (off & (opp[pairs[:, 0]] == t)).astype(float)
                row[k] = 8
                A.append(row); lo.append(0); hi.append(8)
    res = milp(-points[pairs[:, 0]].astype(float), constraints=[LinearConstraint(np.array(A), lo, hi)],
               integrality=np.ones(m), bounds=Bounds(0, 1), options={"time_limit": 5})
    if res.x is None:
        return None
    return {int(s): int(i) for i, s in pairs[res.x > 0.5]}


def _init(ctx):
    _W.update(ctx)


def _job(args):
    cur, seed = args
    return options_for_entry(cur, _W, np.random.default_rng(seed))


def options_for_entry(cur, ctx, rng):
    """Options (slot order) for one entry: current lineup first, then best fills of its open slots
    in sampled simulated slates (+ projection and ceiling)."""
    sal, pos, team, game, opp = ctx["sal"], ctx["pos"], ctx["team"], ctx["game"], ctx["opp"]
    locked, elig, proj, fp = ctx["locked"], ctx["elig"], ctx["proj"], ctx["fp"]
    complete = all(x is not None for x in cur)
    locked_slots = [k for k in range(9) if cur[k] is not None and locked[cur[k]]]
    open_slots = [k for k in range(9) if k not in locked_slots]
    opts = [list(cur)] if complete else []
    if not open_slots:
        return opts
    keep = [cur[k] for k in locked_slots]
    budget = CAP - int(sal[keep].sum()) if keep else CAP
    allowed = ~locked & ctx["allow"] & (proj > 0.3)
    allowed[keep] = False
    tmax, gmax = ctx["max_per_team"], ctx["game_max"]
    team_room = {t: tmax - int((team[keep] == t).sum()) for t in set(team)}
    game_room = {g: gmax - int((game[keep] == g).sum()) for g in set(game)}
    dst_locked = [i for i in keep if pos[i] == "DST"]
    dst_team = team[dst_locked[0]] if dst_locked else None
    seen = {tuple(sorted(cur))} if complete else set()
    worlds = [proj, ctx["ceil"]] + [fp[:, s] for s in rng.choice(fp.shape[1], ctx["n_opts"], replace=False)]
    for pts in worlds:
        ok = np.flatnonzero(allowed & elig[:, open_slots].any(axis=1) & (sal <= budget))
        top = ok[np.argsort(-pts[ok])[:max(40, 6 * len(open_slots))]]
        mask = np.zeros(len(sal), bool)
        mask[top] = True
        fill = _fill(pts, open_slots, mask, elig, sal, budget, team, game, opp, pos, team_room, game_room,
                     dst_team, ctx["no_vs_dst"])
        if fill is None:
            continue
        lu = list(cur)
        for s, i in fill.items():
            lu[s] = i
        k = tuple(sorted(lu))
        if k not in seen and len(set(game[lu])) >= 2:
            seen.add(k)
            opts.append(lu)
    return opts


def build_options(currents, ctx, workers=None, seed=0) -> list[list[list[int]]]:
    workers = int(S("build.workers")) if workers is None else workers
    jobs = [(cur, seed + 31 * k) for k, cur in enumerate(currents)]
    if workers <= 1 or len(jobs) < 8:
        rng = np.random.default_rng(seed)
        return [options_for_entry(cur, ctx, rng) for cur, _ in jobs]
    return pmap_init(_job, jobs, _init, (ctx,), workers=workers)


def significant(opts, perA, perB, forced, margin, z_min, n_test):
    """Per entry: current lineup + options whose gain is confirmed on independent sims B."""
    from scipy.stats import norm
    z_val = float(norm.isf(norm.sf(z_min) / max(n_test, 1)))
    out, k = [], 0
    for o, fc in zip(opts, forced):
        a = perA[k:k + len(o)].astype(np.float64)
        b = perB[k:k + len(o)].astype(np.float64)
        k += len(o)
        sa, sb = a.mean(axis=1), b.mean(axis=1)
        if fc:                                   # must change: every option, ranked on both sets
            out.append([(j, float((sa[j] + sb[j]) / 2)) for j in range(1, len(o))] or [(0, float(sb[0]))])
            continue
        keep = [(0, float(sb[0]))]
        gain = sa[1:] - sa[0]
        for j in [x + 1 for x in np.argsort(-gain)[:n_test] if gain[x] > margin]:
            d = b[j] - b[0]
            se = d.std(ddof=1) / np.sqrt(len(d))
            if d.mean() > margin and d.mean() > z_val * se:
                keep.append((j, float(sb[j])))
        out.append(keep)
    return out


def _marginal(top1: np.ndarray, eo: list, has_cur: list) -> np.ndarray:
    """Per option per simulated slate: any-top-1% of (other entries' current lineups + this option)
    + top1_weight x option / n. Rows in the same flat order as the options."""
    w = float(S("portfolio.top1_weight"))
    n = len(eo)
    starts = np.cumsum([0] + [len(o) for o in eo])[:-1]
    cur = np.array([top1[s] if h else np.zeros(top1.shape[1], bool) for s, h in zip(starts, has_cur)])
    tot = cur.sum(axis=0)
    out = np.zeros(top1.shape, dtype=np.float32)
    for e, s in enumerate(starts):
        others = (tot - cur[e]) > 0
        rows = top1[s:s + len(eo[e])]
        out[s:s + len(eo[e])] = (others[None, :] | rows) + w * rows / n
    return out


def choose(entry_opts, scores, swappable, n_players, max_exposure, keep_bonus=1e-4):
    """One option per entry maximizing the summed objective, each swappable player in at most
    max_exposure of entries (locked players aren't counted - they can't move)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    E = len(entry_opts)
    flat = [(e, o) for e in range(E) for o in range(len(entry_opts[e]))]
    K = len(flat)
    obj = np.array([scores[e][o] + (keep_bonus if o == 0 else 0) for e, o in flat])
    A1 = np.zeros((E, K))
    for k, (e, _) in enumerate(flat):
        A1[e, k] = 1
    cnt = np.zeros((n_players, K))
    for k, (e, o) in enumerate(flat):
        for p in entry_opts[e][o]:
            cnt[p, k] += 1
    ps = [p for p in range(n_players) if swappable[p] and cnt[p].any()]
    cap = max(1, int(np.floor(max_exposure * E + 1e-9)))
    cons = [LinearConstraint(A1, 1, 1)]
    if ps and E >= 2:
        cons.append(LinearConstraint(cnt[ps], 0, cap))
    res = milp(-obj, constraints=cons, integrality=np.ones(K), bounds=Bounds(0, 1),
               options={"time_limit": 60, "mip_rel_gap": 0.002})
    if res.x is None:                      # caps can't hold (locked chalk): best per entry
        return [int(np.argmax([scores[e][o] for o in range(len(entry_opts[e]))])) for e in range(E)]
    pick = [0] * E
    for k, (e, o) in enumerate(flat):
        if res.x[k] > 0.5:
            pick[e] = o
    return pick


def reevaluate(pool: pd.DataFrame, fpA: np.ndarray, fpB: np.ndarray, currents: list, contests_of: list,
               contests: dict, fld: np.ndarray, allow: np.ndarray, rules, dead: np.ndarray,
               now: datetime | None = None, seed: int = 7) -> dict:
    """currents: per entry, 9 pool indices in slot order (None = empty); contests_of: contest id per
    entry. Returns {"lineups": per entry, "changed": [...], "locked": mask, "log": [...]}."""
    locked = locked_mask(pool, now)
    P = L.Pool.from_df(pool)
    proj = fpA.mean(axis=1)
    ctx = dict(sal=P.sal, pos=P.pos, team=P.team, game=P.game, opp=P.opp, locked=locked,
               elig=eligibility(P.pos), proj=proj, ceil=np.percentile(fpA, 85, axis=1).astype(np.float32),
               fp=fpA[:, :min(fpA.shape[1], 2000)], allow=allow & ~dead, max_per_team=rules.max_per_team,
               game_max=rules.game_max, no_vs_dst=rules.no_offense_vs_dst, n_opts=int(S("lateswap.options")))
    t0 = datetime.now()
    opts = build_options(currents, ctx, seed=seed)
    print(f"[lateswap] options for {len(opts)} entries in {(datetime.now() - t0).seconds}s "
          f"({int(locked.sum())} players locked)")
    own = pool["Ownership"].to_numpy(float)
    q = lambda x: np.prod(own[x] / 100, axis=1)
    margin, z, n_test = float(S("lateswap.margin")), float(S("portfolio.swap_z")), int(S("lateswap.confirm_options"))
    out = [list(c) if c is not None else None for c in currents]
    log = []
    for cid in dict.fromkeys(contests_of):
        c = contests[cid]
        ent = [k for k, x in enumerate(contests_of) if x == cid]
        eo = [opts[k] for k in ent]
        has = [len(o) > 0 for o in eo]
        ent = [k for k, h in zip(ent, has) if h]
        eo = [o for o in eo if o]
        if not eo:
            continue
        flat = np.array([sorted(x) for o in eo for x in o])
        dup = field_dupe_rate([L.key(x) for x in flat], [L.key(x) for x in fld], q(flat), q(fld))
        mA = evaluate(L.scores(flat, fpA), fld, fpA, c, dup)
        mB = evaluate(L.scores(flat, fpB), fld, fpB, c, dup)
        if c.large:
            # portfolio objective, as in the build: an option is worth what it adds to
            # P(at least one entry top 1%) given the OTHER entries' current lineups, plus
            # top1_weight x its own top-1% rate / n - so swaps don't all chase the same players
            perA = _marginal(mA["top1"], eo, has_cur=[currents[k] is not None for k in ent])
            perB = _marginal(mB["top1"], eo, has_cur=[currents[k] is not None for k in ent])
        else:
            w = float(S("contest.small_finish_w"))
            perA, perB = mA["cash"] + w * mA["beat"], mB["cash"] + w * mB["beat"]
        forced = [currents[k] is None or any(dead[p] and not locked[p] for p in currents[k]) for k in ent]
        sig = significant(eo, perA, perB, forced, margin, z, n_test)
        kept = [[eo[i][j] for j, _ in s] for i, s in enumerate(sig)]
        sc = [[v for _, v in s] for s in sig]
        pick = choose(kept, sc, ~locked, len(pool), float(S("portfolio.max_exposure")))
        names, ownr = pool["Name"].to_numpy(object), own
        for i, k in enumerate(ent):
            new = kept[i][pick[i]]
            old = currents[k]
            if old is None or sorted(new) != sorted(old):
                outs = [p for p in (old or []) if p not in new]
                ins = [p for p in new if p not in (old or [])]
                log.append({"entry": k, "contest": c.name, "out": [names[p] for p in outs],
                            "in": [names[p] for p in ins],
                            "own_out": float(ownr[outs].mean()) if outs else 0.0,
                            "own_in": float(ownr[ins].mean()) if ins else 0.0,
                            "forced": forced[i], "gain": float(sc[i][pick[i]] - (sc[i][0] if not forced[i] else 0))})
            out[k] = new
    return {"lineups": out, "log": log, "locked": locked}
