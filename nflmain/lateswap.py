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


def _fill(points, open_slots, allowed, ctx, keep, budget, floor, team_room, game_room):
    """Best players for the open slots (max points) among `allowed`, with the locked players `keep`
    counted: salary between the $49,500 floor and the cap, team/game room, RBs per team, the QB's
    pass-catchers (1-2), never a DST facing our QB, at most max_vs_own_dst hitters facing our DST."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    elig, sal, team, game, opp, pos = ctx["elig"], ctx["sal"], ctx["team"], ctx["game"], ctx["opp"], ctx["pos"]
    players = np.flatnonzero(allowed)
    pairs = np.array([(i, s) for s in open_slots for i in players if elig[i, s]])
    if not len(pairs):
        return None
    m = len(pairs)
    pi = pairs[:, 0]
    A, lo, hi = [], [], []

    def add(row, l, h):
        A.append(np.asarray(row, float)); lo.append(l); hi.append(h)

    for s in open_slots:
        add(pairs[:, 1] == s, 1, 1)
    for i in np.unique(pi):
        add(pi == i, 0, 1)
    add(sal[pi], max(floor, 0), budget)                            # M10 floor (hard) + cap
    for t, room in team_room.items():
        if (team[pi] == t).any():
            add(team[pi] == t, 0, max(room, 0))
    for g, room in game_room.items():
        if (game[pi] == g).any():
            add(game[pi] == g, 0, max(room, 0))
    kp = list(keep)
    skill = np.isin(pos[pi], ["RB", "WR", "TE"])
    pc = np.isin(pos[pi], ["WR", "TE"])
    M = 8
    for t in set(team[pi]) | set(team[kp]):                          # M8: RBs per team
        n_locked = sum(1 for i in kp if team[i] == t and pos[i] == "RB")
        row = (team[pi] == t) & (pos[pi] == "RB")
        if row.any():
            add(row, 0, max(ctx["rb_per_team"] - n_locked, 0))
    qb_locked = [i for i in kp if pos[i] == "QB"]
    dst_locked = [i for i in kp if pos[i] == "DST"]
    if qb_locked:                                                     # M2 (hard): <= 2 pass-catchers
        t = team[qb_locked[0]]
        have = sum(1 for i in kp if team[i] == t and pos[i] in ("WR", "TE"))
        row = pc & (team[pi] == t)
        if row.any():
            add(row, max(1 - have, 0), max(2 - have, 0))               # M1 (hard): >= 1
        elif have < 1:
            return None
    else:                                                             # open QB: for each candidate QB
        for k in np.flatnonzero(pos[pi] == "QB"):
            t = team[pi[k]]
            have = sum(1 for i in kp if team[i] == t and pos[i] in ("WR", "TE"))
            row = (pc & (team[pi] == t)).astype(float)
            r2 = row.copy(); r2[k] += M
            add(r2, -np.inf, 2 - have + M)                            # <= 2 when this QB is used
            r1 = row.copy(); r1[k] -= max(1 - have, 0)
            add(r1, 0, np.inf)                                        # >= 1 when this QB is used
    k_dst = ctx["max_vs_own_dst"]
    if dst_locked:
        d = team[dst_locked[0]]
        qrow = (pos[pi] == "QB") & (opp[pi] == d)                     # M5 (hard)
        if qrow.any():
            add(qrow, 0, 0)
        if any(pos[i] == "QB" and opp[i] == d for i in kp):
            return None
        n_vs = sum(1 for i in kp if opp[i] == d and pos[i] in ("RB", "WR", "TE"))
        row = skill & (opp[pi] == d)
        if row.any():
            add(row, 0, max(k_dst - n_vs, 0))                          # M6
    else:
        for k in np.flatnonzero(pos[pi] == "DST"):
            d = team[pi[k]]
            if any(pos[i] == "QB" and opp[i] == d for i in kp):
                add(np.arange(m) == k, 0, 0)                          # M5: our locked QB faces it
                continue
            qrow = ((pos[pi] == "QB") & (opp[pi] == d)).astype(float)
            qrow[k] += 1
            add(qrow, -np.inf, 1)                                      # M5
            n_vs = sum(1 for i in kp if opp[i] == d and pos[i] in ("RB", "WR", "TE"))
            row = (skill & (opp[pi] == d)).astype(float)
            row[k] += M - max(k_dst - n_vs, 0)
            add(row, -np.inf, M)                                       # M6
    res = milp(-points[pi].astype(float), constraints=[LinearConstraint(np.array(A), lo, hi)],
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
    floor = ctx["min_salary"] - (int(sal[keep].sum()) if keep else 0)
    seen = {tuple(sorted(cur))} if complete else set()
    worlds = [proj, ctx["ceil"]] + [fp[:, s] for s in rng.choice(fp.shape[1], ctx["n_opts"], replace=False)]
    for pts in worlds:
        ok = np.flatnonzero(allowed & elig[:, open_slots].any(axis=1) & (sal <= budget))
        top = ok[np.argsort(-pts[ok])[:max(40, 6 * len(open_slots))]]
        mask = np.zeros(len(sal), bool)
        mask[top] = True
        fill = _fill(pts, open_slots, mask, ctx, keep, budget, floor, team_room, game_room)
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


def choose(entry_opts, scores, swappable, n_players, max_exposure, keep_bonus=1e-4, pos=None, sal=None, exc=None):
    """One option per entry maximizing the summed objective, each swappable player in at most
    max_exposure of entries (locked players aren't counted - they can't move), and never the same
    lineup in two entries (Kenny's rule)."""
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
    capv = np.full(n_players, cap)
    if pos is not None:                       # workbook M15: QB / DST exposure caps
        for p_, key in (("QB", "rules.qb_max_exposure"), ("DST", "rules.dst_max_exposure")):
            capv[pos == p_] = np.minimum(cap, max(1, int(np.floor(float(S(key)) * E + 1e-9))))
    if sal is not None:                       # Kenny 11/08: players at or under $4,500 at most 30%
        cheap = np.asarray(sal) <= float(S("rules.cheap_salary"))
        capv[cheap] = np.minimum(capv[cheap], max(1, int(np.floor(float(S("rules.cheap_max_exposure")) * E + 1e-9))))
    for i, frac in (exc or {}).items():       # research-argued mispricing exceptions
        if pos is None or pos[i] not in ("QB", "DST"):
            capv[i] = min(cap, max(1, int(np.floor(frac * E + 1e-9))))
    cons = [LinearConstraint(A1, 1, 1)]
    if ps and E >= 2:
        cons.append(LinearConstraint(cnt[ps], 0, capv[ps]))
    by_key: dict = {}
    for k, (e, o) in enumerate(flat):
        by_key.setdefault(tuple(sorted(entry_opts[e][o])), set()).add((e, k))
    dup_rows = []
    for key, members in by_key.items():
        if len({e for e, _ in members}) > 1:            # the same lineup offered to 2+ entries
            row = np.zeros(K)
            row[[k for _, k in members]] = 1
            dup_rows.append(row)
    if dup_rows:
        cons.append(LinearConstraint(np.array(dup_rows), 0, 1))
    res = milp(-obj, constraints=cons, integrality=np.ones(K), bounds=Bounds(0, 1),
               options={"time_limit": 60, "mip_rel_gap": 0.002})
    if res.x is None:                      # caps can't hold (locked chalk): best per entry
        return [int(np.argmax([scores[e][o] for o in range(len(entry_opts[e]))])) for e in range(E)]
    pick = [0] * E
    for k, (e, o) in enumerate(flat):
        if res.x[k] > 0.5:
            pick[e] = o
    return pick


def _legal(lu, P, ctx) -> bool:
    """DK rules + QB with 1-2 of his pass-catchers + salary floor + no QB facing our DST."""
    lu = np.asarray(lu)
    if not L.valid(lu, P) or P.sal[lu].sum() < ctx["min_salary"]:
        return False
    q = lu[P.pos[lu] == "QB"][0]
    d = lu[P.pos[lu] == "DST"][0]
    mates = sum(1 for i in lu if P.team[i] == P.team[q] and P.pos[i] in ("WR", "TE"))
    return 1 <= mates <= 2 and P.opp[q] != P.team[d]


def reevaluate(pool: pd.DataFrame, fpA: np.ndarray, fpB: np.ndarray, currents: list, contests_of: list,
               contests: dict, fld: np.ndarray, allow: np.ndarray, rules, dead: np.ndarray,
               now: datetime | None = None, seed: int = 7, exc: dict | None = None) -> dict:
    """currents: per entry, 9 pool indices in slot order (None = empty); contests_of: contest id per
    entry. Returns {"lineups": per entry, "changed": [...], "locked": mask, "log": [...]}."""
    locked = locked_mask(pool, now)
    P = L.Pool.from_df(pool)
    proj = fpA.mean(axis=1)
    ctx = dict(sal=P.sal, pos=P.pos, team=P.team, game=P.game, opp=P.opp, locked=locked,
               elig=eligibility(P.pos), proj=proj, ceil=np.percentile(fpA, 85, axis=1).astype(np.float32),
               fp=fpA[:, :min(fpA.shape[1], 2000)], allow=allow & ~dead, max_per_team=rules.max_per_team,
               game_max=rules.game_max, max_vs_own_dst=rules.max_vs_own_dst, rb_per_team=rules.rb_per_team,
               min_salary=rules.min_salary,
               # before the first lock every spot is open (a full rebuild per entry): half the options
               # keeps the check under 3 minutes for 190 entries; after a lock few spots are open
               n_opts=int(S("lateswap.options")) if locked.any() else max(10, int(S("lateswap.options")) // 2))
    t0 = datetime.now()
    opts = build_options(currents, ctx, seed=seed)
    print(f"[lateswap] options for {len(opts)} entries in {(datetime.now() - t0).seconds}s "
          f"({int(locked.sum())} players locked)")
    own = pool["Ownership"].to_numpy(float)
    q = lambda x: np.prod(own[x] / 100, axis=1)
    margin, z, n_test = float(S("lateswap.margin")), float(S("portfolio.swap_z")), int(S("lateswap.confirm_options"))
    out = [list(c) if c is not None else None for c in currents]
    log = []
    taken: set = set()                     # lineups already used by entries of earlier contests
    for cid in dict.fromkeys(contests_of):
        c = contests[cid]
        ent = [k for k, x in enumerate(contests_of) if x == cid]
        eo = [opts[k] for k in ent]
        has = [len(o) > 0 for o in eo]
        ent = [k for k, h in zip(ent, has) if h]
        eo = [o for o in eo if o]
        # options that would repeat a lineup another contest's entry already has are dropped
        eo = [[x for j, x in enumerate(o) if j == 0 or tuple(sorted(x)) not in taken] for o in eo]
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
        pick = choose(kept, sc, ~locked, len(pool), float(S("portfolio.max_exposure")), pos=P.pos, sal=P.sal, exc=exc)
        names, ownr = pool["Name"].to_numpy(object), own
        for i, k in enumerate(ent):
            new = kept[i][pick[i]]
            old = currents[k]
            if old is not None and sorted(new) != sorted(old) and not _legal(new, P, ctx):   # safety net
                print(f"[lateswap] entry {k}: the new lineup broke a rule - kept the current one")
                new = old
            taken.add(tuple(sorted(new)))
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
