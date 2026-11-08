"""(Adapted from the NFL Showdown repo's portfolio picker, 11/07/2026.)
Pick a set of lineups for one contest as a portfolio.

Large fields (contest.large_field+ entries): maximize P(at least one lineup finishes top 1%)
+ top1_weight x the average top-1% rate, so lineups that hit on DIFFERENT simulated slates are
worth more. Small fields: cash rate + small_finish_w x share of the field beaten.
Caps: each player in at most max_exposure (50%) of lineups; initial build only, contests with
10+ entries: each player in 0 or at least min_count (5) lineups. Each QB stack team at most
max_exposure too (diversifies game stacks).
Then swaps: a lineup is replaced only if the replacement is better on the build sims AND
confirmed on an independent set of sims by more than swap_z standard errors.
"""
from __future__ import annotations

import numpy as np

from .settings import get as S

MIN_COUNT_ENTRIES = 10


class Extra:
    """Kenny's workbook portfolio rules for one candidate set (lineups.py has the lineup rules):
    per-position exposure caps (M15: QB 25%, DST 15%), shares of lineup shapes (M1 double stacks
    ~50%, M4 one bring-back ~50%, M11 TE in FLEX <= 10%) and lineups differing by 2+ (M16).
    pos: position of every player (by pool index); props: {name: bool per candidate}."""

    def __init__(self, pos=None, props=None, bands=None, pos_caps=None, min_unique=None, sal=None,
                 cheap=None):
        self.pos = pos
        self.sal = sal                         # salaries by pool index (for the cheap-player cap)
        self.cheap = cheap                     # (salary at or under, max share) or None
        self.props = props or {}
        self.bands = bands or {}               # name -> (low share, high share)
        self.pos_caps = pos_caps or {}         # position -> max share
        self.min_unique = int(S("rules.min_unique")) if min_unique is None else min_unique

    @classmethod
    def from_settings(cls, pos, props, sal=None):
        b = float(S("rules.share_band"))
        bands = {}
        for name, key in (("double", "rules.double_stack_share"), ("bringback", "rules.bringback_share")):
            t = float(S(key))
            if b < 1.0:
                bands[name] = (max(t - b, 0.0), min(t + b, 1.0))
        bands["te_flex"] = (0.0, float(S("rules.flex_te_max")))
        bands["rb_flex"] = (0.0, float(S("rules.flex_rb_max")))
        caps_ = {"QB": float(S("rules.qb_max_exposure")), "DST": float(S("rules.dst_max_exposure"))}
        cheap = (float(S("rules.cheap_salary")), float(S("rules.cheap_max_exposure")))
        return cls(pos, props, bands, caps_, sal=sal, cheap=cheap)

    def player_caps(self, n: int, k_min: int, P: int) -> np.ndarray:
        """Most lineups (of n) each player may be in: the 50% cap, tighter for QB / DST, never below
        the 0-or-5+ minimum (so a QB/DST cap can't make the 0-or-5+ rule impossible)."""
        cap = np.full(P, caps(n))
        floor_ = max(k_min, min_count(n, True), 2)      # never below the 0-or-5+ minimum, nor 2
        if self.pos is not None:
            for p_, frac in self.pos_caps.items():
                m = np.flatnonzero(self.pos[:P] == p_)
                cap[m] = np.minimum(cap[m], max(floor_, int(np.floor(frac * n + 1e-9))))
        if self.sal is not None and self.cheap is not None:   # Kenny 11/08: cheap players are boom-or-bust
            m = np.flatnonzero(np.asarray(self.sal[:P]) <= self.cheap[0])
            cap[m] = np.minimum(cap[m], max(floor_, int(np.floor(self.cheap[1] * n + 1e-9))))
        return cap

    def band_counts(self, n: int) -> dict:
        return {k: (int(np.ceil(lo * n - 1e-9)), int(np.floor(hi * n + 1e-9))) for k, (lo, hi) in self.bands.items()
                if k in self.props}

    def without_bands(self, keep=()):
        return Extra(self.pos, self.props, {k: v for k, v in self.bands.items() if k in keep},
                     self.pos_caps, self.min_unique, self.sal, self.cheap)


NO_EXTRA = Extra(min_unique=2)


def _w() -> float:
    return float(S("portfolio.top1_weight"))


def _small_w() -> float:
    return float(S("contest.small_finish_w"))


def per_sim_value(metrics: dict, idx: list[int], large: bool) -> np.ndarray:
    n = max(len(idx), 1)
    if large:
        t = metrics["top1"][idx]
        return t.any(axis=0).astype(float) + _w() * t.mean(axis=0) * len(idx) / n
    return (metrics["cash"][idx].astype(float) + _small_w() * metrics["beat"][idx]).mean(axis=0)


def caps(n: int) -> int:
    return max(1, int(np.floor(float(S("portfolio.max_exposure")) * n + 1e-9)))


def min_count(n: int, initial: bool = True) -> int:
    k = int(S("portfolio.min_count"))
    return k if (initial and k > 1 and n >= MIN_COUNT_ENTRIES) else 1


def meets_min_count(idx, lus, n, initial=True) -> bool:
    k = min_count(n, initial)
    if k <= 1 or not len(idx):
        return True
    cnt = np.bincount(lus[list(idx)].ravel())
    return bool((cnt[cnt > 0] >= k).all())


def _ok(j, chosen, lus, cap, stack=None, extra=None, n=None):
    """Can lineup j join `chosen` without breaking a cap? (cap: int or per-player array)"""
    if chosen:
        cnt = np.bincount(lus[chosen].ravel(), minlength=lus.max() + 1)
        capv = np.broadcast_to(np.asarray(cap), (lus.max() + 1,)) if np.ndim(cap) else np.full(lus.max() + 1, cap)
        if any(cnt[p] + 1 > capv[p] for p in lus[j]):
            return False
        if stack is not None and sum(stack[c] == stack[j] for c in chosen) + 1 > np.max(capv):
            return False
        if extra is not None:
            mu = extra.min_unique
            if mu > 1 and any(len(set(lus[j]) & set(lus[c])) > lus.shape[1] - mu for c in chosen):
                return False
            for k, (lo, hi) in extra.band_counts(n or len(chosen) + 1).items():
                if extra.props[k][j] and sum(extra.props[k][c] for c in chosen) + 1 > hi:
                    return False
    return True


def _value(metrics, large, n):
    if large:
        v = metrics["top1"].mean(axis=1) * (1 + _w() / n)
    else:
        v = metrics["cash"].mean(axis=1) + _small_w() * metrics["beat"].mean(axis=1)
    return v + 1e-6 * metrics["pay"].mean(axis=1)


def exact(metrics, lus, n, large, exclude=(), initial=True, stack=None, time_limit=30, extra=None) -> list[int]:
    """All n lineups at once (MILP): best total stand-alone value under the caps; [] if infeasible."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, diags, hstack, identity
    L = len(lus)
    cap, k = caps(n), min_count(n, initial)
    val = _value(metrics, large, n)
    P = int(lus.max()) + 1
    capv = extra.player_caps(n, k, P) if extra is not None else np.full(P, cap)
    rows = lus.ravel()
    cols = np.repeat(np.arange(L), lus.shape[1])
    A_p = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(P, L))
    extra = []
    if stack is not None:
        kinds = sorted(set(stack))
        A_s = csr_matrix(np.array([[1.0 if stack[j] == kk else 0.0 for j in range(L)] for kk in kinds]))
        extra.append(("s", A_s))
    nz = P if k > 1 else 0
    def pad(A):
        return hstack([A, csr_matrix((A.shape[0], nz))]).tocsr() if nz else A
    if k > 1:
        Z = identity(P, format="csr")
        cons = [LinearConstraint(np.r_[np.ones(L), np.zeros(P)][None, :], n, n),
                LinearConstraint(hstack([A_p, -diags(capv.astype(float))]).tocsr(), -np.inf, 0),
                LinearConstraint(hstack([A_p, -k * Z]).tocsr(), 0, np.inf)]
    else:
        cons = [LinearConstraint(np.ones((1, L)), n, n), LinearConstraint(A_p, 0, capv)]
    for _, A in extra:
        cons.append(LinearConstraint(pad(A), 0, cap))
    ub = np.ones(L + nz)
    ub[list(exclude)] = 0
    res = milp(-np.r_[val, np.zeros(nz)], constraints=cons, integrality=np.ones(L + nz),
               bounds=Bounds(0, ub), options={"time_limit": time_limit})
    if res.x is None:
        return []
    x = np.flatnonzero(np.rint(res.x[:L]) == 1)
    return [int(j) for j in x[np.argsort(-val[x])]] if len(x) == n else []


def greedy(metrics, lus, n, large, exclude=(), stack=None, extra=None) -> list[int]:
    cap = extra.player_caps(n, 1, int(lus.max()) + 1) if extra is not None else caps(n)
    chosen: list[int] = []
    any_hit = np.zeros(metrics["top1"].shape[1], dtype=bool)
    rate = metrics["top1"].mean(axis=1)
    small = metrics["cash"].mean(axis=1) + _small_w() * metrics["beat"].mean(axis=1)
    banned = set(exclude)
    while len(chosen) < n:
        gain = (metrics["top1"] & ~any_hit).mean(axis=1) + _w() * rate / n if large else small
        pick = None
        for j in np.argsort(-gain):
            if j not in banned and j not in chosen and _ok(int(j), chosen, lus, cap, stack, extra, n):
                pick = int(j)
                break
        if pick is None:
            break
        chosen.append(pick)
        any_hit |= metrics["top1"][pick]
    return chosen


def joint(metrics, lus, n, large, exclude=(), k_min=1, stack=None, pool=700, min_unique=None,
          time_limit=90, cover_sims=1500, extra=None) -> list[int]:
    """(NBA pick_portfolio_milp, adapted.) Choose all n lineups at once from the best `pool`
    candidates: every player unused or in >= k_min lineups, none above the exposure cap, each QB
    stack team at most the cap, lineups differ by >= min_unique players. Large fields maximize
    P(at least one lineup top 1%) + top1_weight x average top-1% rate over `cover_sims` simulated
    slates; small fields the summed stand-alone value. [] if infeasible."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, csr_matrix, diags, hstack, identity
    val = _value(metrics, large, n)
    ex_set = set(exclude)
    ranked = [j for j in np.argsort(-val) if j not in ex_set]
    order = ranked[:pool]
    if extra is not None and extra.pos is not None and extra.pos_caps:
        # with per-QB / per-DST caps the best `pool` lineups can all share a few QBs (e.g. one cheap
        # QB stack dominates), leaving too few lineups to fill n under the caps: also take each
        # QB's and each DST's best lineups so every capped position has enough to choose from
        have = set(order)
        capv_all = extra.player_caps(n, k_min, int(lus.max()) + 1)
        for p_ in extra.pos_caps:
            by: dict = {}
            for j in ranked:
                for pl in lus[j]:
                    if extra.pos[pl] == p_:
                        lst = by.setdefault(int(pl), [])
                        if len(lst) < 3 * int(capv_all[pl]):
                            lst.append(j)
            for lst in by.values():
                for j in lst:
                    if j not in have:
                        have.add(j)
                        order.append(j)
    top = np.array(order)
    K = len(top)
    if K < n:
        return []
    Cn = lus[top]
    players = np.unique(Cn)
    Pn = len(players)
    pidx = {p: i for i, p in enumerate(players)}
    extra = extra or NO_EXTRA
    min_unique = extra.min_unique if min_unique is None else min_unique
    cap = max(k_min, caps(n))
    capv = extra.player_caps(n, k_min, int(lus.max()) + 1)[players]
    rows = np.array([pidx[p] for lu in Cn for p in lu])
    M = coo_matrix((np.ones(len(rows)), (rows, np.repeat(np.arange(K), Cn.shape[1]))), shape=(Pn, K)).tocsr()
    Sn = 0
    if large:
        rng = np.random.default_rng(0)
        sims = rng.choice(metrics["top1"].shape[1], size=min(cover_sims, metrics["top1"].shape[1]), replace=False)
        T = csr_matrix(metrics["top1"][top][:, sims].T.astype(float))
        T = T[np.asarray(T.sum(axis=1)).ravel() > 0]           # slates where no candidate hits add nothing
        Sn = T.shape[0]
    nv = K + Pn + Sn
    Z = lambda r, c: csr_matrix((r, c))
    cons = [LinearConstraint(hstack([csr_matrix(np.ones((1, K))), Z(1, Pn + Sn)]).tocsr(), n, n),
            LinearConstraint(hstack([M, -diags(capv.astype(float)), Z(Pn, Sn)]).tocsr(), -np.inf, 0),
            LinearConstraint(hstack([M, -k_min * identity(Pn), Z(Pn, Sn)]).tocsr(), 0, np.inf)]
    if stack is not None:
        st = np.array([stack[j] for j in top], dtype=object)
        kinds = sorted(set(st))
        A = coo_matrix((np.ones(K), ([kinds.index(s) for s in st], np.arange(K))), shape=(len(kinds), K))
        cons.append(LinearConstraint(hstack([A, Z(len(kinds), Pn + Sn)]).tocsr(), 0, cap))
    for k, (lo_c, hi_c) in extra.band_counts(n).items():          # shape shares (M1, M4, M11)
        row = np.asarray(extra.props[k])[top].astype(float)
        cons.append(LinearConstraint(hstack([csr_matrix(row[None, :]), Z(1, Pn + Sn)]).tocsr(), lo_c, hi_c))
    if min_unique > 1:
        shared = (M.T @ M).toarray()
        ii, jj = np.nonzero(np.triu(shared > Cn.shape[1] - min_unique, k=1))
        if len(ii):
            r = np.r_[np.arange(len(ii)), np.arange(len(ii))]
            cons.append(LinearConstraint(coo_matrix((np.ones(2 * len(ii)), (r, np.r_[ii, jj])),
                                                    shape=(len(ii), nv)).tocsr(), -np.inf, 1))
    if Sn:
        cons.append(LinearConstraint(hstack([-T, Z(Sn, Pn), identity(Sn)]).tocsr(), -np.inf, 0))
        rate = metrics["top1"][top].mean(axis=1)
        S_all = len(sims)
        c = np.r_[-_w() * rate / n, np.zeros(Pn), -np.ones(Sn) / S_all]
    else:
        c = np.r_[-val[top], np.zeros(Pn)]
    integ = np.r_[np.ones(K + Pn), np.zeros(Sn)]
    sol = milp(c, constraints=cons, integrality=integ, bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.005})
    if sol.x is None:
        return []
    x = np.flatnonzero(sol.x[:K] > 0.5)
    return [int(top[j]) for j in x[np.argsort(-val[top][x])]] if len(x) == n else []


def select(metrics, lus, n, large, exclude=(), initial=True, stack=None, quiet=False, time_limit=20,
           extra=None) -> list[int]:
    """Whole-portfolio MILP first (0-or-5+ on the initial build, 50% cap, stack-team cap); if the
    0-or-5+ rule can't be met from these candidates it steps down (5 -> 3 -> off) and says so.
    Greedy picker as the last resort."""
    k0 = min_count(n, initial)
    if n >= 2:
        for k in sorted({k0, min(k0, 3), 1}, reverse=True):
            # if every shape share can't be met: drop the double-stack / run-back targets first and keep
            # the FLEX limits (RB / TE in FLEX), then drop those too
            ex_list = [extra] if extra is None else [extra, extra.without_bands(("rb_flex", "te_flex")),
                                                     extra.without_bands()]
            for ex in ex_list:
                for st in ((stack, None) if stack is not None else (None,)):
                    ch = joint(metrics, lus, n, large, exclude, k, st, time_limit=time_limit, extra=ex)
                    if ch:
                        if k < k0 and not quiet:
                            print(f"[portfolio] the 0-or-{k0}+ rule couldn't be met from these candidates; used 0-or-{k}+")
                        if extra is not None and ex is not extra and not quiet:
                            print("[portfolio] the workbook's lineup-shape shares couldn't all be met; picked without them")
                        return ch
    ch = greedy(metrics, lus, n, large, exclude, stack, extra)
    if not meets_min_count(ch, lus, n, initial):
        ch = []
    ex = exact(metrics, lus, n, large, exclude, initial, stack, extra=extra) if n >= 2 else []
    if ex and (len(ch) < n or per_sim_value(metrics, ex, large).mean() > per_sim_value(metrics, ch, large).mean()):
        return ex
    if len(ch) < n and stack is not None:          # relax the stack-team cap before anything else
        return select(metrics, lus, n, large, exclude, initial, None, quiet, time_limit, extra)
    if len(ch) < n and not quiet:
        print(f"[portfolio] only {len(ch)} lineups fit the exposure caps")
    return ch


def confirm_swaps(chosen, mA, mB, lus, large, exclude=(), initial=True, stack=None, extra=None):
    """Try replacing each lineup with the best alternative on sims A; keep it only if it also wins
    on independent sims B by > swap_z standard errors (paired over the same simulated slates)."""
    n = len(chosen)
    cap = extra.player_caps(n, min_count(n, initial), int(lus.max()) + 1) if extra is not None else caps(n)
    z = float(S("portfolio.swap_z"))
    tries = int(S("portfolio.swap_tries"))
    log, cur = [], list(chosen)
    for slot in range(n):
        others = cur[:slot] + cur[slot + 1:]
        base_a = per_sim_value(mA, cur, large).mean()
        if large:
            oth = mA["top1"][others].any(axis=0) if others else np.zeros(mA["top1"].shape[1], bool)
            g = (mA["top1"] & ~oth).mean(axis=1) + _w() * mA["top1"].mean(axis=1) / n
        else:
            g = mA["cash"].mean(axis=1) + _small_w() * mA["beat"].mean(axis=1)
        for j in np.argsort(-g)[:tries]:
            j = int(j)
            if j in cur or j in exclude or not _ok(j, others, lus, cap, stack, extra, n):
                continue
            trial = others[:slot] + [j] + others[slot:]
            if extra is not None and any(sum(extra.props[k][t] for t in trial) < min(lo, sum(extra.props[k][t] for t in cur))
                                         for k, (lo, hi) in extra.band_counts(n).items()):
                continue
            if not meets_min_count(trial, lus, n, initial):
                continue
            if per_sim_value(mA, trial, large).mean() <= base_a + 1e-12:
                break
            d = per_sim_value(mB, trial, large) - per_sim_value(mB, cur, large)
            se = d.std(ddof=1) / np.sqrt(len(d))
            ok = bool(d.mean() > z * se and d.mean() > 0)
            log.append({"slot": slot, "out": cur[slot], "in": j, "gain_B": float(d.mean()), "se_B": float(se),
                        "accepted": ok})
            if ok:
                cur = trial
            break
    return cur, log
