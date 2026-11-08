"""Classic lineups: DK rules, the stacking optimizer (candidate builds) and the simulated field.

A lineup is an int array of 9 player row indices. The set is what scores; slot order is decided
only when writing the upload file (`assign_slots`: the latest-kickoff player goes to FLEX when he
can, so late swap keeps the most freedom).

DK rules: QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE), DST; $50,000 cap; players from 2+ games.
Our rules (settings): QB stacked with stack_min..stack_max pass-catchers, bring-backs per slate
type, max players per team / per game, no QB/RB/WR/TE facing our own DST.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .slate import CAP, SLOT_POS, SLOTS

ROSTER = 9


def scores(lus: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """(L, 9) lineups x (n, S) points -> (L, S)."""
    lus = np.atleast_2d(lus)
    out = np.zeros((len(lus), fp.shape[1]), dtype=np.float32)
    for k in range(lus.shape[1]):
        out += fp[lus[:, k]]
    return out


def salary(lus: np.ndarray, sal: np.ndarray) -> np.ndarray:
    return sal[np.atleast_2d(lus)].sum(axis=1).astype(int)


def canon(lus: np.ndarray) -> np.ndarray:
    return np.sort(np.atleast_2d(lus), axis=1)


def key(lu) -> tuple:
    return tuple(sorted(int(x) for x in lu))


# ---- Kenny's HARD rules (workbook research/rules, graded Strong; never broken, not settings) ----
HARD_MIN_SALARY = 49500     # M10: spend at least $49,500
HARD_STACK_MAX = 2          # M2: at most 2 same-team pass-catchers with the QB
HARD_STACK_MIN = 1          # M1: the QB always has at least 1 same-team WR/TE
HARD_NO_DST_VS_QB = True    # M5: never a DST facing our own QB
HARD_CHALK = (40.0, 1)      # M14: at most 1 player owned 40%+ per lineup (projected ownership)


@dataclass
class Pool:
    """What the optimizer needs to know about each player (same row order as the projections)."""
    sal: np.ndarray
    pos: np.ndarray        # QB / RB / WR / TE / DST
    team: np.ndarray
    opp: np.ndarray
    game: np.ndarray
    kickoff: np.ndarray    # iso strings (sortable)
    tgt: np.ndarray | None = None   # target share (for "receiving back" in M7)
    own: np.ndarray | None = None   # projected ownership % (for M14)

    @classmethod
    def from_df(cls, df):
        g = lambda c: df[c].to_numpy(float) if c in df else None
        return cls(df["Salary"].to_numpy(float), df["pos"].to_numpy(object), df["team"].to_numpy(object),
                   df["opp"].to_numpy(object), df["game"].to_numpy(object), df["kickoff"].astype(str).to_numpy(object),
                   g("tgt_share"), g("Ownership"))


@dataclass
class Rules:
    """Lineup rules. Hard rules (constants above) always apply; the rest are settings whose defaults
    are Kenny's workbook values (config/settings.yaml, groups rules.* and per slate type)."""
    stack_min: int = 1
    stack_max: int = 2
    bringback: int = 1          # 0 off, 1 allowed (candidates with and without), 2 required
    max_per_team: int = 4       # M8 (SS1: 5 on short slates)
    game_max: int = 5
    max_vs_own_dst: int = 1     # M6: skill players facing our DST (QBs never: M5)
    min_salary: int = HARD_MIN_SALARY
    rb_per_team: int = 1        # M8 (SS2: 2 on short slates)
    no_qb_team_max: int = 2     # M8: skill players from a team whose QB we don't use
    qb_own_rb: int = 1          # M7: 0 never, 1 only receiving backs, 2 allowed (SS1)
    rb_min_tgt: float = 0.12    # M7: "receiving back" = target share at least this
    max_rb_runback: int = 8     # M4: RBs from our QB's opponent (run-backs should be WR/TE)
    max_cheap: int = 9          # Kenny 11/08: QB/RB/WR at or under cheap_salary per lineup
    cheap_salary: float = 4500
    hard: bool = True           # False only in research (e.g. the real best lineup, no rules at all)

    def __post_init__(self):            # hard rules can't be loosened by settings
        if not self.hard:
            return
        self.stack_min = max(self.stack_min, HARD_STACK_MIN)
        self.stack_max = min(max(self.stack_max, self.stack_min), HARD_STACK_MAX)
        self.min_salary = max(int(self.min_salary), HARD_MIN_SALARY)

    @classmethod
    def for_slate(cls, slate_type: str):
        from .settings import for_type, get
        return cls(int(for_type(slate_type, "stack_min")), int(for_type(slate_type, "stack_max")),
                   int(for_type(slate_type, "bringback")), int(for_type(slate_type, "max_per_team")),
                   int(for_type(slate_type, "game_max")), int(for_type(slate_type, "max_vs_own_dst")),
                   int(get("build.min_salary")), int(for_type(slate_type, "rb_per_team")),
                   int(get("rules.no_qb_team_max")), int(for_type(slate_type, "qb_own_rb")),
                   float(get("rules.rb_min_tgt")), int(get("rules.max_rb_runback")),
                   int(get("rules.max_cheap_per_lineup")), float(get("rules.cheap_salary")))


def violations(lu, P: Pool, rules: Rules | None = None) -> list[str]:
    """Every DK rule and lineup rule this lineup breaks (empty = legal)."""
    lu = np.asarray(lu)
    out = []
    if len(set(lu.tolist())) != ROSTER:
        return ["duplicate player"]
    if P.sal[lu].sum() > CAP:
        out.append("over the cap")
    pos = list(P.pos[lu])
    c = {p: pos.count(p) for p in ("QB", "RB", "WR", "TE", "DST")}
    if c["QB"] != 1 or c["DST"] != 1 or not (2 <= c["RB"] <= 3 and 3 <= c["WR"] <= 4 and 1 <= c["TE"] <= 2) \
            or c["RB"] + c["WR"] + c["TE"] != 7:
        return out + ["positions"]
    if len(set(P.game[lu])) < 2:
        out.append("one game")
    if rules is None:
        return out
    team, opp = P.team[lu], P.opp[lu]
    q = lu[P.pos[lu] == "QB"][0]
    d = lu[P.pos[lu] == "DST"][0]
    mates = sum(1 for i in lu if P.team[i] == P.team[q] and P.pos[i] in ("WR", "TE"))
    if P.sal[lu].sum() < rules.min_salary:
        out.append("salary floor")
    if mates < rules.stack_min or mates > rules.stack_max:
        out.append("QB stack size")
    if rules.hard and HARD_NO_DST_VS_QB and P.opp[q] == P.team[d]:
        out.append("DST vs own QB")
    if sum(1 for i in lu if P.opp[i] == P.team[d] and P.pos[i] in ("RB", "WR", "TE")) > rules.max_vs_own_dst:
        out.append("too many vs own DST")
    if np.unique(team, return_counts=True)[1].max() > rules.max_per_team:
        out.append("max per team")
    if np.unique(P.game[lu], return_counts=True)[1].max() > rules.game_max:
        out.append("max per game")
    rbs = [i for i in lu if P.pos[i] == "RB"]
    if rbs and np.unique(P.team[rbs], return_counts=True)[1].max() > rules.rb_per_team:
        out.append("RBs per team")
    for t in set(team) - {P.team[q]}:
        if sum(1 for i in lu if P.team[i] == t and P.pos[i] in ("RB", "WR", "TE")) > rules.no_qb_team_max:
            out.append("skill players without their QB")
            break
    if sum(1 for i in lu if P.pos[i] in ("QB", "RB", "WR") and P.sal[i] <= rules.cheap_salary) > rules.max_cheap:
        out.append("too many cheap QB/RB/WR")
    if sum(1 for i in rbs if P.team[i] == P.opp[q]) > rules.max_rb_runback:
        out.append("RB run-back")
    own_rb = [i for i in rbs if P.team[i] == P.team[q]]
    if own_rb and (rules.qb_own_rb == 0 or (rules.qb_own_rb == 1 and (
            P.tgt is None or any(P.tgt[i] < rules.rb_min_tgt for i in own_rb)))):
        out.append("QB with own RB")
    if rules.hard and P.own is not None and (P.own[lu] >= HARD_CHALK[0]).sum() > HARD_CHALK[1]:
        out.append("two 40%+ owned players")
    return out


def shapes(lus: np.ndarray, P: Pool) -> dict:
    """Per lineup: QB + exactly 2 pass-catchers (double), exactly 1 run-back (an opposing WR/TE -
    workbook M4 counts receivers, not RBs), TE in FLEX (two TEs), RB in FLEX (three RBs) - the
    workbook's portfolio shares (M1, M4, M11)."""
    out = {"double": [], "bringback": [], "te_flex": [], "rb_flex": []}
    for lu in np.atleast_2d(lus):
        q = lu[P.pos[lu] == "QB"][0]
        mates = sum(1 for i in lu if P.team[i] == P.team[q] and P.pos[i] in ("WR", "TE"))
        bb = sum(1 for i in lu if P.team[i] == P.opp[q] and P.pos[i] in ("WR", "TE"))
        out["double"].append(mates == 2)
        out["bringback"].append(bb == 1)
        out["te_flex"].append(int((P.pos[lu] == "TE").sum()) == 2)
        out["rb_flex"].append(int((P.pos[lu] == "RB").sum()) == 3)
    return {k: np.array(v, bool) for k, v in out.items()}


def valid(lu, P: Pool, rules: Rules | None = None) -> bool:
    return not violations(lu, P, rules)


def assign_slots(lu, P: Pool) -> list[int]:
    """Order a lineup into DK's slots. FLEX gets the latest-kickoff player among those who can go
    there (ties: cheaper), so late swap has the most room."""
    lu = [int(x) for x in lu]
    by = {p: sorted([i for i in lu if P.pos[i] == p], key=lambda i: (P.kickoff[i], -P.sal[i])) for p in
          ("QB", "RB", "WR", "TE", "DST")}
    need = {"RB": 2, "WR": 3, "TE": 1}
    extra = [p for p in need if len(by[p]) > need[p]]
    flex_pos = extra[0] if extra else "WR"
    # the extra player at the over-filled position that kicks off latest goes to FLEX
    flex = max(by[flex_pos], key=lambda i: (P.kickoff[i], -P.sal[i]))
    by[flex_pos].remove(flex)
    return by["QB"] + by["RB"][:2] + by["WR"][:3] + by["TE"][:1] + [flex] + by["DST"]


class Optimizer:
    """Max-points Classic lineup under DK rules + stacking (MILP via HiGHS).
    Variables: x_i (player i in lineup), y_g (game g used)."""

    def __init__(self, P: Pool, rules: Rules, eligible: np.ndarray | None = None):
        from scipy.optimize import LinearConstraint
        self.P, self.rules = P, rules
        n = len(P.sal)
        games = sorted(set(P.game))
        G = len(games)
        self.n, self.G = n, G
        rows, lo, hi = [], [], []

        def add(r, l, h):
            rows.append(r), lo.append(l), hi.append(h)

        z = np.zeros(G)
        is_ = {p: (P.pos == p).astype(float) for p in ("QB", "RB", "WR", "TE", "DST")}
        add(np.r_[np.ones(n), z], ROSTER, ROSTER)
        add(np.r_[is_["QB"], z], 1, 1)
        add(np.r_[is_["DST"], z], 1, 1)
        add(np.r_[is_["RB"], z], 2, 3)
        add(np.r_[is_["WR"], z], 3, 4)
        add(np.r_[is_["TE"], z], 1, 2)
        add(np.r_[P.sal, z], rules.min_salary, CAP)
        for t in sorted(set(P.team)):
            add(np.r_[(P.team == t).astype(float), z], 0, rules.max_per_team)
        for k, g in enumerate(games):
            m = (P.game == g).astype(float)
            e = np.zeros(G)
            e[k] = -ROSTER
            add(np.r_[m, e], -np.inf, 0)          # players in g <= 9 * y_g
            e2 = np.zeros(G)
            e2[k] = -1
            add(np.r_[m, e2], 0, np.inf)          # y_g <= players in g
            add(np.r_[m, z], 0, rules.game_max)
        add(np.r_[np.zeros(n), np.ones(G)], 2, np.inf)   # 2+ games
        if rules.max_cheap < ROSTER:                    # Kenny 11/08: at most N cheap QB/RB/WR per lineup
            ch = (np.isin(P.pos, ["QB", "RB", "WR"]) & (P.sal <= rules.cheap_salary)).astype(float)
            add(np.r_[ch, z], 0, rules.max_cheap)
        pc = np.isin(P.pos, ["WR", "TE"])
        skill = np.isin(P.pos, ["RB", "WR", "TE"])
        M = 8
        for t in sorted(set(P.team)):
            qb = ((P.team == t) & (P.pos == "QB")).astype(float)
            on_t = (P.team == t)
            # M8: at most rb_per_team RBs from one team
            add(np.r_[(on_t & (P.pos == "RB")).astype(float), z], 0, rules.rb_per_team)
            if qb.any():
                mates = (on_t & pc).astype(float)
                if rules.stack_min > 0:
                    add(np.r_[mates - rules.stack_min * qb, z], 0, np.inf)      # M1 (hard): >= 1
                add(np.r_[mates + M * qb, z], -np.inf, rules.stack_max + M)     # M2 (hard): <= 2
                if rules.bringback >= 2:
                    opp = ((P.opp == t) & skill).astype(float)
                    add(np.r_[opp - qb, z], 0, np.inf)
                # M4: opposing RBs as run-backs are limited (receivers are the run-back)
                if rules.max_rb_runback < 3:
                    opp_rb = ((P.opp == t) & (P.pos == "RB")).astype(float)
                    add(np.r_[opp_rb + M * qb, z], -np.inf, rules.max_rb_runback + M)
                # M8: skill players from a team whose QB isn't used: <= no_qb_team_max
                add(np.r_[(on_t & skill).astype(float) - M * qb, z], -np.inf, rules.no_qb_team_max)
                # M7: QB with his own RB only if the back is a receiver (mode 1) / never (mode 0)
                if rules.qb_own_rb < 2:
                    tg = P.tgt if P.tgt is not None else np.zeros(n)
                    bad = on_t & (P.pos == "RB") & ((tg < rules.rb_min_tgt) if rules.qb_own_rb == 1 else True)
                    if bad.any():
                        add(np.r_[bad.astype(float) + M * qb, z], -np.inf, M)
            d = (on_t & (P.pos == "DST")).astype(float)
            if d.any():
                facing_skill = ((P.opp == t) & skill).astype(float)
                k = rules.max_vs_own_dst                                       # M6
                add(np.r_[facing_skill + (M - k) * d, z], -np.inf, M)
                if HARD_NO_DST_VS_QB and rules.hard:                            # M5 (hard)
                    facing_qb = ((P.opp == t) & (P.pos == "QB")).astype(float)
                    add(np.r_[facing_qb + d, z], -np.inf, 1)
        if P.own is not None and rules.hard:                                     # M14 (hard)
            chalk = (P.own >= HARD_CHALK[0]).astype(float)
            if chalk.sum() > HARD_CHALK[1]:
                add(np.r_[chalk, z], 0, HARD_CHALK[1])
        self.A, self.lo, self.hi = np.array(rows), np.array(lo, float), np.array(hi, float)
        self.base = [LinearConstraint(self.A, self.lo, self.hi)]
        self.ub = np.r_[np.ones(n) if eligible is None else (np.asarray(eligible) > 0).astype(float), np.ones(G)]
        self.games = games
        self._pc, self._skill = pc, skill

    def solve(self, pts: np.ndarray, banned=(), forced=(), qb_mates: int | None = None, rb_max: int | None = None,
              bringback: int | None = None, team: str | None = None, extra=(), time_limit: float = 10.0):
        """Best lineup for these points. qb_mates = exact pass-catchers with the QB (else the rules'
        range); bringback = exact opposing skill players (None = any); team = force this QB's team."""
        from scipy.optimize import Bounds, LinearConstraint, milp
        P, n, G = self.P, self.n, self.G
        ub = self.ub.copy()
        lb = np.zeros(n + G)
        ub[list(banned)] = 0
        lb[list(forced)] = 1
        if (lb > ub).any():
            return None
        cons = list(self.base) + list(extra)
        z = np.zeros(G)
        teams = [team] if team else sorted(set(P.team[(P.pos == "QB") & (ub[:n] > 0)]))
        if team:
            q = ((P.team == team) & (P.pos == "QB")).astype(float)
            cons.append(LinearConstraint(np.r_[q, z][None, :], 1, 1))
        rows, lo, hi = [], [], []
        for t in teams:
            qb = ((P.team == t) & (P.pos == "QB")).astype(float)
            if not qb.any():
                continue
            mates = ((P.team == t) & self._pc).astype(float)
            M = 8
            if qb_mates is not None:            # mates >= k*qb and mates <= k + M(1-qb)
                rows.append(np.r_[mates - qb_mates * qb, z]); lo.append(0); hi.append(np.inf)
                rows.append(np.r_[mates + M * qb, z]); lo.append(-np.inf); hi.append(qb_mates + M)
            if bringback is not None:
                opp = ((P.opp == t) & self._pc).astype(float)        # run-back = opposing WR/TE
                rows.append(np.r_[opp - bringback * qb, z]); lo.append(0); hi.append(np.inf)
                rows.append(np.r_[opp + M * qb, z]); lo.append(-np.inf); hi.append(bringback + M)
        if rb_max is not None:                                     # e.g. 2 = no RB in FLEX
            rows.append(np.r_[(P.pos == "RB").astype(float), z]); lo.append(0); hi.append(rb_max)
        if rows:
            cons.append(LinearConstraint(np.array(rows), lo, hi))
        c = -np.r_[np.asarray(pts, float), np.zeros(G)]
        res = milp(c, constraints=cons, integrality=np.ones(n + G), bounds=Bounds(lb, ub),
                   options={"time_limit": time_limit, "mip_rel_gap": 1e-4})
        if res.x is None:
            return None
        lu = np.flatnonzero(res.x[:n] > 0.5)
        return np.sort(lu) if len(lu) == ROSTER else None


# ------------------------------------------------------------------ simulated field

def field(n: int, own: np.ndarray, P: Pool, rng, stack: float = 0.7, double: float = 0.3,
          bringback: float = 0.3, min_salary: int = 48000, max_tries: int = 60) -> np.ndarray:
    """Opponent lineups drawn from projected ownership, built the way real players build: pick a
    QB, stack him (share `stack`) with 1 (or 2) of his pass-catchers and sometimes a bring-back,
    then fill each position by ownership. Lineups must fit the cap and mostly spend >= min_salary."""
    own = np.maximum(np.asarray(own, float), 0)
    pos, team, opp, sal = P.pos, P.team, P.opp, P.sal
    idx = {p: np.flatnonzero((pos == p) & (own > 0)) for p in ("QB", "RB", "WR", "TE", "DST")}
    w = {p: own[i] / own[i].sum() for p, i in idx.items() if len(i)}
    flex_idx = np.r_[idx["RB"], idx["WR"], idx["TE"]]
    # FLEX ownership = ownership beyond the 2 RB / 3 WR / 1 TE base slots (approx: proportional)
    flex_w = own[flex_idx] / own[flex_idx].sum()
    out = np.zeros((n, ROSTER), dtype=np.int64)
    k = 0
    tries = 0
    while k < n and tries < n * max_tries:
        tries += 1
        lu = []
        q = rng.choice(idx["QB"], p=w["QB"])
        lu.append(q)
        need = {"RB": 2, "WR": 3, "TE": 1}
        if rng.random() < stack:
            mates = [i for i in np.r_[idx["WR"], idx["TE"]] if team[i] == team[q]]
            if mates:
                m = np.array(mates)
                pm = own[m] / own[m].sum()
                kk = 2 if (rng.random() < double and len(m) >= 2) else 1
                for i in rng.choice(m, size=kk, replace=False, p=pm):
                    lu.append(i)
            if rng.random() < bringback:
                bb = np.array([i for i in flex_idx if team[i] == opp[q]])
                if len(bb):
                    lu.append(rng.choice(bb, p=own[bb] / own[bb].sum()))
        cnt = {p: sum(pos[i] == p for i in lu) for p in need}
        if any(cnt[p] > need[p] + (1 if p != "TE" or True else 0) for p in need):
            continue
        for p in ("RB", "WR", "TE"):
            free = need[p] - cnt[p]
            if free > 0:
                cand = idx[p][~np.isin(idx[p], lu)]
                pp = own[cand] / own[cand].sum()
                lu += list(rng.choice(cand, size=free, replace=False, p=pp))
        n_flex = 7 - sum(1 for i in lu if pos[i] in ("RB", "WR", "TE"))
        if n_flex < 0:
            continue
        if n_flex:
            cand = flex_idx[~np.isin(flex_idx, lu)]
            pp = flex_w[np.isin(flex_idx, cand)]
            lu += list(rng.choice(cand, size=n_flex, replace=False, p=pp / pp.sum()))
        d = rng.choice(idx["DST"], p=w["DST"])
        lu.append(d)
        lu = np.array(lu)
        s = sal[lu].sum()
        if s > CAP or len(set(lu.tolist())) != ROSTER or len(set(P.game[lu])) < 2:
            continue
        if s < min_salary and rng.random() < 0.85:
            continue
        out[k] = np.sort(lu)
        k += 1
    return out[:k]
