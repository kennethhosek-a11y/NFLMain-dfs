"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
Game-script Monte Carlo for one NFL game (shared by Showdown and Classic).

Every simulated game is played from the team level down:
  1. pace (shared by both teams) and each team's efficiency are drawn;
  2. drives -> touchdowns / field goals (calibrated so the average score equals the Vegas implied
     total), plus a small chance of a defensive / return touchdown;
  3. the score margin sets the game script: the trailing team throws more, the leading team runs
     more (blowouts and comebacks come out of this naturally);
  4. team dropbacks, sacks, attempts, carries and yards follow from pace, script and efficiency;
  5. players split their team's targets, carries, yards and touchdowns (shares get random noise
     each game; a questionable player who sits gives his share to teammates).
So a QB and his receivers rise and fall together (same pass yards / TDs), a RB gains in a
positive script, a DST scores off the opposing QB's sacks / interceptions / points, and a kicker
benefits from drives that stall in range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import scoring
from .project import LG, Team
from .settings import get as S

EFF_SD = 0.30          # team efficiency (log scale on TD rate)
PACE_SD = 0.07
DRIVES = 11.0
P_FG = 0.20            # field-goal try share of non-TD drives
P_XP = 0.94
P_DEF_TD = 0.045       # defensive / special-teams TD per team-game
SHARE_CONC = 20.0      # (tuned to real history 11/05/2026; was 35) concentration of the per-game share noise (higher = steadier roles)
REC_NOISE = 0.90       # (tuned to real history 11/05/2026; was 0.45) per-player yards-per-catch noise (log scale)
RUSH_NOISE = 0.40      # per-player yards-per-carry noise
TD_CONC = 60.0         # how evenly a team's TDs spread over players (lower = more multi-TD games)
TEAM_PASS_NOISE = 0.15 # team passing-yards noise beyond efficiency
TEAM_RUSH_NOISE = 0.20
EFF_CORR = 0.0         # opposing teams' efficiency: >0 = one side's good night is the other's bad night
SCRIPT_PASS = 0.010    # log change in dropbacks per point of margin (trailing -> more)
SCRIPT_RUN = 0.012
FG_DIST = [0.50, 0.30, 0.20]   # 0-39, 40-49, 50+


def _load_settings() -> None:
    """Module constants above are the defaults; the live values come from config/settings.yaml
    (gamesim.*) plus Kenny's layers, read at the start of every simulation."""
    global EFF_SD, PACE_SD, DRIVES, P_FG, P_DEF_TD, SHARE_CONC, REC_NOISE, RUSH_NOISE, TD_CONC, \
        TEAM_PASS_NOISE, TEAM_RUSH_NOISE, EFF_CORR
    EFF_SD = float(S("gamesim.eff_sd"))
    PACE_SD = float(S("gamesim.pace_sd"))
    DRIVES = float(S("gamesim.drives"))
    P_FG = float(S("gamesim.p_fg"))
    P_DEF_TD = float(S("gamesim.p_def_td"))
    SHARE_CONC = float(S("gamesim.share_conc"))
    REC_NOISE = float(S("gamesim.rec_noise"))
    RUSH_NOISE = float(S("gamesim.rush_noise"))
    TD_CONC = float(S("gamesim.td_conc"))
    TEAM_PASS_NOISE = float(S("gamesim.team_pass_noise"))
    TEAM_RUSH_NOISE = float(S("gamesim.team_rush_noise"))
    EFF_CORR = float(S("gamesim.eff_corr"))


def _p_fg(tm: Team) -> float:
    """Field-goal tries per non-TD drive: needs drives that reach range, so weaker offenses
    (low implied totals) get fewer; wind/rain lower it further."""
    return P_FG * tm.kick_mult * float(np.clip(tm.implied / LG["pts"], 0.5, 1.4)) ** 0.5


def _lognorm(rng, sd, shape):
    return np.exp(sd * rng.standard_normal(shape) - sd * sd / 2)


def _alloc(rng, total: np.ndarray, w: np.ndarray, conc: float) -> np.ndarray:
    """Split integer totals (sims,) by weights (sims, k) with Dirichlet noise -> (sims, k)."""
    w = np.maximum(w, 1e-9)
    g = rng.gamma(np.maximum(conc * w / w.sum(axis=1, keepdims=True), 1e-3))
    p = g / g.sum(axis=1, keepdims=True)
    return rng.multinomial(total.astype(np.int64), p)


def simulate(teams: dict[str, Team], players: pd.DataFrame, n_sims: int, seed: int = 0,
             calibrate: bool = True) -> dict:
    """players: rows with team, pos, p_play, qb_start, tgt_share, car_share, rz_tgt_share,
    rz_car_share, catch, ypr, ypc. Returns {"fp": (n, sims) DK points, "score": {team: (sims,)},
    "stats": {name: (n, sims)}}."""
    _load_settings()
    names = list(teams)
    p_td = {}
    for t in names:   # starting point: solve the drive model for the implied total
        tm = teams[t]
        per = tm.implied / DRIVES - 6 * P_DEF_TD / DRIVES
        pf = _p_fg(tm)
        p_td[t] = float(np.clip((per - 3 * pf) / (6 + P_XP - 3 * pf), 0.03, 0.6))
    if calibrate:
        for _ in range(3):
            res = _run(teams, players, min(n_sims, 6000), seed + 977, p_td, stats=False)
            for t in names:
                m = res["score"][t].mean()
                p_td[t] = float(np.clip(p_td[t] * (teams[t].implied / max(m, 1)) ** 1.15, 0.02, 0.7))
    return _run(teams, players, n_sims, seed, p_td, stats=True)


def _run(teams, players, n_sims, seed, p_td, stats=True):
    rng = np.random.default_rng(seed)
    names = list(teams)
    S = n_sims
    pace = _lognorm(rng, PACE_SD, S)
    eff = {t: rng.standard_normal(S) * EFF_SD for t in names}
    if EFF_CORR and len(names) == 2:     # one side's big night tends to be the other's bad night
        a, b = names
        eff[b] = -EFF_CORR * eff[a] + np.sqrt(1 - EFF_CORR ** 2) * eff[b]
    td, fg, xp, dtd, score = {}, {}, {}, {}, {}
    for t in names:
        tm = teams[t]
        drives = np.maximum(np.rint(DRIVES * pace + rng.normal(0, 0.7, S)), 6).astype(np.int64)
        p = np.clip(p_td[t] * np.exp(eff[t] - EFF_SD ** 2 / 2), 0.0, 0.75)
        td[t] = rng.binomial(drives, p)
        fg[t] = rng.binomial(drives - td[t], np.clip(_p_fg(tm) * np.exp(-0.25 * eff[t]), 0, 0.6))
        dtd[t] = (rng.random(S) < P_DEF_TD).astype(np.int64)
        xp[t] = rng.binomial(td[t] + dtd[t], P_XP)
        score[t] = 6 * (td[t] + dtd[t]) + xp[t] + 3 * fg[t]
    out = {"score": score, "td": td, "fg": fg}
    if not stats:
        return out

    df = players.reset_index(drop=True)
    n = len(df)
    fp = np.zeros((n, S), dtype=np.float32)
    st = {k: np.zeros((n, S), dtype=np.float32) for k in
          ("pass_yd", "pass_td", "int", "rush_yd", "rush_td", "rec", "rec_yd", "rec_td", "tgt", "car")}
    pos = df["pos"].values
    active = rng.random((n, S)) < df["p_play"].values[:, None]
    for t in names:
        tm, o = teams[t], names[1 - names.index(t)]
        margin = np.clip(score[t] - score[o], -24, 24)
        plays = tm.plays * pace * np.exp(0.15 * eff[t])
        db = rng.poisson(plays * tm.pass_rate * np.exp(-SCRIPT_PASS * margin))
        car = rng.poisson(plays * (1 - tm.pass_rate) * np.exp(SCRIPT_RUN * margin))
        sack_rate = np.clip(tm.sack_rate * teams[o].def_sack_rate / LG["sack_rate"], 0.02, 0.14)
        sacks = rng.binomial(db, sack_rate)
        att = db - sacks
        lp = np.log(tm.ptd_frac / (1 - tm.ptd_frac)) - 0.04 * margin
        ptd = rng.binomial(td[t], 1 / (1 + np.exp(-lp)))
        rtd = td[t] - ptd
        yeff = np.exp(0.35 * eff[t])
        pass_yd = att * tm.ypa * yeff / np.exp(0.35 ** 2 * EFF_SD ** 2 / 2) * _lognorm(rng, TEAM_PASS_NOISE, S)
        rush_yd = car * tm.ypc * np.exp(0.2 * eff[t]) / np.exp(0.2 ** 2 * EFF_SD ** 2 / 2) * _lognorm(rng, TEAM_RUSH_NOISE, S)
        ints = rng.binomial(att, np.clip(tm.int_rate * np.exp(-0.6 * eff[t]), 0, 0.2))
        fum = rng.poisson(tm.fum_lost * np.ones(S))

        idx = np.flatnonzero((df["team"].values == t))
        skill = idx[np.isin(pos[idx], ["QB", "RB", "WR", "TE"])]
        act = active[skill].T                                   # (S, k)
        def w(col, other):
            v = df[col].values[skill][None, :] * act
            return np.c_[v, np.full((S, 1), other)]
        other_t = max(1 - df["tgt_share"].values[skill].sum(), 0.03)
        other_c = max(1 - df["car_share"].values[skill].sum(), 0.02)
        tg = _alloc(rng, att, w("tgt_share", other_t), SHARE_CONC)[:, :-1]
        cr = _alloc(rng, car, w("car_share", other_c), SHARE_CONC)[:, :-1]
        catch = df["catch"].values[skill][None, :]
        rec = rng.binomial(tg, np.clip(catch, 0, 1))
        # receiving yards: per-catch draws, then scaled so the team adds up to the QB's yards
        raw = rec * df["ypr"].values[skill][None, :] * _lognorm(rng, REC_NOISE, (S, len(skill)))
        other_raw = np.maximum(att * other_t * 0.65 * 9.0, 0.0)
        tot = raw.sum(axis=1) + other_raw
        ryd = raw * (pass_yd / np.maximum(tot, 1e-6))[:, None]
        rawr = cr * df["ypc"].values[skill][None, :] * _lognorm(rng, RUSH_NOISE, (S, len(skill)))
        other_r = car * other_c * tm.ypc
        rush = rawr * (rush_yd / np.maximum(rawr.sum(axis=1) + other_r, 1e-6))[:, None]
        tdw_r = 0.55 * df["rz_tgt_share"].values[skill] + 0.45 * df["tgt_share"].values[skill]
        tdw_c = 0.65 * df["rz_car_share"].values[skill] + 0.35 * df["car_share"].values[skill]
        rec_td = _alloc(rng, ptd, np.c_[tdw_r[None, :] * act, np.full((S, 1), max(1 - tdw_r.sum(), 0.04))], TD_CONC)[:, :-1]
        rush_td = _alloc(rng, rtd, np.c_[tdw_c[None, :] * act, np.full((S, 1), max(1 - tdw_c.sum(), 0.03))], TD_CONC)[:, :-1]
        rec = np.maximum(rec, rec_td)
        touches = tg + cr
        fl = _alloc(rng, fum, np.c_[touches + 0.01, np.full((S, 1), 3.0)], 1e4)[:, :-1]

        qb = skill[(df["qb_start"].values[skill]) & (pos[skill] == "QB")]
        pyd = np.zeros((S, len(skill)))
        ptd_q = np.zeros((S, len(skill)))
        ints_q = np.zeros((S, len(skill)))
        if len(qb):
            j = list(skill).index(qb[0])
            pyd[:, j], ptd_q[:, j], ints_q[:, j] = pass_yd, ptd, ints
        pts = scoring.offense(pass_yd=pyd, pass_td=ptd_q, ints=ints_q, rush_yd=rush, rush_td=rush_td,
                              rec=rec, rec_yd=ryd, rec_td=rec_td, fum_lost=fl)
        fp[skill] = (pts * act).T
        for k, v in (("pass_yd", pyd), ("pass_td", ptd_q), ("int", ints_q), ("rush_yd", rush), ("rush_td", rush_td),
                     ("rec", rec), ("rec_yd", ryd), ("rec_td", rec_td), ("tgt", tg), ("car", cr)):
            st[k][skill] = (v * act).T

        # kicker (the active one with the highest salary)
        ks = idx[(pos[idx] == "K") & (df["p_play"].values[idx] > 0)]
        if len(ks):
            k = ks[np.argmax(df["Salary"].values[ks])]
            d = _fg_split(rng, fg[t])
            fp[k] = scoring.kicker(d[:, 0], d[:, 1], d[:, 2], xp[t])
        # defense (vs the other team's offense, computed when we get to that team below)
        out.setdefault("_def", {})[o] = dict(sacks=sacks, ints=ints, fum=fum, pa=score[t])
    for t in names:
        dsts = np.flatnonzero((df["team"].values == t) & (pos == "DST"))
        if len(dsts):
            d = out["_def"][t]
            fp[dsts[0]] = scoring.dst(sacks=d["sacks"], ints=d["ints"], fum_rec=d["fum"], td=dtd[t], pa=d["pa"])
    out.pop("_def", None)
    fp[~active] = 0.0
    out["fp"] = fp
    out["stats"] = st
    return out


def _fg_split(rng, n):
    d = np.zeros((len(n), 3), dtype=np.int64)
    for k in range(int(n.max()) if len(n) else 0):
        has = n > k
        r = rng.random(len(n))
        b = np.where(r < FG_DIST[0], 0, np.where(r < FG_DIST[0] + FG_DIST[1], 1, 2))
        d[np.arange(len(n))[has], b[has]] += 1
    return d
