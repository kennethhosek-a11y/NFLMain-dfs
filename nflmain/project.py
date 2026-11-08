"""(Copied from the NFL Showdown repo, nfl-dfs, 11/07/2026; adapted for Classic.)
Usage model -> per-player inputs for the game simulator (shared by Showdown and Classic).

Projections are the MEAN of the game simulation (gamesim), so projections, floors, ceilings and
lineup sims all come from one model. This module builds the inputs:

Team level (per game): implied team total from the ESPN spread/total, plays, pass rate (shifted
by the spread: underdogs throw more), yards per attempt / carry, TD rate, sacks, turnovers;
weather trims passing and kicking.

Player level (nflverse, current season, recent games weighted more): target share, carry share,
red-zone target / carry share, air-yards share, catch rate, yards per catch / carry. Shares are
shrunk to position defaults when a player has few games. When a player is OUT, most of his
usage moves to active teammates (same position first).

Sportsbook props: OFF. `PROPS_ENABLED = False`; `apply_props()` is the hook where prop-implied
means would be blended in (scaling a player's per-touch rates) once a props source is added.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .names import norm
from .settings import get as S

PROPS_ENABLED = False      # owner decision 11/05/2026: no Odds API for now
PROPS_WEIGHT = 0.5         # weight given to props when enabled

# league averages (2026 weeks 1-8, nflverse team-game sums)
LG = dict(pts=22.3, att=33.1, car=26.9, ypa=7.06, ypc=4.20, pass_rate=0.569, ptd_frac=0.615,
          int_rate=0.021, sack_rate=0.065, catch=0.673, fum_lost=0.56, fg=1.74, plays=62.3)
HALF_LIFE = 4.0            # games (default; live value: settings model.half_life)
POS_DEFAULT = {            # share defaults for players with no history (per active game)
    "WR": dict(tgt=0.07, car=0.0, ypr=12.0, catch=0.62),
    "TE": dict(tgt=0.06, car=0.0, ypr=10.0, catch=0.70),
    "RB": dict(tgt=0.04, car=0.12, ypr=7.5, catch=0.78),
    "QB": dict(tgt=0.0, car=0.08, ypr=0.0, catch=0.0),
}
P_SCRATCH = {"Q": 0.0, "D": 1.0, "GTD": 0.0}   # built fully in or out; live values: p_scratch()


def p_scratch() -> dict:
    """Chance a tagged player sits, by status (config/settings.yaml model.p_scratch_*)."""
    q, d = float(S("model.p_scratch_q")), float(S("model.p_scratch_d"))
    return {"Q": q, "D": d, "GTD": q}


# Classic changes vs the Showdown copy (11/07/2026): the pre-game pass-rate shift by spread is
# smaller (the simulator's in-game script already makes trailing teams throw more - the old 0.007
# double-counted it), and implied totals scale yards per play.
SPREAD_PASS = 0.004
YPA_ELAST = 0.45
YPC_ELAST = 0.30

BACKUP_QB = dict(ypa=6.0, int_rate=0.030, sack_rate=0.085, td_mult=0.85)


@dataclass
class Team:
    team: str
    opp: str
    home: bool
    implied: float
    spread: float              # this team's spread (negative = favorite)
    plays: float
    pass_rate: float
    ypa: float
    ypc: float
    ptd_frac: float
    int_rate: float
    sack_rate: float           # sacks per dropback allowed by this offense
    def_sack_rate: float       # sacks per opponent dropback by this defense
    fum_lost: float
    kick_mult: float = 1.0     # weather on field goals
    notes: list = field(default_factory=list)


def implied_totals(total: float, spread_home: float) -> tuple[float, float]:
    """(home, away) implied points. spread_home < 0 means the home team is favored."""
    home = total / 2 - spread_home / 2
    return home, total - home


def _shrink(x, n, prior, k):
    x = np.where(np.isfinite(x), x, prior)
    return (n * x + k * prior) / (n + k)


# ------------------------------------------------------------------ history

def team_games(stats: pd.DataFrame) -> pd.DataFrame:
    g = stats.groupby(["game_id", "week", "team"]).agg(
        att=("attempts", "sum"), car=("carries", "sum"), tgt=("targets", "sum"),
        py=("passing_yards", "sum"), ry=("rushing_yards", "sum"), ptd=("passing_tds", "sum"),
        rtd=("rushing_tds", "sum"), sk=("sacks_suffered", "sum"), ints=("passing_interceptions", "sum"),
        fl=("fumbles_lost_total", "sum"), dsk=("def_sacks", "sum"), ay=("receiving_air_yards", "sum"),
        opp=("opponent_team", "first")).reset_index()
    return g


def team_history(stats: pd.DataFrame) -> pd.DataFrame:
    g = team_games(stats)
    opp_db = g[["game_id", "team", "att", "sk"]].rename(columns={"team": "opp", "att": "o_att", "sk": "o_sk"})
    g = g.merge(opp_db, on=["game_id", "opp"], how="left")
    agg = g.groupby("team").agg(n=("game_id", "size"), att=("att", "sum"), car=("car", "sum"),
                                py=("py", "sum"), ry=("ry", "sum"), ptd=("ptd", "sum"), rtd=("rtd", "sum"),
                                sk=("sk", "sum"), ints=("ints", "sum"), fl=("fl", "sum"),
                                o_att=("o_att", "sum"), o_sk=("o_sk", "sum"))
    out = pd.DataFrame(index=agg.index)
    n = agg["n"]
    out["n"] = n
    out["plays"] = _shrink((agg["att"] + agg["sk"] + agg["car"]) / n, n, LG["plays"], 4)
    out["pass_rate"] = _shrink((agg["att"] + agg["sk"]) / (agg["att"] + agg["sk"] + agg["car"]), n, LG["pass_rate"], 4)
    out["ypa"] = _shrink(agg["py"] / agg["att"], n, LG["ypa"], 4)
    out["ypc"] = _shrink(agg["ry"] / agg["car"], n, LG["ypc"], 6)
    out["ptd_frac"] = _shrink(agg["ptd"] / (agg["ptd"] + agg["rtd"]), n, LG["ptd_frac"], 8)
    out["int_rate"] = _shrink(agg["ints"] / agg["att"], n, LG["int_rate"], 8)
    out["sack_rate"] = _shrink(agg["sk"] / (agg["att"] + agg["sk"]), n, LG["sack_rate"], 6)
    out["def_sack_rate"] = _shrink(agg["o_sk"] / (agg["o_att"] + agg["o_sk"]), n, LG["sack_rate"], 6)
    out["fum_lost"] = _shrink(agg["fl"] / n, n, LG["fum_lost"], 6)
    return out


def player_usage(stats: pd.DataFrame, snaps: pd.DataFrame | None, rz: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (name_key, team): recency-weighted shares and per-touch efficiency."""
    tg = team_games(stats)[["game_id", "team", "att", "car", "tgt", "ay"]].rename(
        columns={"att": "t_att", "car": "t_car", "tgt": "t_tgt", "ay": "t_ay"})
    p = stats[["player_id", "name_key", "player_display_name", "position", "team", "game_id", "week",
               "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
               "sacks_suffered", "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
               "receiving_yards", "receiving_tds", "receiving_air_yards", "fumbles_lost_total"]].copy()
    p = p.merge(tg, on=["game_id", "team"], how="left")
    if rz is not None and len(rz):
        rz = rz.copy()
        rzt = rz.groupby(["game_id", "team"])[["team_rz_pass", "team_rz_run"]].first().reset_index()
        p = p.merge(rz[["game_id", "player_id", "rz_tgt", "rz_car"]], on=["game_id", "player_id"], how="left")
        p = p.merge(rzt, on=["game_id", "team"], how="left")
    for c in ("rz_tgt", "rz_car", "team_rz_pass", "team_rz_run"):
        p[c] = p.get(c, 0)
        p[c] = p[c].fillna(0)
    # games played with zero touches (in snap counts but not the stats file) count as 0 usage
    if snaps is not None and len(snaps):
        sn = snaps[snaps["offense_snaps"] > 0][["name_key", "team", "week", "offense_pct"]]
        p = p.merge(sn, on=["name_key", "team", "week"], how="left")
        played = sn.merge(p[["name_key", "team", "week"]].drop_duplicates(), how="left", indicator=True,
                          on=["name_key", "team", "week"])
        extra = played[played["_merge"] == "left_only"].drop(columns="_merge")
        if len(extra):
            extra = extra.merge(tg.merge(p[["game_id", "team", "week"]].drop_duplicates(), on=["game_id", "team"]),
                                on=["team", "week"], how="inner")
            p = pd.concat([p, extra], ignore_index=True)
    else:
        p["offense_pct"] = np.nan
    num = p.select_dtypes("number").columns
    p[num] = p[num].fillna(0)
    p["position"] = p["position"].fillna("")
    last_week = p.groupby("team")["week"].transform("max")
    p["w"] = 0.5 ** ((last_week - p["week"]) / float(S("model.half_life")))

    def wsum(c):
        return (p[c] * p["w"]).groupby([p["name_key"], p["team"]]).sum()

    keys = ["name_key", "team"]
    W = p.groupby(keys)["w"].sum()
    out = pd.DataFrame({
        "games": p.groupby(keys)["game_id"].nunique(),
        "last_week": p.groupby(keys)["week"].max(),
        "nv_pos": p.groupby(keys)["position"].agg(lambda s: s[s != ""].iloc[-1] if (s != "").any() else ""),
        "snap_pct": wsum("offense_pct") / W,
        "tgt_share": wsum("targets") / wsum("t_tgt").clip(lower=1),
        "car_share": wsum("carries") / wsum("t_car").clip(lower=1),
        "att_share": wsum("attempts") / wsum("t_att").clip(lower=1),
        "ay_share": wsum("receiving_air_yards") / wsum("t_ay").clip(lower=1),
        "rz_tgt_share": wsum("rz_tgt") / wsum("team_rz_pass").clip(lower=1),
        "rz_car_share": wsum("rz_car") / wsum("team_rz_run").clip(lower=1),
        "tgt": p.groupby(keys)["targets"].sum(), "rec": p.groupby(keys)["receptions"].sum(),
        "rec_yd": p.groupby(keys)["receiving_yards"].sum(), "car": p.groupby(keys)["carries"].sum(),
        "rush_yd": p.groupby(keys)["rushing_yards"].sum(), "att": p.groupby(keys)["attempts"].sum(),
        "pass_yd": p.groupby(keys)["passing_yards"].sum(), "pass_td": p.groupby(keys)["passing_tds"].sum(),
        "ints": p.groupby(keys)["passing_interceptions"].sum(), "sk": p.groupby(keys)["sacks_suffered"].sum(),
        "fum": p.groupby(keys)["fumbles_lost_total"].sum(),
    }).reset_index()
    return out


# ------------------------------------------------------------------ build inputs

def build(pool: pd.DataFrame, game: dict, stats: pd.DataFrame, snaps=None, rz=None,
          wx: dict | None = None, th: pd.DataFrame | None = None,
          use: pd.DataFrame | None = None) -> tuple[dict[str, Team], pd.DataFrame]:
    """pool: one row per DK player (FLEX rows) with team, Position, Salary, status, name_key.
    game: {"home","away","total","spread_home"}. Returns (teams, players with sim inputs)."""
    # Classic: the slate computes team history and player usage once and passes them in
    th = team_history(stats) if th is None else th
    use = player_usage(stats, snaps, rz) if use is None else use
    home, away = game["home"], game["away"]
    total = float(game.get("total") or 2 * LG["pts"])
    sh = float(game.get("spread_home") or 0.0)
    imp_h, imp_a = implied_totals(total, sh)
    wx = wx or {}
    wind = float(wx.get("wind") or 0)
    precip = float(wx.get("precip") or 0)
    teams = {}
    for t, o, imp, spr, is_home in ((home, away, imp_h, sh, True), (away, home, imp_a, -sh, False)):
        h = th.loc[t] if t in th.index else None
        g = (lambda c: float(h[c]) if h is not None else LG.get(c, np.nan))
        tm = Team(t, o, is_home, imp, spr,
                  plays=g("plays") * (1 + 0.004 * (imp - LG["pts"])),
                  pass_rate=float(np.clip(g("pass_rate") + SPREAD_PASS * spr, 0.40, 0.72)),
                  ypa=g("ypa"), ypc=g("ypc"), ptd_frac=g("ptd_frac"), int_rate=g("int_rate"),
                  sack_rate=g("sack_rate"),
                  def_sack_rate=float(th.loc[t, "def_sack_rate"]) if t in th.index else LG["sack_rate"],
                  fum_lost=g("fum_lost"))
        if wind >= 15:
            k = min((wind - 10) / 10, 1.5)
            tm.pass_rate -= 0.03 * k
            tm.ypa *= 1 - 0.05 * k
            tm.kick_mult = 1 - 0.12 * k
            tm.notes.append(f"wind {wind:.0f} mph: passing and kicking trimmed")
        if precip >= 50:
            tm.ypa *= 0.97
            tm.kick_mult *= 0.97
            tm.notes.append(f"rain chance {precip:.0f}%")
        teams[t] = tm

    pl = pool.copy()
    pl["pos"] = pl["Position"].astype(str)
    pl = pl.merge(use, on=["name_key", "team"], how="left")
    pl["games"] = pl["games"].fillna(0)
    pl["p_play"] = 1.0 - pl["status"].map(p_scratch()).fillna(0.0)
    pl.loc[pl["status"].isin(["OUT", "IR", "O"]), "p_play"] = 0.0
    for c in ("tgt_share", "car_share", "rz_tgt_share", "rz_car_share", "ay_share", "att_share", "snap_pct"):
        pl[c] = pl[c].fillna(0.0)

    # --- starting QB per team: highest recent pass share among QBs who will play; else top salary
    pl["qb_start"] = False
    for t in (home, away):
        q = pl[(pl["team"] == t) & (pl["pos"] == "QB") & (pl["p_play"] > 0)]
        if not len(q):
            continue
        q = q.assign(_k=q["att_share"] * (q["last_week"].fillna(0) >= q["last_week"].max() - 2) + q["Salary"] / 1e6)
        if "qb_override" in q and q["qb_override"].any():
            q = q[q["qb_override"]]
        pl.loc[q["_k"].idxmax(), "qb_start"] = True
    pl.loc[(pl["pos"] == "QB") & ~pl["qb_start"], "p_play"] = 0.0      # backups: no snaps expected

    # --- efficiency, shrunk to position defaults
    n_t = pl["tgt"].fillna(0)
    d = pl["pos"].map(lambda p: POS_DEFAULT.get(p, POS_DEFAULT["WR"]))
    pl["catch"] = _shrink(pl["rec"] / n_t.replace(0, np.nan), n_t, d.map(lambda x: x["catch"]), 25)
    n_r = pl["rec"].fillna(0)
    pl["ypr"] = _shrink(pl["rec_yd"] / n_r.replace(0, np.nan), n_r, d.map(lambda x: x["ypr"]), 15)
    n_c = pl["car"].fillna(0)
    pl["ypc"] = _shrink(pl["rush_yd"] / n_c.replace(0, np.nan), n_c, LG["ypc"], 40)
    # shares: shrink players with few games toward a salary-based default
    for c, key in (("tgt_share", "tgt"), ("car_share", "car")):
        prior = d.map(lambda x: x[key]) * np.clip(pl["Salary"] / 6000, 0.2, 1.5)
        pl[c] = _shrink(pl[c], pl["games"], prior, 1.0)
    pl.loc[pl["pos"].isin(["K", "DST"]), ["tgt_share", "car_share", "rz_tgt_share", "rz_car_share"]] = 0.0

    # --- QB passing efficiency (starter): his own numbers if he has them, else backup defaults
    for t, tm in teams.items():
        q = pl[(pl["team"] == t) & pl["qb_start"]]
        if not len(q):
            continue
        r = q.iloc[0]
        att = float(r.get("att") or 0)
        if att >= 40:
            tm.ypa = float(_shrink(r["pass_yd"] / att, att, tm.ypa, 120))
            tm.int_rate = float(_shrink(r["ints"] / att, att, LG["int_rate"], 200))
        else:
            tm.ypa = min(tm.ypa, BACKUP_QB["ypa"]) * (1 - (0.05 if wind >= 15 else 0))
            tm.int_rate = BACKUP_QB["int_rate"]
            tm.sack_rate = max(tm.sack_rate, BACKUP_QB["sack_rate"])
            tm.notes.append(f"{r['Name']} has little NFL history: backup-QB passing rates")
        # QB rushing share (scrambles + designed runs)
        qcar = float(r["car_share"]) if r["games"] >= 2 else 0.08
        pl.loc[q.index, "car_share"] = qcar
        pl.loc[q.index, "rz_car_share"] = max(float(r["rz_car_share"]), qcar * 0.8)
        pl.loc[q.index, "tgt_share"] = 0.0

    # Classic change (11/07/2026): the implied total also moves yards per play, not only TDs - a
    # 28-point team gains more yards than a 22-point team (league elasticity of yards to points ~0.5)
    for t, tm in teams.items():
        r = float(np.clip(tm.implied / LG["pts"], 0.6, 1.5))
        tm.ypa *= r ** YPA_ELAST
        tm.ypc *= r ** YPC_ELAST
    pl = redistribute(pl)
    pl = apply_props(pl)
    return teams, pl


SAME_POS_SHARE = 0.85    # share of an OUT player's usage that goes to his own position group
BACKUP_FLOOR = 0.02      # so a backup with almost no history still picks up some of the work


def redistribute(pl: pd.DataFrame) -> pd.DataFrame:
    """Usage of players who won't play (p_play == 0) moves to active teammates: 85% to his own
    position (RB carries to the other RBs, not the QB), in proportion to their shares plus a small
    floor so backups get some, and the rest to every active player in proportion to his share.
    Then usage of recent contributors not on the slate (traded, released) mostly moves to the
    active players, as before. Q/D players are built fully in or out (model.p_scratch_*)."""
    pl = pl.copy()
    for t in pl["team"].unique():
        m = (pl["team"] == t).to_numpy()
        pos = pl.loc[m, "pos"].to_numpy(dtype=object)
        playing = pl.loc[m, "p_play"].to_numpy() > 0
        # only real roles are handed off: players with recent games, never a QB (the starting QB's
        # runs belong to whoever starts; backups' default shares aren't work anyone loses)
        real = (pl.loc[m, "games"].fillna(0).to_numpy() > 0) & (pos != "QB")
        for c, base_other in (("tgt_share", 0.05), ("car_share", 0.03),
                              ("rz_tgt_share", 0.05), ("rz_car_share", 0.03)):
            s = pl.loc[m, c].astype(float).to_numpy().copy()
            new = np.where(playing, s, 0.0)
            for q in set(pos[~playing & real & (s > 0)]):
                vac = s[~playing & real & (pos == q)].sum()
                same = playing & (pos == q)
                if same.any():
                    w = s[same] + BACKUP_FLOOR
                    new[same] += SAME_POS_SHARE * vac * w / w.sum()
                    rest = (1 - SAME_POS_SHARE) * vac
                else:
                    rest = vac
                if new[playing].sum() > 0:
                    new[playing] += rest * new[playing] / new[playing].sum()
            S = new.sum()
            if S <= 0:
                continue
            target = min(0.97 - base_other, S + 0.75 * max(1 - base_other - S, 0))
            pl.loc[m, c] = new * (target / S) if S > target else new + (target - S) * new / S
    return pl


def apply_props(pl: pd.DataFrame, props: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hook for sportsbook props (OFF). When enabled, a player's prop-implied mean for yards /
    receptions would be blended with the model at PROPS_WEIGHT by scaling his per-touch rates
    (ypr, catch, ypc), so the simulator's mean moves toward the market."""
    if not PROPS_ENABLED or props is None:
        return pl
    pl = pl.copy()
    for _, r in props.iterrows():
        m = pl["name_key"] == norm(r["Name"])
        if not m.any():
            continue
        for col, rate in (("rec_yd", "ypr"), ("rush_yd", "ypc")):
            if col in r and pd.notna(r[col]) and f"model_{col}" in r and r[f"model_{col}"] > 0:
                k = (1 - PROPS_WEIGHT) + PROPS_WEIGHT * r[col] / r[f"model_{col}"]
                pl.loc[m, rate] *= k
    return pl
