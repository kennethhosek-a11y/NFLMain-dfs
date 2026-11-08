"""Local NFL history database, 2021 through the last completed 2026 week (free sources only).

Sources (downloaded to data/cache/, not committed):
  nflverse  player + team weekly stats, play-by-play (red-zone usage), snap counts, depth charts,
            injury reports, schedules with closing spread / total (nfldata games.csv)
  RotoGuru  real DraftKings salaries + DK points by week - ONLY 2020-2021 exist (checked
            11/07/2026: 2022-2026 pages are empty), so other weeks get synthetic DK-style salaries
            (DK pricing of each player's trailing average, fitted on the real 2021 salaries, like
            the NBA gate backtest), and from Week 9 2026 on the real DK salary files this system
            fetches are kept (data/research/history/dk_salaries/).

Tables written to data/research/history/ (gzip CSV, committed):
  games.csv.gz        one row per game: kickoff, slate tag, closing spread/total, implied totals,
                      roof, weather, final score
  player_week.csv.gz  QB/RB/WR/TE per game: DK points (computed with DK Classic scoring), stats,
                      targets / carries / air yards / red-zone looks / snap share, depth-chart rank,
                      injury report + practice status, salary (+ source), trailing average
  dst_week.csv.gz     team defenses per game: DK points, sacks, takeaways, TDs, points allowed, salary
  validation.csv      computed DK points vs RotoGuru's real DK points (2021); synthetic vs real salaries
Slate tags (Eastern kickoff): morning (international, before noon Sunday), early (Sunday 12-3 PM),
afternoon (Sunday 3-7 PM), night (Sunday 7 PM+), thanksgiving (Thursday day games in Thanksgiving
week), thu, mon, sat, other. DK's Main slate = early + afternoon.

    python -m research.history.build_db              # build or refresh everything
    python -m research.history.build_db --season 2026  # refresh only this season (Tuesday update)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nflmain import scoring, sources
from nflmain.names import norm
from nflmain.teams import canon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "research" / "history"
SEASONS = list(range(2021, 2027))
B = sources.NFLVERSE
RG_TEAMS = {"gnb": "GB", "kan": "KC", "nor": "NO", "sfo": "SF", "tam": "TB", "lvr": "LV", "oak": "LV",
            "nwe": "NE", "jac": "JAX", "was": "WAS", "lar": "LAR", "lac": "LAC"}
SKILL = ["QB", "RB", "WR", "TE"]


def _fetch(season: int, rel: str, name: str):
    age = 24 * 30 if season < 2026 else 12
    return sources._cached(name, f"{B}/{rel}/{name}", age)


# ------------------------------------------------------------------ games + slate tags

def slate_tag(weekday: str, gametime: str, gameday: str) -> str:
    h = int(str(gametime)[:2]) + int(str(gametime)[3:5]) / 60
    d = pd.Timestamp(gameday)
    if weekday == "Sunday":
        return "morning" if h < 12 else "early" if h < 15 else "afternoon" if h < 19 else "night"
    if weekday == "Thursday":
        return "thanksgiving" if (d.month == 11 and 22 <= d.day <= 28) else "thu"   # all 3 games
    return {"Monday": "mon", "Saturday": "sat"}.get(weekday, "other")


def games() -> pd.DataFrame:
    f = sources._cached("games.csv", "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", 12)
    g = pd.read_csv(f)
    g = g[(g["game_type"] == "REG") & g["season"].isin(SEASONS)].copy()
    for c in ("away_team", "home_team"):
        g[c] = g[c].map(canon)
    g["slate"] = [slate_tag(w, t, d) for w, t, d in zip(g["weekday"], g["gametime"], g["gameday"])]
    g["main"] = g["slate"].isin(["early", "afternoon"])
    # spread_line is the home margin (positive = home favored); implied = total/2 +- spread/2
    g["home_implied"] = g["total_line"] / 2 + g["spread_line"] / 2
    g["away_implied"] = g["total_line"] - g["home_implied"]
    g["completed"] = g["home_score"].notna()
    keep = ["game_id", "season", "week", "gameday", "weekday", "gametime", "slate", "main", "away_team",
            "home_team", "away_score", "home_score", "spread_line", "total_line", "home_implied", "away_implied",
            "roof", "surface", "temp", "wind", "completed", "away_qb_name", "home_qb_name"]
    return g[keep].reset_index(drop=True)


# ------------------------------------------------------------------ players

def dk_points(s: pd.DataFrame) -> pd.Series:
    """DK Classic scoring from nflverse weekly stats."""
    two = s[["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"]].fillna(0).sum(axis=1)
    ret = s["special_teams_tds"].fillna(0) + s["fumble_recovery_tds"].fillna(0)
    f = lambda c: s[c].fillna(0)
    return pd.Series(scoring.offense(pass_yd=f("passing_yards"), pass_td=f("passing_tds"), ints=f("passing_interceptions"),
                                     rush_yd=f("rushing_yards"), rush_td=f("rushing_tds"), rec=f("receptions"),
                                     rec_yd=f("receiving_yards"), rec_td=f("receiving_tds"), fum_lost=f("fumbles_lost_total"),
                                     two_pt=two, ret_td=ret), index=s.index)


def depth_ranks(season: int, g: pd.DataFrame) -> pd.DataFrame:
    """(player_id, week) -> depth rank at his position (1 = starter)."""
    d = pd.read_csv(_fetch(season, "depth_charts", f"depth_charts_{season}.csv"), low_memory=False)
    if "depth_team" in d:                                     # 2021-2024 format: by week
        d = d[(d["game_type"] == "REG") & (d["formation"] == "Offense")]
        d = d[d["position"].isin(SKILL + ["FB"])]
        d["rank"] = pd.to_numeric(d["depth_team"], errors="coerce")
        return d.groupby(["gsis_id", "week"])["rank"].min().rename("depth").reset_index().rename(columns={"gsis_id": "player_id"})
    # 2025+ format: daily snapshots -> the last snapshot before each week's first kickoff
    d = d[d["pos_abb"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
    d["day"] = pd.to_datetime(d["dt"].str[:10])
    first = g[g["season"] == season].groupby("week")["gameday"].min().map(pd.Timestamp)
    rows = []
    for wk, day in first.items():
        snap = d[d["day"] < day]
        if not len(snap):
            continue
        last = snap["day"].max()
        s = snap[snap["day"] == last].groupby("gsis_id")["pos_rank"].min()
        rows.append(pd.DataFrame({"player_id": s.index, "week": wk, "depth": s.values}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["player_id", "week", "depth"])


def players(season: int, g: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(_fetch(season, "stats_player", f"stats_player_week_{season}.csv"), low_memory=False)
    s = s[s["season_type"] == "REG"].copy()
    s["position"] = s["position"].replace({"FB": "RB"})
    s = s[s["position"].isin(SKILL)].copy()
    s["team"] = s["team"].map(canon)
    s["opp"] = s["opponent_team"].map(canon)
    s["dk_pts"] = dk_points(s).round(2)
    s["name_key"] = s["player_display_name"].map(norm)
    # snaps (pfr names -> name_key) and red zone (pbp, by player id)
    sn = pd.read_csv(_fetch(season, "snap_counts", f"snap_counts_{season}.csv"))
    sn = sn[sn["game_type"] == "REG"].assign(team=lambda d: d["team"].map(canon), name_key=lambda d: d["player"].map(norm))
    s = s.merge(sn[["name_key", "team", "week", "offense_snaps", "offense_pct"]].drop_duplicates(["name_key", "team", "week"]),
                on=["name_key", "team", "week"], how="left")
    try:
        rz = sources.redzone_usage(season, max_age_h=24 * 30 if season < 2026 else 12)
        rz = rz.groupby(["player_id", "week"])[["rz_tgt", "rz_car"]].sum().reset_index()
        s = s.merge(rz, on=["player_id", "week"], how="left")
    except Exception as e:  # noqa: BLE001
        print(f"[history] {season}: red zone unavailable ({e})")
    inj = pd.read_csv(_fetch(season, "injuries", f"injuries_{season}.csv"), low_memory=False)
    inj = inj[inj["game_type"] == "REG"][["gsis_id", "week", "report_status", "practice_status"]]
    inj = inj.drop_duplicates(["gsis_id", "week"], keep="last").rename(columns={"gsis_id": "player_id"})
    s = s.merge(inj, on=["player_id", "week"], how="left")
    s = s.merge(depth_ranks(season, g), on=["player_id", "week"], how="left")
    keep = ["season", "week", "game_id", "player_id", "player_display_name", "name_key", "position", "team", "opp",
            "dk_pts", "fantasy_points_ppr", "attempts", "completions", "passing_yards", "passing_tds",
            "passing_interceptions", "sacks_suffered", "carries", "rushing_yards", "rushing_tds", "targets",
            "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards", "target_share",
            "air_yards_share", "fumbles_lost_total", "offense_snaps", "offense_pct", "rz_tgt", "rz_car",
            "depth", "report_status", "practice_status"]
    return s[[c for c in keep if c in s]].rename(columns={"player_display_name": "name", "position": "pos"})


def dsts(season: int, g: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_csv(_fetch(season, "stats_team", f"stats_team_week_{season}.csv"), low_memory=False)
    t = t[t["season_type"] == "REG"].copy()
    t["team"] = t["team"].map(canon)
    t["opp"] = t["opponent_team"].map(canon)
    gg = g[g["season"] == season]
    pa = pd.concat([gg[["game_id", "home_team", "away_score"]].rename(columns={"home_team": "team", "away_score": "pa"}),
                    gg[["game_id", "away_team", "home_score"]].rename(columns={"away_team": "team", "home_score": "pa"})])
    t = t.merge(pa, on=["game_id", "team"], how="left")
    f = lambda c: t[c].fillna(0) if c in t else 0
    blocks = f("def_punt_blocks") + f("def_fg_blocks") + f("def_pat_blocks")
    tds = f("def_tds") + f("special_teams_tds")
    t["dk_pts"] = scoring.dst(sacks=f("def_sacks"), ints=f("def_interceptions"), fum_rec=f("fumble_recovery_opp"),
                              td=tds, safeties=f("def_safeties"), blocks=blocks, pa=t["pa"].fillna(99))
    t["sacks"], t["ints"], t["fum_rec"], t["tds"] = f("def_sacks"), f("def_interceptions"), f("fumble_recovery_opp"), tds
    return t[["season", "week", "game_id", "team", "opp", "dk_pts", "sacks", "ints", "fum_rec", "tds", "pa"]]


# ------------------------------------------------------------------ salaries

def rotoguru(season: int) -> pd.DataFrame:
    """Real DK salaries + DK points (RotoGuru; only 2020-2021 exist). Cached per week."""
    import requests
    rows = []
    for wk in range(1, 19):
        f = sources.CACHE / f"rotoguru_dk_{season}_{wk}.txt"
        if not f.exists():
            try:   # RotoGuru resets HTTPS connections from this machine; its plain-HTTP page works
                r = requests.get("http://rotoguru1.com/cgi-bin/fyday.pl",
                                 params={"week": wk, "year": season, "game": "dk", "scsv": 1},
                                 headers=sources.UA, timeout=60)
                m = re.search(r"<pre>(.*?)</pre>", r.text, re.S)
                f.write_text(m.group(1) if m else "")
            except Exception as e:  # noqa: BLE001
                print(f"[rotoguru] {season} week {wk}: {e}")
                continue
        txt = f.read_text().strip()
        if txt.count("\n") < 10:
            continue
        from io import StringIO
        d = pd.read_csv(StringIO(txt), sep=";")
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows)
    d.columns = ["week", "season", "gid", "name", "pos", "team", "ha", "opp", "rg_dk_pts", "salary"]
    d["team"] = d["team"].map(lambda t: RG_TEAMS.get(str(t).lower(), str(t).upper())).map(canon)
    d["pos"] = d["pos"].replace({"Def": "DST", "PK": "K"})
    first_last = d["name"].map(lambda n: " ".join(reversed([x.strip() for x in str(n).split(",", 1)])))
    d["name_key"] = np.where(d["pos"] == "DST", d["team"].str.lower() + " dst", first_last.map(norm))
    d = d[d["salary"].notna() & (d["salary"].astype(str).str.strip() != "")]
    d["salary"] = pd.to_numeric(d["salary"], errors="coerce")
    return d.dropna(subset=["salary"])


def trailing(df: pd.DataFrame, key: str) -> pd.Series:
    """Each player's DK points average from games BEFORE this one (exp-weighted, half-life 4
    games, last season's games count too at half weight) - what DK prices off."""
    df = df.sort_values(["season", "week"])
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(key, sort=False):
        pts = g["dk_pts"].to_numpy(float)
        season = g["season"].to_numpy()
        vals = np.full(len(g), np.nan)
        for i in range(1, len(g)):
            w = 0.5 ** ((i - 1 - np.arange(i)) / 4.0) * np.where(season[:i] == season[i], 1.0, 0.5)
            vals[i] = (w * pts[:i]).sum() / w.sum()
        out[g.index] = vals
    return out


POS_RANGE = {"QB": (4000, 9000), "RB": (4000, 10000), "WR": (3000, 10000), "TE": (2500, 8500), "DST": (2000, 5000)}
POS_DEFAULT_PTS = {"QB": 12.0, "RB": 4.0, "WR": 4.0, "TE": 3.0, "DST": 6.0}


def _X(trail, implied, pos):
    t = np.where(np.isfinite(trail), trail, [POS_DEFAULT_PTS.get(x, 4.0) for x in pos])
    imp = np.where(np.isfinite(implied), implied, 22.0) - 22.0
    return np.c_[np.ones(len(t)), t, imp]


def fit_pricing(real: pd.DataFrame) -> dict:
    """Per position: salary ~ a + b * trailing points + c * (team implied total - 22), least
    squares on the real 2021 DK salaries (DK prices recent production and the matchup)."""
    fit = {}
    for pos, g in real.dropna(subset=["salary"]).groupby("pos"):
        if pos not in POS_RANGE or len(g) < 50:
            continue
        X = _X(g["trail"].to_numpy(float), g["implied"].to_numpy(float), g["pos"].to_numpy(object))
        coef, *_ = np.linalg.lstsq(X, g["salary"].to_numpy(float), rcond=None)
        fit[pos] = {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2])}
    return fit


def synth_salary(trail: pd.Series, pos: pd.Series, fit: dict, seed: int = 0, implied=None) -> pd.Series:
    rng = np.random.default_rng(seed)
    implied = np.full(len(trail), np.nan) if implied is None else np.asarray(implied, float)
    out = []
    for t, p, im in zip(trail, pos, implied):
        f = fit.get(p)
        if f is None:
            out.append(np.nan)
            continue
        t = POS_DEFAULT_PTS[p] if not np.isfinite(t) else t
        im = 0.0 if not np.isfinite(im) else im - 22.0
        s = (f["a"] + f["b"] * t + f["c"] * im) * np.exp(rng.normal(0, 0.08))   # 8% pricing noise, as in the NBA test
        lo, hi = POS_RANGE[p]
        out.append(float(np.clip(np.round(s / 100) * 100, lo, hi)))
    return pd.Series(out, index=trail.index)


def kept_dk_salaries(g: pd.DataFrame | None = None) -> pd.DataFrame:
    """Real DK Classic salary files saved by every `nflmain run` (2026 Week 9 on), mapped to the
    season/week of the game each player is in."""
    d = OUT / "dk_salaries"
    frames = [pd.read_csv(f) for f in sorted(d.glob("*.csv.gz"))] if d.exists() else []
    if not frames:
        return pd.DataFrame()
    k = pd.concat(frames)
    g = games() if g is None else g
    gm = pd.concat([g[["game_id", "season", "week", "away_team", "home_team"]].assign(
        game=lambda x: x["away_team"] + "@" + x["home_team"], day=g["gameday"])])
    k = k.merge(gm[["game", "day", "season", "week"]], on=["game", "day"], how="inner")
    return k


# ------------------------------------------------------------------ build

def build(seasons=SEASONS) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = games()
    g.to_csv(OUT / "games.csv.gz", index=False)
    old_p = pd.read_csv(OUT / "player_week.csv.gz", low_memory=False) if (OUT / "player_week.csv.gz").exists() else None
    old_d = pd.read_csv(OUT / "dst_week.csv.gz") if (OUT / "dst_week.csv.gz").exists() else None
    pw, dw = [], []
    for s in SEASONS:
        if s in seasons or old_p is None:
            print(f"[history] {s}: players and defenses")
            pw.append(players(s, g))
            dw.append(dsts(s, g))
        else:
            pw.append(old_p[old_p["season"] == s].drop(columns=["trail", "implied", "salary", "sal_source", "showdown_sal"], errors="ignore"))
            dw.append(old_d[old_d["season"] == s].drop(columns=["trail", "implied", "salary", "sal_source", "name_key", "pos"], errors="ignore"))
    p, d = pd.concat(pw, ignore_index=True), pd.concat(dw, ignore_index=True)
    d["name_key"] = d["team"].str.lower() + " dst"
    d["pos"] = "DST"
    p["trail"] = trailing(p, "player_id")
    d["trail"] = trailing(d, "team")
    imp = pd.concat([g[["season", "week", "home_team", "home_implied"]].set_axis(["season", "week", "team", "implied"], axis=1),
                     g[["season", "week", "away_team", "away_implied"]].set_axis(["season", "week", "team", "implied"], axis=1)])
    imp = imp.drop_duplicates(["season", "week", "team"])
    p = p.drop(columns="implied", errors="ignore").merge(imp, on=["season", "week", "team"], how="left")
    d = d.drop(columns="implied", errors="ignore").merge(imp, on=["season", "week", "team"], how="left")
    # real salaries: RotoGuru 2021, kept DK files 2026+
    rg = rotoguru(2021)
    real = pd.concat([rg[["season", "week", "name_key", "team", "salary", "rg_dk_pts"]]], ignore_index=True)
    cols = ["season", "week", "name_key", "team", "pos", "trail", "implied", "dk_pts"]
    allp = pd.concat([p[cols], d[cols]], ignore_index=True)
    m = allp.merge(real, on=["season", "week", "name_key", "team"], how="left")
    fit = fit_pricing(m.dropna(subset=["salary"]))
    print("[history] DK pricing fitted on real 2021 salaries: " +
          "; ".join(f"{k} ${v['a']:,.0f} + ${v['b']:,.0f}/pt" for k, v in fit.items()))
    val = []
    m21 = m[(m["season"] == 2021) & m["salary"].notna()]
    for pos, gp in m21.groupby("pos"):
        val.append({"check": f"DK points: ours vs RotoGuru ({pos}, 2021)", "n": len(gp),
                    "mean_abs_diff": round(float((gp["dk_pts"] - gp["rg_dk_pts"]).abs().mean()), 3),
                    "share_within_0.1": round(float(((gp["dk_pts"] - gp["rg_dk_pts"]).abs() <= 0.11).mean()), 3)})
    kept = kept_dk_salaries(g)
    for df, key in ((p, "player"), (d, "dst")):
        mm = df.merge(real[["season", "week", "name_key", "team", "salary"]], on=["season", "week", "name_key", "team"], how="left")
        if len(kept):
            kk = kept.groupby(["season", "week", "name_key", "team"])["salary"].max().rename("kept").reset_index()
            mm = mm.merge(kk, on=["season", "week", "name_key", "team"], how="left")
            mm["salary"] = mm["salary"].fillna(mm["kept"])
            mm = mm.drop(columns="kept")
        syn = synth_salary(mm["trail"], mm["pos"], fit, seed=7 if key == "player" else 8, implied=mm["implied"])
        mm["sal_source"] = np.where(mm["salary"].notna(), "real", "synthetic")
        if key == "player":
            r = mm[(mm["season"] == 2021) & mm["salary"].notna()]
            for pos, gp in r.groupby("pos"):
                sy = syn[gp.index]
                val.append({"check": f"synthetic vs real salary ({pos}, 2021)", "n": len(gp),
                            "corr": round(float(np.corrcoef(sy, gp["salary"])[0, 1]), 3),
                            "mean_abs_diff": round(float((sy - gp["salary"]).abs().mean()), 0)})
        mm["salary"] = mm["salary"].fillna(syn)
        if key == "player":
            p = mm
        else:
            d = mm
    p = attach_showdown(p)
    p.to_csv(OUT / "player_week.csv.gz", index=False, float_format="%.4g")
    d.to_csv(OUT / "dst_week.csv.gz", index=False, float_format="%.4g")
    pd.DataFrame(val).to_csv(OUT / "validation.csv", index=False)
    wk = g.groupby(["season", "week"])["completed"].all()
    done = wk[wk].reset_index().groupby("season")["week"].max().to_dict()
    print(f"[history] {len(g)} games, {len(p):,} player-games, {len(d):,} defense-games; last completed week "
          f"by season: {done}")
    print(pd.DataFrame(val).to_string(index=False))


def attach_showdown(p: pd.DataFrame) -> pd.DataFrame:
    """Real DK Showdown FLEX salaries (2025 W1 - 2026 W8) from the NFL Showdown database, copied
    read-only from the nfl-dfs repo if it is present next to this repo. Reference only."""
    f = ROOT.parent / "nfl-dfs" / "data" / "research" / "showdown_db" / "players.csv"
    if not f.exists():
        return p
    sd = pd.read_csv(f)
    sd["name_key"] = sd["player"].map(norm)
    sd["team"] = sd["team"].map(canon)
    sd = sd.groupby(["season", "week", "name_key", "team"])["sal"].max().rename("showdown_sal").reset_index()
    return p.merge(sd, on=["season", "week", "name_key", "team"], how="left")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, action="append", help="refresh only these seasons")
    a = ap.parse_args(argv)
    build(a.season or SEASONS)


if __name__ == "__main__":
    main()
