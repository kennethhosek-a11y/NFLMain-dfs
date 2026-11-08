"""Classic pipeline: projections for a whole slate (run) and portfolio builds (build).

run:   DK salary file for the slate's draft group -> injuries (DK tag + ESPN, the more severe) ->
       research overrides -> per game: ESPN line, weather, usage model, game-script simulation ->
       projections (mean of the sims), floor/ceiling, ceiling gate, ownership.
build: candidate lineups (best stacked lineup in many simulated slates + structured QB stacks with
       1-3 pass-catchers and 0/1 bring-backs + neighbors) -> simulated field from ownership ->
       every candidate ranked against the field in every simulated slate and paid by the DK payout
       table -> portfolio picked as a whole (50% cap, 0-or-5+ rule) -> swaps confirmed on an
       independent set of sims.
"""
from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import gamesim, overrides, ownership, project, slate, sources
from . import lineups as L
from .contests import Contest, evaluate, field_dupe_rate
from .parallel import pmap_init
from .portfolio import Extra, caps, confirm_swaps, meets_min_count, min_count, per_sim_value, select
from .settings import for_type
from .settings import get as S

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
HIST = ROOT / "data" / "history"
STATE = ROOT / "data" / "state"
ET = sources.ET
SEV = {"OUT": 3, "D": 2, "Q": 1, "": 0}


@dataclass
class Projections:
    day: str
    label: str                       # output folder name: main / early / afternoon / turbo / thanksgiving
    slate_type: str                  # settings group: early / main / afternoon / thanksgiving
    draft_group: int
    df: pd.DataFrame                 # one row per player (whole pool), sim inputs + projections
    games: list                      # per game: info, teams (project.Team), weather
    fp: np.ndarray | None = None     # (n, proj_sims) simulated points (overrides applied)
    meta: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return f"{self.day}_{self.label}"


def outdir(day: str, label: str) -> Path:
    d = OUT / f"{day}_{label}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def label_for(g: dict) -> str:
    s = (g.get("suffix") or "").lower()
    if "turbo" in s:
        return "turbo"
    return g.get("slate_type", "main")


def merge_injuries(pool: pd.DataFrame, inj: pd.DataFrame | None) -> pd.DataFrame:
    pool = pool.copy()
    pool["dk_status"] = pool["status"]
    pool["espn_status"] = ""
    pool["injury_note"] = ""
    if inj is None or not len(inj):
        return pool
    inj = inj[inj["team"].isin(pool["team"].unique()) | (inj["team"] == "")]
    m = inj.drop_duplicates("name_key").set_index("name_key")
    for i, r in pool.iterrows():
        k = r["name_key"]
        if k in m.index:
            e = m.loc[k]
            pool.at[i, "espn_status"] = e["espn_status"]
            pool.at[i, "injury_note"] = e["comment"]
            if SEV.get(e["status"], 0) > SEV.get(r["status"], 0):     # the more severe wins
                pool.at[i, "status"] = e["status"]
    return pool


# ------------------------------------------------------------------ run (projections)

def run(slate_type: str | None = None, day: str | None = None, draft_group: int | None = None,
        salaries: str | None = None, n_sims: int | None = None, seed: int = 11, save: bool = True,
        quiet: bool = False, inj: pd.DataFrame | None = None, data: dict | None = None) -> Projections:
    from .settings import slate_date
    day = day or slate_date().isoformat()
    n_sims = int(S("build.proj_sims")) if n_sims is None else n_sims
    if salaries:
        pool = slate.load_classic(salaries)
        stype = slate_type or slate.slate_type_of(pool)
        g = {"draft_group": draft_group or 0, "slate_type": stype, "suffix": ""}
    else:
        g = slate.find_group(day, slate_type, draft_group)
        pool = slate.load_classic(slate.fetch_csv(g["draft_group"]))
        stype = g["slate_type"]
        if stype == "other":
            raise RuntimeError(f"DK slate {g['draft_group']} {g['suffix']} isn't an Early/Main/Afternoon/"
                               "Thanksgiving slate - this system doesn't build it")
    label = label_for(g)
    if inj is None:
        try:
            inj = sources.espn_injuries()
        except Exception as e:  # noqa: BLE001
            print(f"[injuries] ESPN injury report unavailable ({e}); DK tags only")
    pool = merge_injuries(pool, inj)
    ov = overrides.load(day, stype, quiet=quiet)
    pool = overrides.apply_status(pool, ov)
    data = data or load_data(day)
    lines = {}
    try:
        for x in sources.espn_games(date.fromisoformat(day)):
            lines[f"{x['away']}@{x['home']}"] = x
    except Exception as e:  # noqa: BLE001
        print(f"[lines] ESPN scoreboard unavailable ({e}); league-average totals")
    pool["pos"] = pool["Position"].astype(str)
    frames, games = [], []
    for gm in sorted(pool["game"].unique(), key=lambda x: pool.loc[pool["game"] == x, "kickoff"].iloc[0]):
        sub = pool[pool["game"] == gm]
        away, home = gm.split("@")
        kick = sub["kickoff"].iloc[0]
        gi = lines.get(gm) or {"away": away, "home": home, "total": None, "spread_home": None,
                               "kickoff": kick, "indoor": False, "line_source": "none"}
        if gi.get("total") is None and not quiet:
            print(f"[lines] {gm}: no ESPN line yet - league-average total")
        wx = sources.weather(home, kick, gi.get("indoor", False)) if kick else {}
        teams, pl = project.build(sub, gi, data["stats"], data["snaps"], data["rz"], wx,
                                  th=data["th"], use=data["use"])
        pl.index = sub.index
        frames.append(pl)
        games.append({"game": gm, "away": away, "home": home, "kickoff": kick, "info": gi, "weather": wx,
                      "teams": teams})
    df = pd.concat(frames).sort_index()
    df["ProjSource"] = "model"
    p = Projections(day, label, stype, int(g.get("draft_group") or 0), df, games)
    fp, res = simulate(p, n_sims, seed)
    fp = overrides.apply_projection(df, fp, ov) if not quiet else _quiet(overrides.apply_projection, df, fp, ov)
    p.df = summarize(df, fp)
    own = ownership.project(p.df)
    own = overrides.apply_ownership(p.df, own, ov) if not quiet else _quiet(overrides.apply_ownership, p.df, own, ov)
    p.df["Ownership"] = np.round(own, 1)
    p.df.loc[p.df["Projection"] <= 0.05, "Ownership"] = 0.0
    p.fp = fp
    gate(p)
    p.meta["scores"] = {t: {"mean": float(s.mean()), "p10": float(np.percentile(s, 10)),
                            "p90": float(np.percentile(s, 90))} for t, s in res["score"].items()}
    if save:
        save_run(p)
    if not quiet:
        print_run(p)
    return p


def _quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def load_data(day: str) -> dict:
    season = sources.season_of(date.fromisoformat(day))
    stats = sources.weekly_stats(season)
    snaps = sources.snap_counts(season)
    try:
        rz = sources.redzone_usage(season)
    except Exception as e:  # noqa: BLE001
        print(f"[data] red-zone data unavailable ({e})")
        rz = None
    return {"stats": stats, "snaps": snaps, "rz": rz, "th": project.team_history(stats),
            "use": project.player_usage(stats, snaps, rz)}


def simulate(p: Projections, n: int, seed: int, df: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    """Every game on the slate played out (game-first): (n_players, n) DK points in df row order."""
    df = p.df if df is None else df
    fp = np.zeros((len(df), n), dtype=np.float32)
    scores = {}
    pos_index = {ix: k for k, ix in enumerate(df.index)}
    for k, g in enumerate(p.games):
        rows = df.index[df["game"] == g["game"]]
        if not len(rows):
            continue
        sub = df.loc[rows].reset_index(drop=True)
        res = gamesim.simulate(g["teams"], sub, n, seed=seed + 7919 * k)
        fp[[pos_index[r] for r in rows]] = res["fp"]
        scores.update(res["score"])
    return fp, {"score": scores}


def summarize(df: pd.DataFrame, fp: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["Projection"] = fp.mean(axis=1)
    df["Floor"] = np.percentile(fp, 25, axis=1)
    df["Ceiling"] = np.percentile(fp, 90, axis=1)
    df["Boom%"] = (fp >= np.maximum(3 * df["Salary"].to_numpy()[:, None] / 1000, 1)).mean(axis=1) * 100
    df["Value"] = df["Projection"] / (df["Salary"] / 1000).clip(lower=0.2)
    return df


def gate(p: Projections) -> None:
    """Ceiling gate (NBA rule, per slate type): winning pace = the best projected lineup's points
    per $1k + 0.1; among playable players (projection >= 3), the bottom gate_pct at each position by
    how often they reach that pace in the sims are kept out of OUR lineups (not the field's)."""
    df, fp = p.df, p.fp
    pct = float(for_type(p.slate_type, "gate_pct"))
    P = L.Pool.from_df(df)
    opt = L.Optimizer(P, L.Rules.for_slate(p.slate_type), eligible=(df["Projection"] > 0.3).to_numpy())
    best = opt.solve(df["Projection"].to_numpy())
    best_pts = float(df["Projection"].to_numpy()[best].sum()) if best is not None else 0.0
    pace = best_pts / (slate.CAP / 1000) + 0.1
    hit = (fp / np.maximum(df["Salary"].to_numpy()[:, None] / 1000, 0.2) >= pace).mean(axis=1) * 100
    df["vsPace"] = (df["Value"] - pace).round(2)
    df["HitPace%"] = hit.round(1)
    df["Gate"] = False
    if pct > 0:
        for pos in ("QB", "RB", "WR", "TE", "DST"):
            m = (df["pos"] == pos) & (df["Projection"] >= 3)
            if m.sum() < 4:
                continue
            cut = np.quantile(df.loc[m, "HitPace%"], pct)
            df.loc[m & (df["HitPace%"] < cut), "Gate"] = True
    p.meta["pace"] = pace
    p.meta["best_proj"] = best_pts


def save_run(p: Projections) -> None:
    d = outdir(p.day, p.label)
    cols = ["Name", "ID", "pos", "team", "opp", "game", "kickoff", "Salary", "status", "dk_status", "espn_status",
            "injury_note", "Projection", "Floor", "Ceiling", "Boom%", "Value", "vsPace", "HitPace%", "Gate",
            "Ownership", "ProjSource", "ResearchAdj", "ResearchNote", "games", "snap_pct", "tgt_share",
            "car_share", "rz_tgt_share", "rz_car_share", "ay_share", "AvgPointsPerGame"]
    out = p.df[[c for c in cols if c in p.df]].sort_values("Projection", ascending=False)
    out.to_csv(d / "projections.csv", index=False)
    stamp = datetime.now(ET).strftime("%Y%m%d_%H%M")
    HIST.mkdir(parents=True, exist_ok=True)
    out.to_csv(HIST / f"proj_{p.day}_{p.label}_{stamp}.csv", index=False)
    games = []
    for g in p.games:
        gi = g["info"]
        games.append({"away": g["away"], "home": g["home"], "kickoff": g["kickoff"], "total": gi.get("total"),
                      "spread_home": gi.get("spread_home"), "line_source": gi.get("line_source", ""),
                      "weather": g["weather"],
                      "teams": {t: {"implied": round(tm.implied, 2), "notes": tm.notes,
                                    "sim_mean": round(p.meta.get("scores", {}).get(t, {}).get("mean", 0), 1),
                                    "sim_p10": round(p.meta.get("scores", {}).get(t, {}).get("p10", 0), 1),
                                    "sim_p90": round(p.meta.get("scores", {}).get(t, {}).get("p90", 0), 1)}
                                for t, tm in g["teams"].items()}})
    meta = {"day": p.day, "slate": p.label, "slate_type": p.slate_type, "draft_group": p.draft_group,
            "games": games, "pace": p.meta.get("pace"), "best_proj": p.meta.get("best_proj"),
            "windows": slate.windows(p.df), "built": datetime.now(ET).isoformat()}
    (d / "slate.json").write_text(json.dumps(meta, indent=1, default=str))
    stacks(p).to_csv(d / "stacks.csv", index=False)
    keep_salaries(p)


def keep_salaries(p: Projections) -> None:
    """Keep this slate's real DK salaries for the history database (research/history): one file
    per game day and slate, replaced on every run."""
    k = p.df[["Name", "name_key", "pos", "team", "game", "Salary", "kickoff"]].rename(columns={"Salary": "salary"})
    k = k.assign(day=k["kickoff"].str[:10], slate=p.label).drop(columns="kickoff")
    d = ROOT / "data" / "research" / "history" / "dk_salaries"
    d.mkdir(parents=True, exist_ok=True)
    k.to_csv(d / f"{p.day}_{p.label}.csv.gz", index=False)


def stacks(p: Projections, top: int = 3) -> pd.DataFrame:
    """Each QB with his best 1-2 pass-catchers (+ the best bring-back): summed projection, 90th
    percentile of the stack's simulated points, ownership product, QB-catcher correlation."""
    df, fp = p.df.reset_index(drop=True), p.fp
    rows = []
    for qi in np.flatnonzero((df["pos"] == "QB") & (df["Projection"] > 5)):
        t = df.at[qi, "team"]
        mates = df[(df["team"] == t) & df["pos"].isin(["WR", "TE"]) & (df["Projection"] > 2)]
        mates = mates.sort_values("Projection", ascending=False).head(top)
        opp = df[(df["team"] == df.at[qi, "opp"]) & df["pos"].isin(["WR", "TE", "RB"])]
        bb = opp.sort_values("Projection", ascending=False).index[:1].tolist()
        for k in (1, 2):
            for combo in ([mates.index[:k].tolist()] if len(mates) >= k else []):
                for b in ([[]] + ([bb] if bb else [])):
                    ids = [qi] + combo + b
                    sim = fp[ids].sum(axis=0)
                    rows.append({"QB": df.at[qi, "Name"], "team": t,
                                 "partners": ", ".join(df.loc[combo + b, "Name"]) + (" (bring-back)" if b else ""),
                                 "Salary": int(df.loc[ids, "Salary"].sum()), "Proj": round(float(sim.mean()), 1),
                                 "Ceiling": round(float(np.percentile(sim, 90)), 1),
                                 "Own": round(float(np.prod(df.loc[ids, "Ownership"] / 100) * 100), 3),
                                 "corr": round(float(np.corrcoef(fp[qi], fp[combo[0]])[0, 1]), 2)})
    return pd.DataFrame(rows).sort_values("Ceiling", ascending=False) if rows else pd.DataFrame(rows)


def print_run(p: Projections) -> None:
    print(f"\n{p.label.upper()} slate {p.day} (DK draft group {p.draft_group}, type {p.slate_type}) - "
          f"{len(p.games)} games, locks: {', '.join(k[11:16] for k in slate.windows(p.df))} ET")
    for g in p.games:
        gi = g["info"]
        imp = ", ".join(f"{t} {tm.implied:.1f}" for t, tm in g["teams"].items())
        wx = g["weather"]
        w = f" | wind {wx.get('wind')} mph, rain {wx.get('precip')}%" if wx and not wx.get("indoor") else ""
        print(f"  {g['game']:9s} {g['kickoff'][11:16]}  total {gi.get('total')}  implied {imp}{w}")
    d = p.df[p.df["Projection"] > 0.5].sort_values("Projection", ascending=False)
    print(d[["Name", "pos", "team", "Salary", "status", "Projection", "Ceiling", "Ownership", "HitPace%", "Gate"]]
          .head(30).round(1).to_string(index=False))
    print(f"  winning pace {p.meta.get('pace', 0):.2f} pts/$1k (best projected lineup {p.meta.get('best_proj', 0):.1f}); "
          f"gate drops {int(p.df['Gate'].sum())} players")


# ------------------------------------------------------------------ candidates (parallel)

_W: dict = {}


def _init(P, rules, eligible):
    _W["opt"] = L.Optimizer(P, rules, eligible)


def _solve_job(job):
    pts, kw = job
    lu = _W["opt"].solve(pts, **kw)
    return None if lu is None else lu.tolist()


def solve_many(P, rules, eligible, jobs, workers=None) -> list:
    """Solve many optimizer jobs (pts, kwargs) - in spawned worker processes when there are enough."""
    workers = int(S("build.workers")) if workers is None else workers
    if workers <= 1 or len(jobs) < 40:
        opt = L.Optimizer(P, rules, eligible)
        out = [opt.solve(pts, **kw) for pts, kw in jobs]
        return [x for x in out if x is not None]
    out = pmap_init(_solve_job, jobs, _init, (P, rules, eligible), workers=workers)
    return [np.array(x) for x in out if x is not None]


def candidates(p: Projections, pool: pd.DataFrame, fp: np.ndarray, rules: L.Rules, allow: np.ndarray,
               n_worlds: int, rng, n_neighbors: int = 40) -> np.ndarray:
    P = L.Pool.from_df(pool)
    proj = fp.mean(axis=1)
    ceil = np.percentile(fp, 85, axis=1)
    elig = (proj > 0.3) & allow
    jobs = []
    for s in rng.choice(fp.shape[1], size=min(n_worlds, fp.shape[1]), replace=False):
        jobs.append((fp[:, s], {} if s % 2 else {"rb_max": 2}))   # half without an RB in FLEX (M11)
    # structured stacks: each reasonable QB with 1..stack_max pass-catchers, with / without a bring-back
    qbs = np.flatnonzero((pool["pos"] == "QB").to_numpy() & elig)
    qbs = qbs[np.argsort(-ceil[qbs])][:16]
    bbs = [None] if rules.bringback == 0 else ([1] if rules.bringback >= 2 else [0, 1])
    for q in qbs:
        t = pool["team"].values[q]
        for k in range(max(rules.stack_min, 1), rules.stack_max + 1):
            for b in bbs:
                for pts in (proj, ceil, 0.5 * (proj + ceil), proj * rng.lognormal(0, 0.25, len(proj))):
                    jobs.append((pts, {"team": t, "qb_mates": k, "bringback": b}))
                    jobs.append((pts, {"team": t, "qb_mates": k, "bringback": b, "rb_max": 2}))
    for _ in range(n_worlds // 4):
        jobs.append((proj * rng.lognormal(0, 0.3, len(proj)), {}))
    out = solve_many(P, rules, elig.astype(float), jobs)
    base = np.unique(L.canon(np.array(out)), axis=0)
    # neighbors: the best lineup without each player of the top builds (room under exposure caps)
    top = base[np.argsort(-L.scores(base, proj[:, None])[:, 0])][:max(n_neighbors, 0)]
    mix = 0.5 * (proj + ceil)
    nb_jobs = [(mix, {"banned": [int(i)]}) for lu in top for i in lu]
    nb_jobs = list({(tuple(kw["banned"])): (pts, kw) for pts, kw in nb_jobs}.values())
    out += solve_many(P, rules, elig.astype(float), nb_jobs)
    lus = np.unique(L.canon(np.array(out)), axis=0)
    ok = np.array([L.valid(lu, P, rules) for lu in lus])
    return lus[ok]


# ------------------------------------------------------------------ build

@dataclass
class BuildResult:
    picks: dict                   # contest id -> list of lineups (row indices into pool)
    table: pd.DataFrame
    players: pd.DataFrame
    plan: list
    pool: pd.DataFrame


def build_pool(p: Projections) -> tuple[pd.DataFrame, np.ndarray]:
    keep = (p.df["Projection"].to_numpy() > 0.05) & ~p.df["status"].isin(["OUT"]).to_numpy()
    pool = p.df[keep].copy()
    return pool, keep


def sims_for(p: Projections, pool: pd.DataFrame, n: int, seed: int, live_fn=None) -> np.ndarray:
    fp, _ = simulate(p, n, seed, df=pool)
    ov = overrides.load(p.day, p.slate_type, quiet=True)
    tmp = pool.copy()
    tmp["ProjSource"] = "model"
    fp = _quiet(overrides.apply_projection, tmp, fp, ov)
    if live_fn is not None:
        fp = live_fn(fp)
    return fp


def core_players(chosen: np.ndarray, pool: pd.DataFrame, n: int, k: int, proj: np.ndarray, ext=None,
                 widen: float = 1.0) -> np.ndarray:
    """The players a 0-or-k+ portfolio of n lineups is built from, most-used first (ties:
    projection), keeping enough at each position to fill lineups. Size: at most 9n/k (everyone
    can reach k lineups) but comfortably more than 9n/cap (nobody has to pass the exposure cap)."""
    cnt = np.bincount(chosen.ravel(), minlength=len(pool))
    used = np.flatnonzero(cnt > 0)
    slots = 9 * n
    cap = caps(n)
    # each position needs one more player than the cap strictly requires, so there is room to choose
    pc = (ext.pos_caps if ext is not None else {})
    cq = max(k, int(np.floor(pc.get("QB", 1.0) * n + 1e-9)), 1) if "QB" in pc else cap
    cd = max(k, int(np.floor(pc.get("DST", 1.0) * n + 1e-9)), 1) if "DST" in pc else cap
    need = {"QB": -(-n // min(cq, cap)) + 1, "RB": -(-2 * n // cap) + 1, "WR": -(-3 * n // cap) + 1,
            "TE": -(-n // cap) + 1, "DST": -(-n // min(cd, cap)) + 1}
    need = {p_: int(np.ceil(m * widen)) for p_, m in need.items()}
    limit = max(int(widen * slots / k), sum(need.values()))
    order = used[np.lexsort((-proj[used], -cnt[used]))]
    pos = pool["pos"].to_numpy(object)
    if any((pos[used] == p_).sum() < m for p_, m in need.items()):      # first portfolio too narrow:
        extra = np.flatnonzero((proj > 0.3) & ~np.isin(np.arange(len(pool)), used))   # top up by projection
        order = np.r_[order, extra[np.argsort(-proj[extra])]]
    keep = []
    for p_, m in need.items():                         # position minimums first
        keep += [i for i in order if pos[i] == p_][:m]
    for i in order:
        if len(keep) >= limit:
            break
        if i not in keep:
            keep.append(i)
    return np.array(sorted(set(keep)))


def stack_team(lus: np.ndarray, pool: pd.DataFrame) -> list:
    """QB team of each lineup (for the stack-team cap)."""
    pos, team = pool["pos"].to_numpy(object), pool["team"].to_numpy(object)
    return [team[[i for i in lu if pos[i] == "QB"][0]] for lu in lus]


def build(p: Projections, contests: list[Contest], n_sims: int | None = None, n_worlds: int | None = None,
          field_n: int | None = None, seed: int = 101, exclude_players: set | None = None) -> BuildResult:
    n_sims = int(S("build.sims")) if n_sims is None else n_sims
    n_worlds = int(S("build.worlds")) if n_worlds is None else n_worlds
    field_n = int(S("build.field_sample")) if field_n is None else field_n
    min_salary, min_frac = int(S("build.min_salary")), float(S("build.min_proj_frac"))
    pool, _ = build_pool(p)
    pool = pool.reset_index(drop=True)
    rules = L.Rules.for_slate(p.slate_type)
    allow = ~pool["Gate"].to_numpy(bool)
    if exclude_players:
        allow &= ~pool["Name"].isin(exclude_players).to_numpy()
    gated = pool.loc[~allow & (pool["Projection"] > 3), "Name"].tolist()
    if gated:
        print(f"[gate] kept out of my lineups (ceiling gate {for_type(p.slate_type, 'gate_pct'):.0%}): "
              f"{', '.join(gated[:25])}{' ...' if len(gated) > 25 else ''}")
    total_entries = sum(len(c.entry_ids) or 1 for c in contests)
    big_n = max((len(c.entry_ids) or 1) for c in contests) if contests else 1
    n_worlds = max(n_worlds, min(3 * total_entries, 1500))
    fpA = sims_for(p, pool, n_sims, seed)
    fpB = sims_for(p, pool, n_sims, seed + 1)
    rng = np.random.default_rng(seed)
    t0 = datetime.now()
    lus = candidates(p, pool, fpA, rules, allow, n_worlds, rng, n_neighbors=max(40, min(big_n, 150)))
    # Kenny 11/08 "inside-out": a few safe lineups (every player at or below his position's usual bust
    # rate), then more risk step by step, then the top-1% chase. Bust = under 2x salary in the sims.
    inside_out = bool(int(S("rules.inside_out")))
    sal_k = pool["Salary"].to_numpy(float) / 1000
    bust = (fpA < 2.0 * sal_k[:, None]).mean(axis=1)
    playable = fpA.mean(axis=1) > 5
    pos_arr = pool["pos"].to_numpy(object)
    med = {q: float(np.median(bust[(pos_arr == q) & playable])) if ((pos_arr == q) & playable).any() else 1.0
           for q in ("QB", "RB", "WR", "TE", "DST")}
    # risk = where a player's bust rate ranks within his position (0 = safest, 1 = riskiest),
    # among playable players; tiers: safest third, safer half, safer two-thirds
    rel_risk = np.ones(len(pool))
    for q in ("QB", "RB", "WR", "TE", "DST"):
        m = (pos_arr == q) & playable
        if m.any():
            ref = np.sort(bust[m])
            rel_risk[pos_arr == q] = np.searchsorted(ref, bust[pos_arr == q], side="right") / len(ref)
    TIERS = (0.34, 0.5, 0.67)
    if inside_out:
        P0 = L.Pool.from_df(pool)
        proj0 = fpA.mean(axis=1)
        p25 = np.percentile(fpA, 25, axis=1)
        ceil0 = np.percentile(fpA, 85, axis=1)
        qbs = np.flatnonzero((pos_arr == "QB") & (proj0 > 12))
        tjobs = []
        for thr in TIERS:
            elig_t = ((proj0 > 0.3) & allow & (rel_risk <= thr)).astype(float)
            pts = 0.5 * (proj0 + p25) if thr == TIERS[0] else 0.5 * (proj0 + ceil0)
            for q in qbs:
                if rel_risk[q] <= thr:
                    tjobs.append((elig_t, pts, {"team": pool["team"].values[q]}))
                    tjobs.append((elig_t, pts * rng.lognormal(0, 0.1, len(pts)), {"team": pool["team"].values[q], "rb_max": 2}))
            for s in rng.choice(fpA.shape[1], size=40, replace=False):
                tjobs.append((elig_t, 0.5 * pts + 0.5 * fpA[:, s], {}))
        by_elig: dict = {}
        for elig_t, pts, kw in tjobs:
            by_elig.setdefault(elig_t.tobytes(), (elig_t, []))[1].append((pts, kw))
        added = []
        for elig_t, jobs_t in by_elig.values():
            added += solve_many(P0, rules, elig_t, jobs_t)
        if added:
            lus = np.unique(np.vstack([lus, L.canon(np.array(added))]), axis=0)
        print(f"[build] inside-out: {len(added)} lower-risk candidate builds (position bust medians "
              + ", ".join(f"{q} {100 * v:.0f}%" for q, v in med.items()) + ")")
        safe_names = pool.loc[(rel_risk <= TIERS[0]) & playable, "Name"].tolist()
        print(f"[build] safest third by position: {', '.join(safe_names[:40])}")
    print(f"[build] {len(lus)} candidate lineups ({(datetime.now() - t0).seconds}s)")
    P = L.Pool.from_df(pool)
    own = pool["Ownership"].to_numpy(float)
    fld = L.field(field_n, own, P, rng, float(S("ownership.field_stack")), float(S("ownership.field_double")),
                  float(S("ownership.field_bringback")), int(S("ownership.field_min_salary")))
    proj_all = fpA.mean(axis=1)
    q = lambda x: np.prod(own[x] / 100, axis=1)

    def score_all(lus):
        d = field_dupe_rate([L.key(x) for x in lus], [L.key(x) for x in fld], q(lus), q(fld))
        return d, L.scores(lus, fpA), L.scores(lus, fpB)

    # quality floor: salary and projection (relaxed step by step only if entries can't be filled)
    lproj = L.scores(lus, proj_all[:, None])[:, 0]
    best_proj = lproj.max()
    # salary never relaxes below the hard $49,500 floor (workbook M10); only the projection floor does
    floors = [(min_frac, min_salary), (0.80, min_salary), (0.75, min_salary), (0.0, min_salary)]
    for frac, msal in floors:
        keep = (L.salary(lus, P.sal) >= msal) & (lproj >= frac * best_proj)
        if keep.sum() >= min(600, 10 * big_n + 100) or frac == 0:
            break
    lus = lus[keep]
    floor_note = "" if (frac, msal) == floors[0] else f"quality floor relaxed to {frac:.0%} of the best projection"
    # workbook M13 (soft): lineup total ownership 75-150% - applied only while enough candidates remain
    osum = own[lus].sum(axis=1)
    band = (osum >= float(S("rules.own_sum_min"))) & (osum <= float(S("rules.own_sum_max")))
    if band.sum() >= min(400, 6 * big_n + 60):
        lus = lus[band]
    else:
        print(f"[build] ownership band 75-150% left too few lineups ({int(band.sum())}); not applied")
    # cap-aware passes so balanced portfolios exist to choose from (big entry counts)
    if big_n >= 5:
        opt = L.Optimizer(P, rules, ((proj_all > 0.3) & allow).astype(float))
        cap = caps(big_n)
        extra = []
        for _ in range(max(2, min(6, 300 // big_n))):
            cnt = np.zeros(len(pool), int)
            for _k in range(big_n):
                s = rng.integers(fpA.shape[1])
                pts = 0.5 * fpA[:, s] + 0.5 * proj_all * rng.lognormal(0, 0.15, len(proj_all))
                lu = opt.solve(pts, banned=list(np.flatnonzero(cnt >= cap)), time_limit=5)
                if lu is None:
                    break
                extra.append(lu)
                cnt[lu] += 1
        if extra:
            ex = L.canon(np.array(extra))
            ep = L.scores(ex, proj_all[:, None])[:, 0]
            ex = ex[(L.salary(ex, P.sal) >= msal) & (ep >= frac * best_proj)]
            lus = np.unique(np.vstack([lus, ex]), axis=0)
            print(f"[build] + cap-aware portfolio builds -> {len(lus)} candidates")
    dupe, scA, scB = score_all(lus)
    stk = stack_team(lus, pool)
    pos_all = pool["pos"].to_numpy(object)
    exc = overrides.exposure_exceptions(pool, overrides.load(p.day, p.slate_type, quiet=True))

    def make_ext(lus_):
        e = Extra.from_settings(pos_all, L.shapes(lus_, P, float(S("rules.cheap_salary"))), P.sal)
        e.exceptions = exc
        if inside_out:
            worst = rel_risk[lus_].max(axis=1)
            for name, thr in zip(("risk0", "risk5", "risk12"), TIERS):
                e.props[name] = worst <= thr + 1e-9
        return e

    def tier_bands(e, n):
        """Lower bounds on the lowest-risk tiers for a contest of n entries (cumulative)."""
        if not inside_out:
            return
        n_safe = max(2, int(round(n * 5 / 150)))
        mid = float(S("rules.middle_share"))
        e.bands["risk0"] = (min(n_safe / n, 1.0), min(max(2 * n_safe / n, 0.1), 1.0))   # about n_safe
        e.bands["risk5"] = (min(max(n_safe / n, mid / 2), 1.0), 1.0)
        e.bands["risk12"] = (min(max(n_safe / n, mid), 1.0), 1.0)
    ext = make_ext(lus)
    picks, plan, rows = {}, [], []
    used: set[int] = set()
    for c in contests:
        n = len(c.entry_ids) or 1
        tier_bands(ext, n)
        mA = evaluate(scA, fld, fpA, c, dupe)
        k0 = min_count(n, True)
        t1 = datetime.now()
        chosen = select(mA, lus, n, c.large, exclude=used, initial=False, stack=stk, quiet=True, extra=ext)
        if k0 > 1 and len(chosen) == n:
            # 0-or-5+ rule: rebuild candidates from only the players this first portfolio uses, so
            # lineups share players, then pick the whole set again with the rule on
            for widen in (1.0, 1.6, 2.4):
                U = core_players(lus[chosen], pool, n, k0, proj_all, ext, widen)
                eu = np.zeros(len(pool))
                eu[U] = 1.0
                eu *= ((proj_all > 0.3) & allow)
                jobs = [(fpA[:, s], {} if j % 2 else {"rb_max": 2})
                        for j, s in enumerate(rng.choice(fpA.shape[1], size=min(600, 25 * n), replace=False))]
                jobs += [(proj_all * rng.lognormal(0, 0.2, len(proj_all)), {}) for _ in range(min(150, 5 * n))]
                new = solve_many(P, rules, eu, jobs)
                if new:
                    new = np.unique(L.canon(np.array(new)), axis=0)
                    npj = L.scores(new, proj_all[:, None])[:, 0]
                    new = new[(L.salary(new, P.sal) >= msal) & (npj >= frac * best_proj)]
                    have = {L.key(x) for x in lus}
                    new = np.array([x for x in new if L.key(x) not in have]).reshape(-1, L.ROSTER)
                    if len(new):
                        lus = np.vstack([lus, new])               # append: earlier indices stay valid
                        dupe, scA, scB = score_all(lus)
                        stk = stack_team(lus, pool)
                        ext = make_ext(lus)
                        tier_bands(ext, n)
                        mA = evaluate(scA, fld, fpA, c, dupe)
                inside = np.isin(lus, U).all(axis=1)
                outside = set(np.flatnonzero(~inside).tolist())
                ch = select(mA, lus, n, c.large, exclude=used | outside, initial=True, stack=stk, extra=ext,
                            quiet=widen < 2.4)
                if len(ch) == n and meets_min_count(ch, lus, n, True):
                    chosen = ch
                    break
                if len(ch) == n and widen == 2.4:
                    chosen = ch
                    print(f"[build] 0-or-{k0}+ couldn't be fully met; nearest portfolio used")
            print(f"[build] 0-or-{k0}+ pass: {int(inside.sum())} lineups from the portfolio's {len(U)} players "
                  f"({(datetime.now() - t1).seconds}s)")
        if k0 > 1 and len(chosen) == n and not meets_min_count(chosen, lus, n, True):
            floor_note = (floor_note + "; " if floor_note else "") + f"0-or-{k0}+ rule not fully met"
            print(f"[build] NOTE: the 0-or-{k0}+ rule couldn't be fully met in {c.name[:40]}")
        mB = evaluate(scB, fld, fpB, c, dupe)
        chosen, log = confirm_swaps(chosen, mA, mB, lus, c.large, exclude=used, initial=True, stack=stk, extra=ext)
        used.update(chosen)
        picks[c.id] = [lus[i] for i in chosen]
        vB = per_sim_value(mB, chosen, c.large) if chosen else np.zeros(1)
        roi = mB["pay"][chosen].mean(axis=1) / max(c.fee, 1e-9) - 1 if (c.fee and chosen) else np.zeros(1)
        plan.append({"contest": c.name, "id": c.id, "entries": n, "field": c.size, "fee": c.fee,
                     "filled": len(chosen), "note": floor_note, "payouts": c.payout_source,
                     "objective": "top-1% (portfolio)" if c.large else "cash rate",
                     "p_any_top1": float(mB["top1"][chosen].any(axis=0).mean()) if chosen else 0.0,
                     "avg_top1": float(mB["top1"][chosen].mean()) if chosen else 0.0,
                     "cash": float(mB["cash"][chosen].mean()) if chosen else 0.0,
                     "roi_reported": float(np.mean(roi)), "value": float(vB.mean()),
                     "swaps": log, "stacks": pd.Series([stk[i] for i in chosen]).value_counts().to_dict()})
        for k, i in enumerate(chosen):
            rows.append({"Contest ID": c.id, "slot": k, "idx": int(i)})
    m0 = evaluate(scB, fld, fpB, contests[0], dupe) if contests else None
    table = _table(lus, pool, scB, m0, dupe, contests[0].size if contests else 0) if m0 else pd.DataFrame()
    chosen_rows = pd.DataFrame(rows)
    if len(table):
        table["Chosen"] = table.index.isin(chosen_rows["idx"]) if len(rows) else False
    players = _player_table(pool, lus, fpA, fld, chosen_rows)
    return BuildResult(picks, table, players, plan, pool)


def _table(lus, pool, sc, m, dupe, field_n) -> pd.DataFrame:
    P = L.Pool.from_df(pool)
    names = pool["Name"].to_numpy(object)
    return pd.DataFrame({
        "Lineup": [", ".join(names[L.assign_slots(lu, P)]) for lu in lus],
        "Stack": stack_team(lus, pool),
        "Salary": L.salary(lus, P.sal), "Proj": sc.mean(axis=1).round(1),
        "p90": np.percentile(sc, 90, axis=1).round(1),
        "Top1%": (m["top1"].mean(axis=1) * 100).round(2), "Cash%": (m["cash"].mean(axis=1) * 100).round(1),
        "AvgPay": m["pay"].mean(axis=1).round(3), "ExpDupes": (dupe * max(field_n - 1, 0)).round(2),
        "OwnSum": pool["Ownership"].to_numpy()[lus].sum(axis=1).round(1)})


def _player_table(pool, lus, fp, fld, chosen_rows) -> pd.DataFrame:
    n, S_ = len(pool), fp.shape[1]
    best = np.argmax(L.scores(lus, fp), axis=0)
    opt = np.bincount(lus[best].ravel(), minlength=n) / S_ * 100
    fl = np.bincount(fld.ravel(), minlength=n) / max(len(fld), 1) * 100
    out = pool[["Name", "pos", "team", "Salary", "Projection", "Ceiling", "Ownership", "HitPace%", "Gate"]].copy()
    out["Field%"] = fl.round(1)
    out["Optimal%"] = opt.round(1)
    out["Leverage"] = (out["Optimal%"] - out["Ownership"]).round(1)
    if len(chosen_rows):
        ch = lus[chosen_rows["idx"].values]
        out["My%"] = (np.bincount(ch.ravel(), minlength=n) / len(ch) * 100).round(0)
    return out.sort_values("Projection", ascending=False)


def ids_for(lu, pool) -> list[str]:
    """DK player IDs in slot order (FLEX = latest kickoff when possible)."""
    P = L.Pool.from_df(pool)
    return [str(pool["ID"].values[i]) for i in L.assign_slots(lu, P)]
