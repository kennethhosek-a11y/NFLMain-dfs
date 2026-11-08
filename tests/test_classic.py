"""Offline tests: DK Classic rules, projections on a synthetic slate, stacking optimizer, field,
portfolio rules (50% cap, 0-or-5+), entries files, and late swap never moving locked players."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nflmain import contests as C
from nflmain import lateswap, pipeline, portfolio, scoring, slate
from nflmain import lineups as L
from nflmain import settings as SET

from . import synth


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("nflmain")
    inj = synth.patch_network(mp, tmp)
    SET.reset_cache()
    SET.use(synth.DAY)
    p = pipeline.run(day=synth.DAY, salaries=synth.salary_csv(), n_sims=3000, data=synth.data(), inj=inj,
                     quiet=True, save=False)
    yield p
    mp.undo()


def test_scoring_matches_dk_classic():
    assert scoring.offense(pass_yd=300, pass_td=2, ints=1) == pytest.approx(12 + 8 - 1 + 3)
    assert scoring.offense(rec=8, rec_yd=100, rec_td=1) == pytest.approx(8 + 10 + 6 + 3)
    assert scoring.offense(rush_yd=100, rush_td=1, fum_lost=1) == pytest.approx(10 + 6 - 1 + 3)
    assert scoring.dst(sacks=3, ints=1, fum_rec=1, pa=0) == 3 + 2 + 2 + 10
    assert scoring.dst(pa=6) == 7 and scoring.dst(pa=13) == 4 and scoring.dst(pa=20) == 1
    assert scoring.dst(pa=27) == 0 and scoring.dst(pa=34) == -1 and scoring.dst(pa=35) == -4


def test_salary_file_and_slate_types():
    pool = slate.load_classic(synth.salary_csv())
    assert len(pool) == 6 * 11 and set(pool["Position"]) == {"QB", "RB", "WR", "TE", "DST"}
    assert slate.slate_type_of(pool) == "main"
    assert len(slate.windows(pool)) == 2
    k = lambda s: [datetime.fromisoformat(x) for x in s]
    assert slate.classify("(Early Only)", k(["2026-11-08T13:00:00-05:00"])) == "early"
    assert slate.classify("(Afternoon Turbo)", k(["2026-11-08T16:25:00-05:00"])) == "afternoon"
    assert slate.classify("(Primetime)", k(["2026-11-08T20:20:00-05:00"])) == "other"
    assert slate.classify(None, k(["2026-11-26T12:30:00-05:00", "2026-11-26T16:30:00-05:00",
                                   "2026-11-26T20:20:00-05:00"])) == "thanksgiving"


def test_projections_follow_vegas(proj):
    df = proj.df
    assert (df["Projection"] >= 0).all()
    tot = df.groupby("team")["Projection"].sum()
    assert tot["FFF"] > tot["EEE"]           # FFF is a 6.5-point home favorite in the highest total
    qb = df[(df["pos"] == "QB") & (df["Projection"] > 1)]
    assert len(qb) == 6                       # one starting QB per team
    assert df["Ownership"].between(0, 100).all()


def _check_rules(lu, pool, rules):
    P = L.Pool.from_df(pool)
    assert L.violations(lu, P, rules) == []
    q = lu[P.pos[lu] == "QB"][0]
    mates = sum(1 for i in lu if P.team[i] == P.team[q] and P.pos[i] in ("WR", "TE"))
    assert 1 <= mates <= 2                                          # hard M1 / M2
    d = lu[P.pos[lu] == "DST"][0]
    assert P.opp[q] != P.team[d]                                    # hard M5
    assert P.sal[lu].sum() >= L.HARD_MIN_SALARY                     # hard M10
    assert sum(1 for i in lu if P.opp[i] == P.team[d] and P.pos[i] != "DST") <= rules.max_vs_own_dst


def test_hard_rules_cannot_be_loosened():
    r = L.Rules(stack_min=0, stack_max=3, min_salary=40000)
    assert r.stack_min == 1 and r.stack_max == 2 and r.min_salary == 49500


def test_optimizer_stacks_and_rules(proj):
    pool = proj.df.reset_index(drop=True)
    rules = L.Rules.for_slate("main")
    opt = L.Optimizer(L.Pool.from_df(pool), rules, (pool["Projection"] > 0.3).to_numpy())
    pts = pool["Projection"].to_numpy()
    lu = opt.solve(pts)
    _check_rules(lu, pool, rules)
    lu2 = opt.solve(pts, team="AAA", qb_mates=2, bringback=1)
    P = L.Pool.from_df(pool)
    assert sum(1 for i in lu2 if P.team[i] == "AAA" and P.pos[i] in ("WR", "TE")) == 2
    assert sum(1 for i in lu2 if P.team[i] == "BBB" and P.pos[i] in ("RB", "WR", "TE")) == 1
    slots = L.assign_slots(lu, P)
    assert [P.pos[i] for i in slots[:7]] == ["QB", "RB", "RB", "WR", "WR", "WR", "TE"] and P.pos[slots[8]] == "DST"
    flex_ok = [i for i in lu if P.pos[i] == P.pos[slots[7]]]
    assert P.kickoff[slots[7]] == max(P.kickoff[i] for i in flex_ok)   # FLEX = latest kickoff


def test_field_lineups_are_legal(proj):
    pool = proj.df[proj.df["Projection"] > 0.05].reset_index(drop=True)
    P = L.Pool.from_df(pool)
    fld = L.field(500, pool["Ownership"].to_numpy(), P, np.random.default_rng(1), min_salary=40000)
    assert len(fld) >= 450
    assert all(L.valid(lu, P) for lu in fld[:200])


def test_build_portfolio_rules(proj):
    SET.override("build.sims", 1500)
    SET.override("build.worlds", 120)
    SET.override("build.field_sample", 3000)
    SET.override("build.workers", 1)
    # the synthetic slate has only 6 teams: the workbook's QB/DST caps and shape shares can't all
    # hold there together with 0-or-5+ (a real 22-team slate can); test the core rules here
    SET.override("rules.qb_max_exposure", 1.0)
    SET.override("rules.dst_max_exposure", 1.0)
    SET.override("rules.share_band", 1.0)
    for k, v in (("main.rb_per_team", 2), ("main.qb_own_rb", 2), ("rules.no_qb_team_max", 8), ("main.max_per_team", 5),
                 ("rules.max_rb_runback", 3), ("rules.flex_rb_max", 1.0),
                 # 0-or-5 is off under 30 entries since 11/08; test the full rule on 12 entries here
                 ("portfolio.min_count_off_below", 10), ("portfolio.min_count_full_entries", 10)):
        SET.override(k, v)
    c = C.Contest("1", "test", 1.0, 5000, entry_ids=[str(i) for i in range(12)], prize_pool=4000)
    res = pipeline.build(proj, [c])
    lus = np.array(res.picks["1"])
    assert len(lus) == 12
    rules = L.Rules.for_slate("main")
    for lu in lus:
        _check_rules(lu, res.pool, rules)
    cnt = np.bincount(lus.ravel())
    assert cnt.max() <= portfolio.caps(12)                  # 50% cap
    for k in ("rules.qb_max_exposure", "rules.dst_max_exposure", "rules.share_band", "main.rb_per_team",
              "main.qb_own_rb", "rules.no_qb_team_max", "main.max_per_team", "rules.max_rb_runback",
              "rules.flex_rb_max", "portfolio.min_count_off_below", "portfolio.min_count_full_entries"):
        SET._overrides.pop(k, None)
    assert len({L.key(x) for x in lus}) == 12               # no lineup twice
    used = cnt[cnt > 0]
    assert (used >= 5).all() or "0-or-" in str(res.plan)    # 0-or-5+ (or reported step-down)


def test_entries_roundtrip(tmp_path, proj):
    pool = proj.df.reset_index(drop=True)
    f = tmp_path / "DKEntries.csv"
    head = "Entry ID,Contest Name,Contest ID,Entry Fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DST,,Instructions"
    f.write_text(head + "\n111,Test,9,$1,,,,,,,,,,,\n222,Test,9,$1,,,,,,,,,,,\n")
    e = C.read_entries(f)
    assert list(e["Entry ID"]) == ["111", "222"]
    ids = [str(x) for x in pool["ID"].iloc[:9]]
    out = tmp_path / "up.csv"
    C.write_entries(e, {"111": ids}, out)
    back = C.read_entries(out)
    assert C.entry_ids_in(back)[0] == ids and C.entry_ids_in(back)[1] == [None] * 9


def test_lateswap_never_moves_locked_players(proj):
    pool = proj.df[proj.df["Projection"] > 0.05].reset_index(drop=True)
    P = L.Pool.from_df(pool)
    rules = L.Rules.for_slate("main")
    opt = L.Optimizer(P, rules, np.ones(len(pool)))
    rng = np.random.default_rng(4)
    fpA = pipeline.sims_for(proj, pool, 1500, 5)
    fpB = pipeline.sims_for(proj, pool, 1500, 6)
    cur = []
    for k in range(6):
        lu = opt.solve(fpA[:, k])
        cur.append(L.assign_slots(lu, P))
    now = datetime.fromisoformat(f"{synth.DAY}T14:00:00-05:00")   # 1:00 games locked, 4:25 open
    locked = lateswap.locked_mask(pool, now)
    assert locked.any() and not locked.all()
    fld = L.field(2000, pool["Ownership"].to_numpy(), P, rng, min_salary=40000)
    c = C.Contest("9", "t", 1.0, 3000, entry_ids=[str(i) for i in range(6)], prize_pool=2500)
    dead = np.zeros(len(pool), bool)
    # mark one unlocked player in entry 0 as OUT: entry 0 must change, and only unlocked spots move
    open_in_0 = [i for i in cur[0] if not locked[i]]
    dead[open_in_0[0]] = True
    SET.override("lateswap.options", 10)
    res = lateswap.reevaluate(pool, fpA, fpB, cur, ["9"] * 6, {"9": c}, fld, np.ones(len(pool), bool),
                              rules, dead, now)
    for old, new in zip(cur, res["lineups"]):
        for s in range(9):
            if locked[old[s]]:
                assert new[s] == old[s]             # locked player kept, same slot
        assert P.sal[new].sum() <= slate.CAP
        assert all(new[s] in [i for i in range(len(pool)) if P.pos[i] in slate.SLOT_POS[s]] for s in range(9))
    assert open_in_0[0] not in res["lineups"][0]


def test_workbook_portfolio_rules():
    """QB / DST exposure caps, shape shares and 2+ different players, on made-up candidates."""
    rng = np.random.default_rng(3)
    pos = np.array(["QB"] * 8 + ["DST"] * 8 + ["WR"] * 40, dtype=object)
    lus = np.array([np.sort(np.r_[rng.choice(8), 8 + rng.choice(8), 16 + rng.choice(40, 7, replace=False)])
                    for _ in range(400)])
    m = {"top1": rng.random((400, 300)) < 0.05, "cash": rng.random((400, 300)) < 0.2,
         "beat": rng.random((400, 300)), "pay": rng.random((400, 300))}
    props = {"double": rng.random(400) < 0.5, "bringback": rng.random(400) < 0.5, "te_flex": rng.random(400) < 0.1}
    ext = portfolio.Extra(pos, props, {"double": (0.4, 0.6), "bringback": (0.4, 0.6), "te_flex": (0.0, 0.1)},
                          {"QB": 0.25, "DST": 0.15}, min_unique=2)
    ch = portfolio.joint(m, lus, 40, True, extra=ext, time_limit=30)
    assert len(ch) == 40
    cnt = np.bincount(lus[ch].ravel(), minlength=56)
    assert cnt[:8].max() <= 10 and cnt[8:16].max() <= 6          # QB 25%, DST 15% of 40
    assert 16 <= props["double"][ch].sum() <= 24 and props["te_flex"][ch].sum() <= 4


def test_run_back_and_flex_rules():
    """Run-backs count opposing WR/TE only; an RB in FLEX is tagged; opposing RBs can be ruled out."""
    pos = np.array(["QB", "WR", "WR", "RB", "RB", "RB", "TE", "WR", "DST"], dtype=object)
    team = np.array(["A", "A", "B", "B", "C", "C", "C", "D", "D"], dtype=object)
    opp = np.array(["B", "B", "A", "A", "E", "E", "E", "F", "F"], dtype=object)
    P = L.Pool(np.full(9, 5500.0), pos, team, opp, np.array(["A@B"] * 3 + ["X"] * 6, dtype=object),
               np.array([""] * 9, dtype=object))
    lu = np.arange(9)
    sh = L.shapes(lu[None, :], P)
    assert sh["bringback"][0] and sh["rb_flex"][0]      # WR from B is the run-back; 3 RBs = RB in FLEX
    assert "RB run-back" in L.violations(lu, P, L.Rules(max_rb_runback=0, max_per_team=8, rb_per_team=3,
                                                        no_qb_team_max=8, qb_own_rb=2, min_salary=0, hard=False))


def test_exposure_exception_needs_a_note(tmp_path):
    """A cheap player can go above the 30% cap only with MaxExposure AND a Note arguing mispricing."""
    from nflmain import overrides
    pool = pd.DataFrame({"Name": ["Cheap Guy", "Other Guy"]})
    f = tmp_path / "2026-11-08.csv"
    f.write_text("Name,ProjectionAdj,Status,Ownership,Note,Slate,MaxExposure\n"
                 "Cheap Guy,,,,Starter now; DK priced him as a backup,,45\nOther Guy,,,,,,0.45\n")
    ov = overrides.load("2026-11-08", directory=tmp_path, quiet=True)
    assert overrides.exposure_exceptions(pool, ov, quiet=True) == {0: 0.45}
    pos = np.array(["WR", "WR", "QB"], dtype=object)
    ext = portfolio.Extra(pos, {}, {}, {"QB": 0.25}, sal=np.array([4000.0, 4000.0, 6000.0]), cheap=(4500, 0.30))
    ext.exceptions = {0: 0.45}
    capv = ext.player_caps(20, 1, 3)
    assert capv[0] == min(9, portfolio.caps(20)) and capv[1] == 6   # 45% (never above the overall cap); plain cheap 30%


def test_rule_updates_1108():
    """Kenny 11/08: 0-or-5 by contest size, leverage caps, bust-aware cheap caps."""
    import numpy as np
    from nflmain.portfolio import Extra, min_count
    assert min_count(20) == 1 and min_count(29) == 1
    assert min_count(30) == 3 and min_count(99) == 3
    assert min_count(150) == 5 and min_count(150, initial=False) == 1
    pos = np.array(["WR", "WR", "WR", "QB"], object)
    sal = np.array([7000, 4000, 4000, 6000])
    own = np.array([40.0, 5.0, 5.0, 2.0])
    bust = np.array([0.3, 0.40, 0.70, 0.3])
    e = Extra.from_settings(pos, {}, sal, own=own, bust=bust)
    cap = e.player_caps(150, 1, 4)
    assert cap[0] == 75                      # 2 x 40% + 10 = 90% -> held to the 50% cap
    assert cap[1] == 30                      # cheap, low bust: leverage 20% (cheap cap 30%)
    assert cap[2] == 22                      # cheap, high bust: 15% cap (22 of 150)
    assert cap[3] == 21                      # 2 x 2% + 10 = 14%
    e.exceptions = {2: 0.25}
    assert e.player_caps(150, 1, 4)[2] == 37  # an argued exception still wins


def test_prelock_news_check_changes_only_news_entries(proj):
    """Kenny 11/08: before any game starts, a news check changes only entries holding a player with
    news (or a player who is out) - no churn from re-running the sims."""
    pool = proj.df[proj.df["Projection"] > 0.05].reset_index(drop=True)
    P = L.Pool.from_df(pool)
    rules = L.Rules.for_slate("main")
    opt = L.Optimizer(P, rules, np.ones(len(pool)))
    rng = np.random.default_rng(4)
    fpA = pipeline.sims_for(proj, pool, 1500, 5)
    fpB = pipeline.sims_for(proj, pool, 1500, 6)
    cur = [L.assign_slots(opt.solve(fpA[:, k]), P) for k in range(6)]
    now = datetime.fromisoformat(f"{synth.DAY}T09:00:00-05:00")   # nothing locked
    assert not lateswap.locked_mask(pool, now).any()
    fld = L.field(2000, pool["Ownership"].to_numpy(), P, rng, min_salary=40000)
    c = C.Contest("9", "t", 1.0, 3000, entry_ids=[str(i) for i in range(6)], prize_pool=2500)
    dead = np.zeros(len(pool), bool)
    dead[cur[0][1]] = True                                        # entry 0 holds a player now out
    news = np.zeros(len(pool), bool)
    news[cur[0][1]] = True
    others = {i for k in range(1, 6) for i in cur[k]}
    SET.override("lateswap.options", 10)
    res = lateswap.reevaluate(pool, fpA, fpB, cur, ["9"] * 6, {"9": c}, fld, np.ones(len(pool), bool),
                              rules, dead, now, news=news)
    assert cur[0][1] not in res["lineups"][0]                     # the out player is replaced
    for k in range(1, 6):
        if cur[0][1] not in cur[k]:
            assert sorted(res["lineups"][k]) == sorted(cur[k])    # untouched entries stay as they are
def test_started_games_keep_their_game_info(tmp_path, monkeypatch):
    """Found live 11/08/2026: once the 1:00 games start, DK's salary file shows 'In-Progress'
    instead of the game - it must be filled back in (else every late swap crashes)."""
    monkeypatch.setattr(slate, "STATE_DIR", tmp_path)
    head = "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
    rows = ["QB,A (1),A,1,QB,6000,{g1},DAL,20", "WR,B (2),B,2,WR/FLEX,5000,{g2},IND,12",
            "WR,C (3),C,3,WR/FLEX,4000,SEA@ARI 11/08/2026 04:25PM ET,SEA,10"]
    before = head + "\n".join(rows).format(g1="DAL@IND 11/08/2026 01:00PM ET", g2="DAL@IND 11/08/2026 01:00PM ET")
    after = head + "\n".join(rows).format(g1="In-Progress", g2="In-Progress")
    slate.load_classic(before)                               # seen before kickoff: remembered
    p = slate.load_classic(after)
    assert list(p["game"]) == ["DAL@IND", "DAL@IND", "SEA@ARI"]
    assert p["kickoff"].iloc[0].startswith("2026-11-08T13:00")
