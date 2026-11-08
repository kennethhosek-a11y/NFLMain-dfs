"""Command line: python -m nflmain <command>.

  slates     NFL Classic slates in the DK lobby (type, games, locks, contests)
  contests   contests on one slate
  run        projections for one slate                        -> output/<date>_<slate>/
  build      full build from a DKEntries file (each contest on its own slate)
  news       refresh + re-evaluate every entry against every UNLOCKED player (late swap); also
             the inactives / T-20 / between-windows check
  entries    where each entry stands (live points, projected final, top-1% chance)
  results    save DK contest downloads for good (they vanish after 10 days) + the field's habits
  settings   what settings apply (--import <folder> from the settings page)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import contests as C
from . import lineups as L
from . import pipeline, settings, slate
from .pipeline import ET, OUT, STATE

SEND = OUT / "send"
CURRENT = OUT / "DKEntries_current.csv"


def _send_copy(src: Path, what: str) -> Path:
    SEND.mkdir(parents=True, exist_ok=True)
    dst = SEND / f"DKEntries_{datetime.now(ET):%m%d_%H%M}_{what}.csv"
    k = 2
    while dst.exists():
        dst = SEND / f"DKEntries_{datetime.now(ET):%m%d_%H%M}_{what}_{k}.csv"
        k += 1
    shutil.copy(src, dst)
    shutil.copy(src, CURRENT)
    return dst


def cmd_slates(a):
    gs = slate.classic_groups(day=a.date)
    if not gs:
        print("No NFL Classic slates in the DK lobby" + (f" for {a.date}" if a.date else ""))
        return
    print(f"{'draft group':>11}  {'date':10}  {'first lock':>10}  {'games':>5}  {'type':12}  {'contests':>8}  name")
    for g in gs:
        print(f"{g['draft_group']:>11}  {g['date']}  {g['start'][11:16]:>10}  {g['games']:>5}  {g['slate_type']:12}  "
              f"{g['contests']:>8}  {g['suffix']}{' *featured' if g['featured'] else ''}")
    slate.pin(gs[0]["date"], gs)


def cmd_contests(a):
    g = slate.find_group(a.date or settings.slate_date().isoformat(), a.slate, a.draft_group)
    df = slate.contests_for(g["draft_group"])
    print(df.head(a.top).to_string(index=False))


def cmd_run(a):
    pipeline.run(a.slate, a.date, a.draft_group, a.salaries, n_sims=a.sims)


# ------------------------------------------------------------------ build

def _groups_for(entries: pd.DataFrame, day: str) -> dict[int, list[str]]:
    cids = list(dict.fromkeys(entries["Contest ID"]))
    m = slate.contest_groups(cids, day)
    miss = [c for c in cids if c not in m]
    if miss:
        names = entries.drop_duplicates("Contest ID").set_index("Contest ID").loc[miss, "Contest Name"]
        raise SystemExit("Couldn't find the DK slate for: " + "; ".join(names) + " - ask Kenny which slate these are")
    out: dict[int, list[str]] = {}
    for c in cids:
        out.setdefault(m[c], []).append(c)
    return out


def _group_info(dg: int, day: str) -> dict:
    pinned = slate.pins(day)["groups"].get(str(dg))
    if pinned:
        return pinned
    try:
        for g in slate.classic_groups():
            if g["draft_group"] == dg:
                slate.pin(day, [g])
                return g
    except Exception:  # noqa: BLE001
        pass
    return {"draft_group": dg, "slate_type": "main", "suffix": ""}


def cmd_build(a):
    day = a.date or settings.slate_date().isoformat()
    t0 = time.time()
    if a.entries_file:
        entries = C.read_entries(a.entries_file)
    else:                                   # test build: one contest, n blank entries
        g = slate.find_group(day, a.slate, a.draft_group)
        cid = a.contest or slate.contests_for(g["draft_group"]).iloc[0]["Contest ID"]
        entries = pd.DataFrame({"Entry ID": [str(9000000000 + k) for k in range(a.n)], "Contest Name": "test",
                                "Contest ID": str(cid), "Entry Fee": "$0"})
        for col in C.SLOT_COLS:
            entries[col] = ""
        slate.pin(day, contests={str(cid): g["draft_group"]})
    groups = _groups_for(entries, day)
    data = pipeline.load_data(day)
    picks, plans, state = {}, [], {"day": day, "slates": {}}
    for dg, cids in groups.items():
        gi = _group_info(dg, day)
        if gi.get("slate_type") == "other":
            print(f"[build] skipping {gi.get('suffix')} (not an Early/Main/Afternoon/Thanksgiving slate)")
            continue
        p = pipeline.run(day=day, draft_group=dg, data=data, quiet=False)
        sub = entries[entries["Contest ID"].isin(cids)]
        cons = C.load_contests(sub)
        print(f"\n[build] {p.label} slate: {len(sub)} entries in {len(cons)} contest(s)")
        res = pipeline.build(p, cons, seed=a.seed)
        d = pipeline.outdir(day, p.label)
        res.table.to_csv(d / "sim_lineups.csv", index=False)
        res.players.to_csv(d / "sim_players.csv", index=False)
        for c in cons:
            ids = [pipeline.ids_for(lu, res.pool) for lu in res.picks[c.id]]
            for eid, lu in zip(c.entry_ids, ids):
                picks[str(eid)] = lu
        plans += [dict(pl, slate=p.label) for pl in res.plan]
        p.df.to_csv(STATE / f"baseline_{day}_{p.label}.csv", index=False)
        state["slates"][str(dg)] = {"label": p.label, "type": p.slate_type, "contests": cids}
        _print_plan(res, p)
    up = OUT / "DKEntries_upload.csv"
    C.write_entries(entries, picks, up)
    (STATE / f"build_{day}.json").write_text(json.dumps({**state, "plans": plans}, indent=1, default=str))
    if a.entries_file:
        dst = _send_copy(up, "build")
        print(f"\nUpload file: {dst}  ({len(picks)} entries, {time.time() - t0:.0f}s)")
    else:
        print(f"\nTest build written to {up} ({time.time() - t0:.0f}s)")


def _print_plan(res, p):
    for pl in res.plan:
        print(f"  {pl['contest'][:50]}: {pl['filled']}/{pl['entries']} entries, field {pl['field']:,}, "
              f"payouts {pl['payouts']} | P(any top 1%) {pl['p_any_top1']:.1%}, avg top-1% {pl['avg_top1']:.2%}, "
              f"cash {pl['cash']:.0%}, sim ROI {pl['roi_reported']:+.0%} (reported only)")
        print(f"    QB stacks: {pl['stacks']}" + (f" | {pl['note']}" if pl['note'] else ""))
        acc = [s for s in pl["swaps"] if s["accepted"]]
        print(f"    swaps confirmed on independent sims: {len(acc)} of {len(pl['swaps'])} tried")
    pt = res.players
    if "My%" in pt:
        show = pt[pt["My%"] > 0].sort_values("My%", ascending=False)
        print(show[["Name", "pos", "team", "Salary", "Projection", "Ownership", "Optimal%", "My%"]].head(25)
              .round(1).to_string(index=False))


# ------------------------------------------------------------------ news / late swap

def _indices(entries: pd.DataFrame, pool: pd.DataFrame) -> list:
    by = {str(i): k for k, i in enumerate(pool["ID"].astype(str))}
    out = []
    for lu in C.entry_ids_in(entries):
        idx = [by.get(x) if x else None for x in lu]
        out.append(idx if any(x is not None for x in idx) else None)
    return out


def cmd_news(a):
    from . import lateswap, live, overrides
    day = a.date or settings.slate_date().isoformat()
    t0 = time.time()
    path = a.entries_file or CURRENT
    entries = C.read_entries(path)
    if a.out or a.active or a.q:
        _quick_overrides(day, a)
    groups = _groups_for(entries, day)
    data = pipeline.load_data(day)
    try:
        lv = live.live_points(datetime.fromisoformat(day).date())
    except Exception as e:  # noqa: BLE001
        print(f"[live] unavailable ({e})")
        lv = None
    picks, changes = {}, []
    for dg, cids in groups.items():
        gi = _group_info(dg, day)
        p = pipeline.run(day=day, draft_group=dg, data=data, quiet=True, save=True)
        _diff(p, day)
        sub = entries[entries["Contest ID"].isin(cids)]
        cons = {c.id: c for c in C.load_contests(sub)}
        # pool: everyone playable plus anyone already in an entry (locked or not)
        df = p.df.reset_index(drop=True)
        in_entries = set(x for lu in C.entry_ids_in(sub) for x in lu if x)
        keep = (df["Projection"] > 0.05) | df["ID"].astype(str).isin(in_entries)
        pool = df[keep].reset_index(drop=True)
        now = datetime.now(ET) if not a.now else datetime.fromisoformat(a.now).replace(tzinfo=ET)
        locked = lateswap.locked_mask(pool, now)
        live_fn = (lambda fp: live.condition(fp, pool, lv, locked)) if (lv is not None and len(lv) and locked.any()) else None
        n = int(settings.get("build.sims"))
        fpA = pipeline.sims_for(p, pool, n, a.seed, live_fn)
        fpB = pipeline.sims_for(p, pool, n, a.seed + 1, live_fn)
        rng = np.random.default_rng(a.seed)
        P = L.Pool.from_df(pool)
        fld = L.field(int(settings.get("build.field_sample")), pool["Ownership"].to_numpy(float), P, rng,
                      float(settings.get("ownership.field_stack")), float(settings.get("ownership.field_double")),
                      float(settings.get("ownership.field_bringback")), int(settings.get("ownership.field_min_salary")))
        dead = (pool["Projection"].to_numpy() <= 0.05) | pool["status"].isin(["OUT"]).to_numpy()
        allow = ~pool["Gate"].to_numpy(bool)
        currents = _indices(sub, pool)
        rules = L.Rules.for_slate(p.slate_type)
        res = lateswap.reevaluate(pool, fpA, fpB, currents, sub["Contest ID"].tolist(), cons, fld, allow, rules,
                                  dead, now, seed=a.seed)
        windows = slate.windows(pool)
        nxt = [w for w in windows if datetime.fromisoformat(w) > now]
        print(f"\n[{p.label}] {int(res['locked'].sum())} players locked; next lock "
              f"{nxt[0][11:16] + ' ET' if nxt else 'none (all games started)'}")
        for (eid, lu_new), lu_old in zip(zip(sub["Entry ID"], res["lineups"]), currents):
            if lu_new is not None:
                picks[str(eid)] = [str(pool["ID"].values[i]) for i in lu_new]
        for ch in res["log"]:
            lean = "into chalk" if ch["own_in"] > ch["own_out"] + 3 else "away from chalk" if ch["own_in"] < ch["own_out"] - 3 else "even"
            why = "must change (player out)" if ch["forced"] else f"gain {ch['gain']:+.3f} confirmed"
            changes.append(f"{p.label}: entry {sub['Entry ID'].iloc[ch['entry']]}: OUT {', '.join(ch['out'])} -> IN "
                           f"{', '.join(ch['in'])} ({why}; own {ch['own_out']:.0f}% -> {ch['own_in']:.0f}%, {lean})")
        _standings(pool, fpB, res["lineups"], sub, cons, fld, lv)
    print(f"\n{len(changes)} entr{'y' if len(changes) == 1 else 'ies'} changed:")
    for c in changes:
        print("  " + c)
    up = OUT / "DKEntries_lateswap.csv"
    C.write_entries(entries, picks, up)
    if changes:
        dst = _send_copy(up, a.what or "lateswap")
        print(f"\nUpload file: {dst}  ({time.time() - t0:.0f}s)")
    else:
        print(f"\nNo changes - keep the current file. ({time.time() - t0:.0f}s)")


def _quick_overrides(day, a):
    """--out / --q / --active names: written into today's overrides file (news Kenny heard first)."""
    from . import overrides
    f = overrides.path_for(day)
    f.parent.mkdir(parents=True, exist_ok=True)
    cur = pd.read_csv(f, dtype=str, keep_default_na=False) if f.exists() else pd.DataFrame(columns=overrides.COLS)
    for names, st in ((a.out, "OUT"), (a.q, "Q"), (a.active, "ACTIVE")):
        for n in [x.strip() for x in (names or "").split(",") if x.strip()]:
            cur = cur[cur["Name"] != n]
            cur = pd.concat([cur, pd.DataFrame([{"Name": n, "Status": st, "Note": f"news {datetime.now(ET):%H:%M}"}])])
    cur.to_csv(f, index=False)


def _diff(p, day):
    f = STATE / f"baseline_{day}_{p.label}.csv"
    if not f.exists():
        return
    b = pd.read_csv(f, dtype={"ID": str}).set_index("ID")
    d = p.df.assign(ID=p.df["ID"].astype(str)).set_index("ID")
    common = d.index.intersection(b.index)
    mv_pts, mv_pct = float(settings.get("news.move_pts")), float(settings.get("news.move_pct"))
    lines = []
    for i in common:
        old, new = float(b.at[i, "Projection"]), float(d.at[i, "Projection"])
        so, sn = str(b.at[i, "status"]) if pd.notna(b.at[i, "status"]) else "", str(d.at[i, "status"])
        if so != sn or (abs(new - old) >= mv_pts and abs(new - old) >= mv_pct * max(old, 1)):
            lines.append(f"{d.at[i, 'Name']} ({d.at[i, 'team']} {d.at[i, 'pos']}): {so or 'active'} -> {sn or 'active'}, "
                         f"proj {old:.1f} -> {new:.1f}")
    print(f"[news] {p.label}: {len(lines)} material change(s) since the build" + (":" if lines else ""))
    for x in lines[:40]:
        print("   " + x)


def _standings(pool, fp, lineups, sub, cons, fld, lv):
    """Each entry: points so far, projected final, chance of a top-1% finish (live-conditioned)."""
    from .contests import evaluate, field_dupe_rate
    have = [(k, lu) for k, lu in enumerate(lineups) if lu is not None]
    if not have:
        return
    lus = np.array([sorted(lu) for _, lu in have])
    sc = L.scores(lus, fp)
    so_far = np.zeros(len(have))
    if lv is not None and len(lv):
        pts = dict(zip(lv["name_key"] + "|" + lv["team"], lv["dk_pts"]))
        k = (pool["name_key"] + "|" + pool["team"]).to_numpy(object)
        so_far = np.array([sum(pts.get(k[i], 0.0) for i in lu) for lu in lus])
    rows = []
    cid = sub["Contest ID"].to_numpy()
    for c in cons.values():
        sel = [j for j, (k, _) in enumerate(have) if cid[k] == c.id]
        if not sel:
            continue
        m = evaluate(sc[sel], fld, fp, c, np.zeros(len(sel)))
        for t, j in enumerate(sel):
            rows.append({"entry": sub["Entry ID"].iloc[have[j][0]], "contest": c.name[:30], "so_far": round(so_far[j], 1),
                         "proj_final": round(float(sc[j].mean()), 1), "top1%": round(float(m["top1"][t].mean() * 100), 2),
                         "cash%": round(float(m["cash"][t].mean() * 100), 1)})
    if rows:
        df = pd.DataFrame(rows).sort_values("top1%", ascending=False)
        print("\nWhere entries stand (live-conditioned sims):")
        print(df.head(30).to_string(index=False))
        if len(df) > 30:
            print(f"  ... {len(df) - 30} more; median top-1% chance {df['top1%'].median():.2f}%")


def cmd_results(a):
    """Save DK contest downloads for good and measure the field's habits (ownership rec. #1)."""
    from . import results
    day = a.date or settings.slate_date().isoformat()
    sal = results.saved_salaries(day)
    if sal is None:
        print(f"[results] no saved DK salary file for {day}: stacks and salary used can't be measured")
    for f in a.files:
        h = results.save(f, day, sal, a.name or "")
        print(f"[results] {Path(f).name}: contest {h['contest_id']}, {h.get('entries', 0):,} entries saved "
              f"-> data/contests/{day}/")
        for k in ("qb_stacked", "double_stack", "triple_stack", "bring_back", "flex_RB", "flex_WR", "flex_TE",
                  "avg_salary", "share_duplicated", "most_copies"):
            if k in h:
                print(f"    {k}: {h[k]}")
        print(f"    top owned: {h.get('top_owned', '')}")


def cmd_settings(a):
    if a.import_path:
        eff = settings.import_dashboard(settings.payload_from(Path(a.import_path)))
        print(settings.plain(eff))
        return
    if a.json:
        print(settings.dumps(settings.report()))
        return
    print(settings.plain())
    if a.all:
        for k, s in settings.schema().items():
            print(f"  {k:32s} {settings.get(k)!s:>8}  ({s['label']})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nflmain", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="slate date YYYY-MM-DD (default: today ET)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slates")
    s.set_defaults(fn=cmd_slates)
    s = sub.add_parser("contests")
    s.add_argument("--slate", default="main")
    s.add_argument("--draft-group", type=int)
    s.add_argument("--top", type=int, default=25)
    s.set_defaults(fn=cmd_contests)
    s = sub.add_parser("run")
    s.add_argument("--slate", default="main")
    s.add_argument("--draft-group", type=int)
    s.add_argument("--salaries")
    s.add_argument("--sims", type=int)
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("build")
    s.add_argument("--entries-file")
    s.add_argument("--slate", default="main", help="test builds without an entries file")
    s.add_argument("--draft-group", type=int)
    s.add_argument("--contest")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=101)
    s.add_argument("--sims-build", type=int)
    s.set_defaults(fn=cmd_build)
    for name in ("news", "lateswap"):
        s = sub.add_parser(name)
        s.add_argument("--entries-file")
        s.add_argument("--out", help="names to mark OUT (comma separated)")
        s.add_argument("--q", help="names to mark Questionable")
        s.add_argument("--active", help="names to mark active")
        s.add_argument("--seed", type=int, default=211)
        s.add_argument("--now", help="pretend it's this ET time (testing), e.g. 2026-11-08T16:00")
        s.add_argument("--what", help="label in the upload file name (e.g. inactives, t20)")
        s.add_argument("--sims-build", type=int)
        s.set_defaults(fn=cmd_news)
    s = sub.add_parser("results", help="save DK contest downloads (full lineups + %Drafted)")
    s.add_argument("files", nargs="+")
    s.add_argument("--name", help="contest name (optional)")
    s.set_defaults(fn=cmd_results)
    s = sub.add_parser("settings")
    s.add_argument("--import", dest="import_path")
    s.add_argument("--json", action="store_true")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_settings)
    a = ap.parse_args(argv)
    day = a.date or settings.slate_date().isoformat()
    settings.use(day)
    if getattr(a, "sims_build", None):
        settings.override("build.sims", a.sims_build)
    if a.cmd != "settings":
        print(settings.summary_line())
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
