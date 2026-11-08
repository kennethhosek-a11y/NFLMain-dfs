"""Excel prep workbook for one NFL Main (DK Classic) slate.

Reads what the build writes to output/<date>_<slate>/ (projections.csv, slate.json and, if
present, stacks.csv) and writes research/prep/out/NFLMain_Prep_<date>_<slate>_<MMDD-HHMM>.xlsx
(a new name every build). Sheets: Read Me, Games, Players, QB, RB, WR, TE, DST, Stacks,
Injuries, Overrides.

Kenny types opinions into the yellow columns of the Overrides sheet; then
  python -m research.prep --export <workbook>
merges them into data/overrides/<date>.csv (Name, ProjectionAdj, Status, Ownership, Note).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter as CL
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
OV_COLS = ["Name", "ProjectionAdj", "Status", "Ownership", "Note"]
STATUSES = ["OUT", "Q", "D", "ACTIVE", "START"]

F = Font(name="Arial", size=10)
FB = Font(name="Arial", size=10, bold=True)
F_HEAD = Font(name="Arial", size=10, bold=True, color="FFFFFF")
F_TITLE = Font(name="Arial", size=16, bold=True, color="1F3864")
FILL_HEAD = PatternFill("solid", fgColor="1F3864")
FILL_IN = PatternFill("solid", fgColor="FFF2CC")
WRAP = Alignment(wrap_text=True, vertical="top")
HEAD_AL = Alignment(wrap_text=True, vertical="center", horizontal="center")

# (column, header, width, number format)
PLAYER_COLS = [
    ("Name", "Player", 22, None), ("pos", "Pos", 5, None), ("team", "Team", 6, None), ("opp", "Opp", 6, None),
    ("kickoff", "Kickoff (ET)", 9, None), ("Salary", "Salary", 8, "$#,##0"), ("status", "Status", 7, None),
    ("Projection", "Projection", 9, "0.0"), ("Floor", "Floor (25th)", 8, "0.0"),
    ("Ceiling", "Ceiling (90th)", 8, "0.0"), ("Boom%", "Boom %", 7, "0"), ("Value", "Value (pts/$1k)", 8, "0.00"),
    ("Ownership", "Proj. own %", 8, "0.0"), ("Leverage", "Leverage", 8, "0.0"), ("HitPace%", "Hits winning pace %", 9, "0.0"),
    ("Gate", "Fails ceiling gate", 8, None),
    ("snap_pct", "Snap %", 7, "0%"), ("tgt_share", "Target share", 7, "0%"), ("car_share", "Carry share", 7, "0%"),
    ("rz_tgt_share", "Red-zone target share", 8, "0%"), ("rz_car_share", "Red-zone carry share", 8, "0%"),
    ("ay_share", "Air-yards share", 7, "0%"), ("games", "Games", 6, "0"),
    ("ProjSource", "Source", 12, None), ("ResearchAdj", "Research adj", 8, "0.0"),
    ("ResearchNote", "Research note", 30, None), ("injury_note", "Injury note", 40, None),
]


def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ inputs

def load_inputs(day: str, slate: str, root: Path = ROOT, run_if_missing: bool = True):
    d = root / "output" / f"{day}_{slate}"
    if not (d / "projections.csv").exists() and run_if_missing:
        from nflmain import pipeline  # imported only when needed (tests use fixtures)
        pipeline.run(slate=slate, day=day)
    if not (d / "projections.csv").exists():
        raise FileNotFoundError(f"{d / 'projections.csv'} not found - run the projections first")
    proj = pd.read_csv(d / "projections.csv")
    meta = json.loads((d / "slate.json").read_text()) if (d / "slate.json").exists() else {}
    stacks = pd.read_csv(d / "stacks.csv") if (d / "stacks.csv").exists() else None
    return proj, meta, stacks


def compute_stacks(proj: pd.DataFrame) -> pd.DataFrame:
    """Each QB + his top 2 pass-catchers + the opponent's top pass-catcher (bring-back)."""
    p = proj[proj["Projection"] > 0]
    pc = p[p["pos"].isin(["WR", "TE"])].sort_values("Projection", ascending=False)
    own = p["Ownership"] if "Ownership" in p else pd.Series(0.0, index=p.index)
    rows = []
    for i, q in p[p["pos"] == "QB"].sort_values("Projection", ascending=False).iterrows():
        mates = pc[pc["team"] == q["team"]].head(2)
        bb = pc[pc["team"] == q["opp"]].head(1)
        grp = pd.concat([p.loc[[i]], mates, bb])
        rows.append({
            "QB": q["Name"], "Team": q["team"], "Opp": q["opp"],
            "Pass-catchers": ", ".join(mates["Name"]), "Bring-back": ", ".join(bb["Name"]),
            "Salary": int(grp["Salary"].sum()), "Projection": float(grp["Projection"].sum()),
            "Ceiling": float(grp["Ceiling"].sum()) if "Ceiling" in grp else None,
            "Own % (sum)": float(own.reindex(grp.index).fillna(0).sum()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ writing helpers

def _header(ws, row, headers, widths=None):
    for j, h in enumerate(headers, 1):
        c = ws.cell(row, j, h)
        c.font, c.fill, c.alignment = F_HEAD, FILL_HEAD, HEAD_AL
        if widths:
            ws.column_dimensions[CL(j)].width = widths[j - 1]
    ws.row_dimensions[row].height = 30


def _clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def _kick(s) -> str:
    try:
        return datetime.fromisoformat(str(s)).strftime("%a %I:%M %p").replace(" 0", " ")
    except ValueError:
        return str(s or "")


def _table(ws, df: pd.DataFrame, cols, start_row=1):
    cols = [c for c in cols if c[0] in df.columns]
    _header(ws, start_row, [c[1] for c in cols], [c[2] for c in cols])
    for i, (_, r) in enumerate(df.iterrows(), start_row + 1):
        for j, (key, _, _, fmt) in enumerate(cols, 1):
            v = r[key]
            if key == "kickoff":
                v = _kick(v)
            elif key == "Gate":
                v = "yes" if str(v).lower() in ("true", "1", "yes") else ""
            c = ws.cell(i, j, _clean(v))
            c.font = F
            if fmt:
                c.number_format = fmt
    last = start_row + len(df)
    ws.freeze_panes = ws.cell(start_row + 1, 2)
    if len(cols):
        ws.auto_filter.ref = f"A{start_row}:{CL(len(cols))}{max(last, start_row)}"
    for key in ("Value", "Leverage"):
        if len(df) and key in [c[0] for c in cols]:
            L = CL([c[0] for c in cols].index(key) + 1)
            ws.conditional_formatting.add(
                f"{L}{start_row + 1}:{L}{last}",
                ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile", mid_value=50,
                               mid_color="FFEB84", end_type="max", end_color="63BE7B"))
    return cols


# ------------------------------------------------------------------ build

def build(day: str, slate: str, root: Path = ROOT, dest: Path | None = None, run_if_missing: bool = True,
          built_at: datetime | None = None) -> Path:
    proj, meta, stacks = load_inputs(day, slate, root, run_if_missing)
    built_at = built_at or datetime.now(ET)
    dest = dest or (root / "research" / "prep" / "out" /
                    f"NFLMain_Prep_{day}_{slate}_{built_at:%m%d-%H%M}.xlsx")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proj = proj.sort_values("Projection", ascending=False).reset_index(drop=True)
    live = proj[proj["Projection"] > 0]

    wb = Workbook()
    _readme(wb.active, day, slate, meta, built_at)
    _games(wb.create_sheet("Games"), meta)
    _table(wb.create_sheet("Players"), live, PLAYER_COLS)
    for pos, n in (("QB", 40), ("RB", 60), ("WR", 80), ("TE", 40), ("DST", 32)):
        _table(wb.create_sheet(pos), live[live["pos"] == pos].head(n), PLAYER_COLS)
    _stacks(wb.create_sheet("Stacks"), stacks if stacks is not None and len(stacks) else compute_stacks(proj))
    _injuries(wb.create_sheet("Injuries"), proj)
    _overrides(wb.create_sheet("Overrides"), live)
    wb.save(dest)
    return dest


def _readme(ws, day, slate, meta, built_at):
    ws.title = "Read Me"
    ws["A1"] = "NFL Main prep workbook"
    ws["A1"].font = F_TITLE
    rows = [("Slate date", day), ("Slate", slate), ("Slate type", meta.get("slate_type", slate)),
            ("Games", len(meta.get("games", []))), ("DK draft group", meta.get("draft_group", "")),
            ("Built (ET)", built_at.strftime("%a %b %d, %Y %I:%M %p"))]
    for i, (k, v) in enumerate(rows, 3):
        ws.cell(i, 1, k).font = FB
        ws.cell(i, 2, _clean(v)).font = F
    ws.cell(10, 1, "How to use it").font = FB
    text = ("Every sheet is information except Overrides. On the Overrides sheet, type only where you have an "
            "opinion, in the yellow columns: 'Proj adj' adds or subtracts DraftKings points from the model's "
            "projection (e.g. -3); 'Status' marks a player OUT, Q (questionable), D (doubtful), ACTIVE (clears an "
            "injury tag) or START (this QB starts); 'My own %' replaces the projected ownership; 'Note' is your "
            "reason. Leave everything else blank. Send the workbook back in the chat and it becomes the research "
            "overrides file for the next build. Projections, ownership and the lineups are all re-simulated from "
            "it, so stacks and game scripts stay consistent.")
    c = ws.cell(11, 1, text)
    c.font, c.alignment = F, WRAP
    ws.merge_cells("A11:H18")
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 28


def _games(ws, meta):
    heads = ["Game", "Kickoff (ET)", "Total", "Home spread", "Team", "Implied total", "Sim average",
             "Sim 10th pct", "Sim 90th pct", "Wind (mph)", "Rain %", "Forecast", "Line source", "Notes"]
    widths = [12, 11, 7, 8, 7, 9, 9, 9, 9, 8, 7, 18, 30, 40]
    _header(ws, 1, heads, widths)
    r = 2
    for g in sorted(meta.get("games", []), key=lambda g: str(g.get("kickoff", ""))):
        wx = g.get("weather") or {}
        for t in (g.get("away"), g.get("home")):
            tm = (g.get("teams") or {}).get(t, {})
            vals = [f"{g.get('away')}@{g.get('home')}", _kick(g.get("kickoff")), g.get("total"), g.get("spread_home"),
                    t, tm.get("implied"), tm.get("sim_mean"), tm.get("sim_p10"), tm.get("sim_p90"),
                    "indoors" if wx.get("indoor") else wx.get("wind"), wx.get("precip"), wx.get("forecast"),
                    g.get("line_source"), "; ".join(tm.get("notes") or [])]
            for j, v in enumerate(vals, 1):
                c = ws.cell(r, j, _clean(v))
                c.font = F
                if j in (6, 7, 8, 9):
                    c.number_format = "0.0"
            r += 1
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{CL(len(heads))}{max(r - 1, 1)}"


def _stacks(ws, st: pd.DataFrame):
    cols = [(c, c, 24 if c in ("QB", "Pass-catchers", "Bring-back", "partners") else 10,
             "0.0" if c in ("Projection", "Proj", "Ceiling", "Own % (sum)", "Own") else
             ("$#,##0" if c == "Salary" else ("0.00" if c == "corr" else None))) for c in st.columns]
    _table(ws, st, cols)


def _injuries(ws, proj: pd.DataFrame):
    m = pd.Series(False, index=proj.index)
    for c in ("status", "dk_status", "espn_status", "injury_note"):
        if c in proj:
            m |= proj[c].fillna("").astype(str).str.strip().ne("")
    cols = [("Name", "Player", 22, None), ("pos", "Pos", 5, None), ("team", "Team", 6, None),
            ("kickoff", "Kickoff (ET)", 9, None), ("Salary", "Salary", 8, "$#,##0"),
            ("status", "Status used", 8, None), ("dk_status", "DK tag", 7, None), ("espn_status", "ESPN", 12, None),
            ("Projection", "Projection", 9, "0.0"), ("injury_note", "Injury note", 60, None)]
    _table(ws, proj[m].sort_values("Salary", ascending=False), cols)


def _overrides(ws, live: pd.DataFrame):
    heads = ["Player", "Team", "Pos", "Salary", "Model projection", "Model own %",
             "Proj adj (+/- pts)", "Status (OUT/Q/D/ACTIVE/START)", "My own %", "Note"]
    widths = [22, 6, 5, 8, 10, 9, 10, 12, 9, 40]
    _header(ws, 1, heads, widths)
    for i, (_, r) in enumerate(live.iterrows(), 2):
        vals = [r["Name"], r["team"], r["pos"], r["Salary"], r["Projection"], r.get("Ownership")]
        for j, v in enumerate(vals, 1):
            c = ws.cell(i, j, _clean(v))
            c.font = F
            c.number_format = {4: "$#,##0", 5: "0.0", 6: "0.0"}.get(j, "General")
        for j in range(7, 11):
            ws.cell(i, j).fill = FILL_IN
    last = max(len(live) + 1, 2)
    dv = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"H2:H{last}")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:J{last}"


# ------------------------------------------------------------------ export

def export_overrides(path: Path, root: Path = ROOT) -> tuple[Path, list[dict]]:
    wb = load_workbook(path, data_only=True)
    day = None
    rm = wb["Read Me"]
    for r in range(1, 20):
        if str(rm.cell(r, 1).value or "").strip() == "Slate date":
            day = str(rm.cell(r, 2).value or "").strip()[:10]
    if not day:
        raise ValueError(f"{path}: no slate date on the Read Me sheet")
    ws = wb["Overrides"]
    rows = []
    for r in range(2, ws.max_row + 1):
        name = ws.cell(r, 1).value
        adj, status, own, note = (ws.cell(r, j).value for j in (7, 8, 9, 10))
        if not name or all(v in (None, "") for v in (adj, status, own, note)):
            continue
        st = str(status or "").strip().upper()
        if st and st not in STATUSES:
            print(f"[export] {name}: unknown status {status!r} - ignored")
            st = ""
        rows.append({"Name": str(name).strip(), "ProjectionAdj": _num(adj), "Status": st,
                     "Ownership": _num(own), "Note": str(note or "").strip()})
    dest = root / "data" / "overrides" / f"{day}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=OV_COLS)
    if dest.exists():
        old = pd.read_csv(dest, dtype=str, keep_default_na=False)
        for c in OV_COLS:
            if c not in old:
                old[c] = ""
        old = old[~old["Name"].str.strip().str.lower().isin(new["Name"].str.lower())][OV_COLS]
        new = pd.concat([old, new.astype(str).replace("nan", "")], ignore_index=True)
    new.to_csv(dest, index=False)
    return dest, rows


def _num(v):
    if v in (None, ""):
        return ""
    try:
        return float(v)
    except (TypeError, ValueError):
        return ""
