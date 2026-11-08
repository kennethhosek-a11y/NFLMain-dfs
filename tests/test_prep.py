"""Prep workbook: builds from a synthetic slate, and the Overrides sheet exports to the overrides file."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from research.prep import workbook as W

DAY = "2026-11-08"


def _fixture(root):
    rng = np.random.default_rng(0)
    games = [("NYJ", "KC", "2026-11-08T13:00:00-05:00"), ("ARI", "SEA", "2026-11-08T16:25:00-05:00")]
    rows = []
    for away, home, kick in games:
        for team, opp in ((away, home), (home, away)):
            for pos, n in (("QB", 2), ("RB", 2), ("WR", 4), ("TE", 1), ("DST", 1)):
                for k in range(n):
                    proj = 0.0 if (pos == "QB" and k == 1) else float(rng.uniform(4, 24))
                    sal = int(rng.integers(30, 90)) * 100
                    rows.append({"Name": f"{team} {pos}{k}", "ID": len(rows) + 1000, "pos": pos, "team": team,
                                 "opp": opp, "Salary": sal, "kickoff": kick,
                                 "status": "Q" if k == 1 and pos == "WR" else "", "dk_status": "",
                                 "espn_status": "", "injury_note": "hamstring" if k == 1 and pos == "WR" else "",
                                 "Projection": proj, "Floor": proj * 0.6, "Ceiling": proj * 1.8, "Boom%": 10,
                                 "Value": proj / sal * 1000, "Ownership": rng.uniform(0, 30), "ProjSource": "model",
                                 "ResearchAdj": 0, "ResearchNote": "", "games": 8, "snap_pct": 0.7,
                                 "tgt_share": 0.15, "car_share": 0.1, "rz_tgt_share": 0.1, "rz_car_share": 0.1,
                                 "ay_share": 0.1})
    d = root / "output" / f"{DAY}_main"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "projections.csv", index=False)
    meta = {"day": DAY, "slate": "main", "slate_type": "main", "draft_group": 1,
            "games": [{"away": a, "home": h, "kickoff": k, "total": 45.5, "spread_home": -3.0, "line_source": "ESPN",
                       "weather": {"wind": 8, "precip": 10, "forecast": "Sunny", "indoor": False},
                       "teams": {a: {"implied": 21.25, "sim_mean": 21.0, "sim_p10": 10, "sim_p90": 33, "notes": []},
                                 h: {"implied": 24.25, "sim_mean": 24.1, "sim_p10": 13, "sim_p90": 37,
                                     "notes": ["wind"]}}}
                      for a, h, k in games], "built": "x"}
    (d / "slate.json").write_text(json.dumps(meta))


def test_build_and_export(tmp_path):
    _fixture(tmp_path)
    p = W.build(DAY, "main", root=tmp_path, run_if_missing=False, built_at=datetime(2026, 11, 8, 9, 5))
    assert p.name == "NFLMain_Prep_2026-11-08_main_1108-0905.xlsx"
    wb = load_workbook(p)
    for s in ("Read Me", "Games", "Players", "QB", "RB", "WR", "TE", "DST", "Stacks", "Injuries", "Overrides"):
        assert s in wb.sheetnames
    assert wb["Stacks"].max_row > 1 and wb["Games"].max_row == 5
    ws = wb["Overrides"]
    name = ws.cell(2, 1).value
    ws.cell(2, 7, -3)
    ws.cell(2, 8, "out")
    ws.cell(2, 10, "beat writer says limited")
    wb.save(p)
    # an existing row for another player is kept; a row for this player is replaced
    od = tmp_path / "data" / "overrides"
    od.mkdir(parents=True)
    pd.DataFrame([{"Name": "Someone Else", "ProjectionAdj": "1", "Status": "", "Ownership": "", "Note": "keep"},
                  {"Name": name, "ProjectionAdj": "5", "Status": "", "Ownership": "", "Note": "old"}]
                 ).to_csv(od / f"{DAY}.csv", index=False)
    dest, rows = W.export_overrides(p, root=tmp_path)
    assert dest == od / f"{DAY}.csv" and len(rows) == 1
    out = pd.read_csv(dest, dtype=str, keep_default_na=False)
    assert set(out["Name"]) == {"Someone Else", name}
    r = out[out["Name"] == name].iloc[0]
    assert float(r["ProjectionAdj"]) == -3 and r["Status"] == "OUT" and r["Note"] == "beat writer says limited"


def test_compute_stacks():
    df = pd.DataFrame([
        {"Name": "Q", "pos": "QB", "team": "A", "opp": "B", "Salary": 6000, "Projection": 20, "Ceiling": 30, "Ownership": 10},
        {"Name": "W1", "pos": "WR", "team": "A", "opp": "B", "Salary": 7000, "Projection": 15, "Ceiling": 28, "Ownership": 20},
        {"Name": "T1", "pos": "TE", "team": "A", "opp": "B", "Salary": 4000, "Projection": 9, "Ceiling": 18, "Ownership": 5},
        {"Name": "W9", "pos": "WR", "team": "A", "opp": "B", "Salary": 3000, "Projection": 4, "Ceiling": 9, "Ownership": 1},
        {"Name": "BB", "pos": "WR", "team": "B", "opp": "A", "Salary": 6500, "Projection": 14, "Ceiling": 26, "Ownership": 12},
    ])
    st = W.compute_stacks(df)
    r = st.iloc[0]
    assert r["Pass-catchers"] == "W1, T1" and r["Bring-back"] == "BB"
    assert r["Projection"] == 58 and r["Salary"] == 23500
