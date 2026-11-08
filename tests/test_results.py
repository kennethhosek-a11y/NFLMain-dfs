"""Offline test: a DK contest download is saved and the field's habits are measured."""
from __future__ import annotations

import zipfile

import pandas as pd

from nflmain import results


def _download(tmp_path):
    rows = ["Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS"]
    lu1 = "DST Jets FLEX Bo Wr QB Al Qb RB Cy Rb RB Di Rb TE Ed Te WR Fa Wr WR Gi Wr WR Ho Opp"
    lu2 = "DST Jets FLEX Iv Rb QB Al Qb RB Cy Rb RB Di Rb TE Ed Te WR Fa Wr WR Gi Wr WR Ho Opp"
    right = [("Al Qb", "QB", "20.5%", "25"), ("Bo Wr", "WR", "10%", "12"), ("Cy Rb", "RB", "30%", "18"),
             ("Di Rb", "RB", "15%", "9"), ("Ed Te", "TE", "8%", "7"), ("Fa Wr", "WR", "12%", "20"),
             ("Gi Wr", "WR", "5%", "3"), ("Ho Opp", "WR", "9%", "11"), ("Iv Rb", "RB", "2%", "1"),
             ("Jets ", "DST", "11%", "6")]
    for k, lu in enumerate([lu1, lu1, lu2]):
        r = right[k] if k < len(right) else ("", "", "", "")
        rows.append(f"{k + 1},{100 + k},kenny (1/3),0,150,{lu},,{','.join(r)}")
    for r in right[3:]:
        rows.append(f",,,,,,,{','.join(r)}")
    f = tmp_path / "contest-standings-197411850.csv"
    f.write_text("\n".join(rows) + "\n")
    z = tmp_path / "contest-standings-197411850.zip"
    with zipfile.ZipFile(z, "w") as zz:
        zz.write(f, f.name)
    return z


def test_parse_lineup():
    lu = results.parse_lineup("DST Jets FLEX Bo Wr QB Al Qb RB Cy Rb RB Di Rb TE Ed Te WR Fa Wr WR Gi Wr WR Ho Opp")
    assert len(lu) == 9 and lu[0] == ("DST", "Jets") and lu[2] == ("QB", "Al Qb")


def test_save_download(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "CONTESTS", tmp_path / "contests")
    sal = pd.DataFrame({"Name": ["Al Qb", "Bo Wr", "Cy Rb", "Di Rb", "Ed Te", "Fa Wr", "Gi Wr", "Ho Opp", "Iv Rb", "Jets"],
                        "name_key": ["al qb", "bo wr", "cy rb", "di rb", "ed te", "fa wr", "gi wr", "ho opp", "iv rb", "nyj dst"],
                        "team": ["BUF"] * 3 + ["MIA"] * 2 + ["BUF", "BUF", "KC", "MIA", "NYJ"],
                        "game": ["KC@BUF"] * 3 + ["NYJ@MIA"] * 2 + ["KC@BUF"] * 3 + ["NYJ@MIA"] * 2,
                        "salary": [8000, 6000, 7000, 5000, 4000, 6000, 3500, 4500, 3000, 3000]})
    h = results.save(_download(tmp_path), "2026-11-08", sal, "Play-Action")
    assert h["contest_id"] == "197411850" and h["entries"] == 3 and h["lineups_parsed"] == 3
    assert h["most_copies"] == 2 and h["unique_lineups"] == 2
    # BUF QB with Bo/Fa/Gi (3 catchers) twice and Fa/Gi (2) once; Ho Opp (KC) is the bring-back
    assert abs(h["triple_stack"] - 2 / 3) < 1e-3 and abs(h["double_stack"] - 1 / 3) < 1e-3 and h["bring_back"] == 1.0
    assert abs(h["flex_WR"] - 2 / 3) < 1e-3 and abs(h["flex_RB"] - 1 / 3) < 1e-3
    assert (tmp_path / "contests" / "2026-11-08" / "197411850.csv.gz").exists()
    assert (tmp_path / "contests" / "field_habits.csv").exists()
