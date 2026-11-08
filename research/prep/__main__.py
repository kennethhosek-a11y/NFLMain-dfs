"""python -m research.prep --date YYYY-MM-DD --slate main | --export WORKBOOK.xlsx"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import workbook

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="research.prep", description="NFL Main prep workbook (Excel)")
    ap.add_argument("--date", help="slate date YYYY-MM-DD (Eastern), default today")
    ap.add_argument("--slate", default="main", help="main / early / afternoon / ... (default main)")
    ap.add_argument("--xlsx", help="output path (default: research/prep/out/NFLMain_Prep_<date>_<slate>_<MMDD-HHMM>.xlsx)")
    ap.add_argument("--export", metavar="WORKBOOK", help="write data/overrides/<date>.csv from a filled-in workbook")
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.export:
        dest, rows = workbook.export_overrides(Path(a.export), root=root)
        print(f"Wrote {len(rows)} override row(s) to {dest}")
        for r in rows:
            print("  " + ", ".join(f"{k}={v}" for k, v in r.items() if v not in ("", None)))
        return
    day = a.date or workbook.today_et()
    dest = workbook.build(day, a.slate, root=root, dest=Path(a.xlsx) if a.xlsx else None)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
