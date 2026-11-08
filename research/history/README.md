# NFL history database (2021 to last week)

Built by `python -m research.history.build_db`. Refresh only 2026 with `--season 2026`; that's
part of every Tuesday's early look. Output goes to `data/research/history/` (gzip CSV, committed).
Downloads are cached in `data/cache/` (not committed, about 310 MB).

| Table | Rows (11/07/2026) | What |
|---|---|---|
| games.csv.gz | 1,631 games | kickoff, **slate tag** (morning / early / afternoon / night / thanksgiving / thu / mon / sat), closing spread + total, implied team totals, roof, weather, score |
| player_week.csv.gz | 32,603 | QB/RB/WR/TE per game: DK points, stats, targets, carries, air yards, red-zone looks, snap share, depth-chart rank, injury report + practice status, salary + source, real Showdown salary (2025-26), trailing DK average, team implied total |
| dst_week.csv.gz | 2,962 | defenses: DK points, sacks, takeaways, TDs, points allowed, salary |
| validation.csv | | our DK points vs RotoGuru's real ones; synthetic vs real salaries |
| dk_salaries/ | | real DK Classic salary files saved by every `nflmain run` (2026 Week 9 on) |

## Salaries: what RotoGuru actually has (checked 11/07/2026)
- **2020 and 2021 only.** Every week of both seasons has about 375-450 players with real DK
  salaries.
- **2022, 2023, 2024, 2025 and 2026 pages are empty.** RotoGuru stopped publishing DK salaries
  after 2021.
- HTTPS to RotoGuru is refused from this machine, but its plain-HTTP page works. The `nfldfs`
  PyPI package reads the same pages, so it can't fill the gap either.

So:
- 2021 uses **real** DK salaries (97% of player-games).
- 2022 through 2026 Week 8 use **synthetic DK-style salaries**, like the NBA gate backtest:
  - DK pricing of each player's trailing average plus his team's Vegas total, fitted on the real
    2021 salaries;
  - 8% pricing noise;
  - DK's ranges by position, rounded to $100.
  - Against the real 2021 salaries they correlate 0.73 (QB), 0.78 (RB), 0.79 (TE) and 0.83 (WR).
    The typical miss is $470-600.
- 2026 Week 9 onward uses **real** salaries. The weekly process saves DK's files itself.
- The Showdown database's real Showdown salaries (2025 W1 to 2026 W8) are attached for reference.
  Showdown pricing isn't Classic pricing.

## Checks
- Our DK points match RotoGuru's real DK points exactly for QB/RB/WR/TE (2021).
- Defenses are within 0.1 point 87.5% of the time. The small misses come from blocked kicks and
  return TDs.

## Not free (so not included)
- Routes run: PFF data. Targets per snap and air-yard share stand in for it.
- Ownership history: FantasyLabs and similar. It isn't bought; ask Kenny first if it looks worth
  it.
