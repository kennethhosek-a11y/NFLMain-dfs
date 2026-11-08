# Handoff: what this system is

Repo: `kennethhosek-a11y/NFLMain-dfs`. Built 11/07/2026 in the "NFL Main · setup" chat, mirroring
the NBA system (NBA-dfs) and reusing the NFL Showdown system's data, projection and simulator code
(copied, never changed there).

| Step | Command | What happens |
|---|---|---|
| Slates | `nflmain slates` | Every DK Classic draft group: type (early/main/afternoon/thanksgiving/other), games, locks |
| Projections | `nflmain run --slate main` | DK salaries + tags, ESPN lines/injuries, NWS weather, nflverse usage -> game sims -> projections, ceiling gate, ownership |
| Build | `nflmain build --entries-file F` | Each contest -> its slate; candidates -> field -> payouts -> portfolio (50% cap, 0-or-5+, stack-team cap) -> swaps confirmed on independent sims |
| News / late swap | `nflmain news --entries-file output/DKEntries_current.csv` | Refresh, report what moved, re-evaluate every entry vs every unlocked player (live-conditioned) |
| Workbook | `python -m research.prep --date D --slate S` | Excel prep (Games, Players, positions, Stacks, Injuries, Overrides) |
| Research in | `python -m research.prep --export wb.xlsx` | Workbook opinions -> `data/overrides/<date>.csv` |
| Gate research | `python -m research.gates.backtest --date D` | Gate cut vs slate size |

## Interfaces
- `output/<date>_<slate>/projections.csv`: DK columns plus Projection, Floor, Ceiling, Boom%,
  Value, vsPace, HitPace%, Gate, Ownership and the usage shares.
- DKEntries files in and out. The latest file sent (`output/DKEntries_current.csv`) is the source
  of truth.
- Research overrides: `data/overrides/<date>.csv` (`Name, ProjectionAdj, Status, Ownership, Note, Slate`).

## Measured on 11/07/2026 (Week 9 slates, 4 CPUs)
- Main slate projections (11 games): 16 s.
- Builds:
  - 20 entries: ~2 min;
  - 150 entries: ~2 min (0-or-5+ met, 50% cap);
  - Afternoon slate, 20 entries: ~1.5 min.
- Late swap, 150 entries, with the 1:00 games locked: 75-78 s.
  - With no news: 3 entries changed.
  - With a 4:25 star ruled OUT: all 40 entries holding him changed.
- DK's payout API refuses some requests from cloud machines. Then the lobby's prize pool and an
  estimated top-heavy table are used, and the plan says "payouts estimated".

## Known limits / next
1. **Real-data backtest** of gate, stacks and bring-backs by slate size (research/gates/README.md).
   Target: before Week 10.
2. **Ownership** is a starting model. Contest-result ingest and retraining (NBA `ingest`) still
   needs porting.
3. Sim ROI is optimistic (the sim treats projections as truth). It's reported only.
4. The field doesn't late swap, and ownership doesn't react to late news (NBA lesson: the field
   moves about 20% of the way). Both are approximations.
5. **Thanksgiving dry run by Sun Nov 22.** Three windows work through the same lock logic. It
   still needs a full rehearsal on the real Thursday salary file once DK posts it (around Nov 23).
6. No Odds API. Props are a switched-off hook.
