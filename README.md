# nflmain-dfs: DraftKings NFL Classic (Early / Main / Afternoon / Thanksgiving)

Projections, a game-first simulator, a simulated contest field, a stacking portfolio builder and
late swap for DraftKings NFL **Classic**. Free data only. Weekly process: `research/RUNBOOK.md`.
Rules: `CLAUDE.md`.

Classic roster: QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE), DST. $50,000 cap. Players lock at their own
game's kickoff.

## Use
```bash
pip install -r requirements.txt
python -m nflmain slates                         # DK Classic slates: type, games, locks, contests
python -m nflmain run --slate main               # projections -> output/<date>_main/
python -m research.prep --date 2026-11-15 --slate main   # Excel prep workbook
python -m nflmain build --entries-file DKEntries.csv     # every contest on its own slate
python -m nflmain news --entries-file output/DKEntries_current.csv   # news / inactives / late swap
python -m nflmain build --slate afternoon --n 20 # test build without an entries file
python -m pytest -q tests                        # offline tests
```

## How it works
**Data (free):**
- DraftKings: lobby, salary CSV, contests and payouts.
- ESPN: lines, injuries and live box scores.
- nflverse: usage (snaps, routes via targets/air yards, carries, red-zone share).
- National Weather Service: weather.
- No Odds API. Props are an off switch (`project.PROPS_ENABLED = False`).

**Projections** (`nflmain/project.py`, copied from the Showdown system and adapted):
- Usage history weighted toward recent games, shrunk to position defaults for players with few games.
- Usage is scaled to the Vegas implied team totals: implied totals move touchdowns *and* yards per
  play, and the spread moves the pass rate.
- Injuries: the more severe of DK's tag and ESPN's report. An OUT player's work goes to teammates,
  same position first.
- Weather: wind and rain trim passing.

**Game-first simulation** (`nflmain/gamesim.py`): every game on the slate is played out.
- Pace and efficiency lead to drives and TDs around the implied totals.
- The score margin sets the script.
- Players split their team's targets, carries, yards and TDs.
- This gives the correlations: QB with his pass-catchers, bring-backs, RB with a positive script,
  DST vs the opposing QB.

**Build** (`nflmain/pipeline.py`):
- Candidates come from several sources:
  - the best stacked lineup in hundreds of simulated slates;
  - structured QB stacks (1-3 pass-catchers, with and without a bring-back);
  - neighbors, and cap-aware passes.
- The rules come from the slate type: stack size, bring-back, ceiling gate, most players per game.
- A simulated field is built from projected ownership. It stacks like real players do.
- Every candidate is ranked against the field in every simulated slate and paid by the DK payout
  table. When DK refuses its payout API, the build uses the lobby's prize pool and an estimated
  top-heavy table.
- The portfolio is picked as a whole (MILP): top-1% coverage for large fields, cash for small.
  - 50% cap per player and per QB stack team.
  - 0-or-5+ on the initial build (100+ entries; 0-or-3 for 30-99; none under 30): candidates are rebuilt from the first portfolio's players so
    the rule can be met.
  - Swaps must be confirmed on an independent set of sims.

**Late swap** (`nflmain/lateswap.py`):
- Locked players stay in their DK slot.
- Open spots get refilled in many live-conditioned simulated slates, in parallel spawned processes.
- Options are judged by what they add to the portfolio's top-1% coverage.
- A swap is kept only if it's confirmed on independent sims. Entries holding a player who's out
  always change.
- FLEX holds the latest-kickoff player when possible, so late swap has room.

**Ceiling gate** (NBA rule):
- Winning pace = the best projected lineup's points per $1k + 0.1.
- The bottom share of players at each position, by how often they reach that pace, are kept out
  of our lineups.
- Share by slate type: Main 15%, Early 10%, Afternoon 5%, Thanksgiving 0%. These start from the
  NBA backtest by slate size. The NFL backtest is `research/gates/`.

## Outputs
Outputs go to `output/<date>_<slate>/`:
- `projections.csv`, `slate.json`, `stacks.csv`
- `sim_lineups.csv`, `sim_players.csv`

Upload files go to `output/DKEntries_upload.csv` (build) and `output/DKEntries_lateswap.csv`
(news). A uniquely named copy goes to `output/send/`.
