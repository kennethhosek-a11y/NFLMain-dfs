# Ceiling gate by slate size

Mirrors the NBA gate study (research/gate in NBA-dfs: a flat 25% cut threw out 15-22% of the
players in each night's actual best lineup; the cut was then scaled by slate size).

## First pass (11/07/2026, model sims as truth, Week 9 Main slate games)
Share of each held-out simulated slate's best-lineup players that each cut would throw out:

| games | 5% cut | 10% | 15% | 25% |
|---|---|---|---|---|
| 2  | 6.9 | 7.6 | 8.4 | 11.7 |
| 3  | 2.9 | 3.5 | 5.5 | 9.9 |
| 4  | 2.8 | 3.6 | 6.4 | 8.3 |
| 7  | 1.6 | 3.8 | 4.5 | 7.4 |
| 11 | 0.6 | 2.5 | 3.6 | 5.8 |

Same shape as the NBA finding: the smaller the slate, the more a cut hurts. Starting settings:
Main (11 games) 15%, Early (7) 10%, Afternoon (3-4) 5%, Thanksgiving (3) 0%.

## Next (before Week 10, Nov 15): real data
These sims trust the model, so they understate the risk. The real backtest:
- data: actual DK points by week from nflverse, plus salaries from Kenny's Showdown database
  (406 games, 2025 W1 to 2026 W8);
- projections rebuilt with only the weeks before each slate;
- slates of each size;
- score: share of each week's real best lineup thrown out, and the top-1% rate of gated vs
  ungated portfolios.
The same harness also tunes stack size and bring-back by slate type.
