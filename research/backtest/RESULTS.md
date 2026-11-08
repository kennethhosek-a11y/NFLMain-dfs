# Slate-size backtests: results (11/07/2026)

Data:
- every completed Sunday from 2021 to 2026 Week 8: 97 Early, 98 Main and 94 Afternoon slates;
- 5 Thanksgiving slates;
- 2026 weeks weighted 3x, 2025 weeks 1.5x.

Salaries are real for 2021 and synthetic for 2022-2026 (see research/history/README.md).

## 1. Ceiling gate (`gate`)
Share of each slate's real best-lineup players that a gate cut would have thrown out.

| | 10% cut | 15% cut | 25% cut |
|---|---|---|---|
| 2021, real salaries, 6+ games | 3.6% | 6.3% | 9.6% |
| 2021, real salaries, 2-5 games | 3.9% | 5.9% | 11.1% |
| synthetic salaries, all sizes | ~2% | ~3% | ~4.5% |

- Synthetic salaries flatter the gate: they carry no information DK's real pricing has. So the
  real-salary rows decide.
- The gate is about as informative as in NBA: the NBA's 25% cut lost 15-18%.
- In the 20-lineup portfolio test (section 3), gate cuts from 0% to 25% moved the best lineup by
  less than 1 point of percent-of-optimal. That's within noise.
- **Decision: keep Main 15%, Early 10%, Afternoon 5%, Thanksgiving 0%.**

## 2. How real best lineups are built (`stacks`)

| | Early (7.5 games) | Main (11.2) | Afternoon (3.7) | Thanksgiving (3) |
|---|---|---|---|---|
| QB + 1 pass-catcher | 54% | 58% | 59% | 64% |
| QB + 2 | 9% | 7% | 17% | 18% |
| QB alone | 36% | 35% | 19% | 18% |
| has a bring-back | 48% | 34% | 70% | 82% |
| hitter vs own DST | 18% | 13% | 52% | 64% |
| 5+ players from one game | 2% | 2% | 29% | 36% |
| pts given up forcing QB+1 | 1.2 | 1.6 | 3.6 | 4.1 |
| pts given up forcing QB+2 | 6.3 | 8.7 | 6.2 | 7.3 |

- **Decision: keep "QB + at least 1" everywhere.** It costs 1-2 points on big slates and buys
  correlation.
- QB+2 stays as an option the build tries, never a requirement.
- **Decision (applied): the own-defense rule is now off on Afternoon and Thanksgiving**
  (`no_vs_dst`).

## 3. 20-lineup portfolios scored on real results (`portfolio`)
Each portfolio was built from 60 correlated simulated outcomes, then scored by its best lineup as
a % of the real optimal.

| | Early | Main | Afternoon |
|---|---|---|---|
| default (QB+1, 50% cap, gate) | 67.5 | 63.6 | 74.7 |
| QB+1 + bring-back required | 68.8 | 63.1 | 73.8 |
| QB+2 required | 67.4 | 63.1 | 76.6 |
| no stack rule | 66.6 | 62.8 | 75.1 |
| exposure cap 30% | 66.4 | 63.4 | 73.4 |
| exposure cap 70% | 67.1 | 63.0 | 76.3 |

- Differences are 1-2 points, about the size of the noise (~95 slates each).
- What's consistent: on small (Afternoon) slates, more concentration and bigger stacks score a
  little higher. That matches section 2.
- But this test has no field and no ownership. Raw "best of 20" always favors concentration,
  while beating a field favors being different.
- **Decision: no change to the 50% cap or the stack rules yet.** Re-test once real 2026 salaries
  and contest results (ownership) have built up.

## Not tested yet
- Top-1% finish vs a real field. That needs ownership history, which is paid (FantasyLabs etc.).
  Not bought; it's Kenny's call. Free alternative: Kenny's own DK contest results, ingested weekly.
