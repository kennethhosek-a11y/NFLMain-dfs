# Workbook rules backtest (11/08/2026)

`python -m research.backtest.rules`. Each rule from Kenny's lineup-rules workbook
(`research/rules/`) graded Moderate or Opinion was tested on its own, against a baseline with only
the hard rules (QB + 1-2 pass-catchers, no DST facing our QB, $49,500 floor).
- **Data:** every real slate 2021 to 2026 Week 8: 294 slates, of which 99 are short (Afternoon /
  Thanksgiving). 2026 weeks count 3x, 2025 weeks 1.5x.
- **Score:** 20-lineup portfolios scored on what really happened (best lineup as a % of the real
  optimal).
- **Verdict:**
  - helps = better by more than 2 standard errors;
  - hurts = worse by more than 2 SE (overall or in 2026);
  - held up = no measurable harm.

| Rule | Evidence | Real best lineup follows it | Portfolio effect (pts of % of optimal, ±SE) | Verdict | Setting now |
|---|---|---|---|---|---|
| M4 ~50% of lineups with 1 bring-back | Moderate | 35% | +0.41 ± 0.18 | **helps** | on (50% ± 10) |
| M6 at most 1 hitter vs own DST | Opinion | 95% | +0.08 ± 0.11 | held up | on |
| M7 QB + own RB only if a receiver | Moderate | 85% | +0.15 ± 0.17 | held up | on |
| M8 max 1 RB per team | Opinion | 96% | +0.33 ± 0.10 | **helps** | on |
| M8 max 2 from a team without its QB | Opinion | 91% | +0.38 ± 0.17 | **helps** | on |
| M8 max 4 per team | Opinion | 100% | +0.06 ± 0.04 | held up | on |
| M9 (test: stacks only from the 2 highest totals) | Moderate | 43% | +0.31 ± 0.27 | not confirmed: restricting didn't hurt | not a setting (sims choose) |
| M11 TE in FLEX at most 10% | Moderate | 86% | +0.39 ± 0.17 | **helps** | on |
| M15 QB max 25%, DST max 15% | Opinion | - | **-1.48 ± 0.25** | **hurts** | **off (50%) until Kenny decides** |
| M16 lineups differ by 2+ | Opinion | - | 0.00 | held up | on (2) |
| M16 lineups differ by 3+ | Opinion | - | +0.12 ± 0.05 | helps (small) | 2 (Kenny's call) |
| SS1+SS2 short slates: 5 per team, 2 RBs, QB+RB ok | Opinion | 40% use one | -0.39 ± 0.47 | held up | on |
| SS3 short slates: 2 hitters vs own DST | Opinion | 8% | -0.21 ± 0.37 | held up (no gain) | off (M6's 1 kept) |
| SS4 short slates: $48,500 floor | Opinion | 70% | +0.43 ± 0.52 | held up | **not used**: Kenny's hard $49,500 wins |
| SS5 short slates: exposure up to 60% | Opinion | - | **+0.89 ± 0.26** | **helps** | 50% (Kenny's own rule; his call) |

Not testable without ownership history (paid, not bought): M12 (cheap low-owned TE/DST), M13
(ownership band 75-150%), M14's soft parts, M4's chalk credit, SS6. They stay as written, and
Kenny's contest downloads (research/ownership #1) will let us test them.

Caveats:
- The portfolio test has no simulated field, so it rewards raw scoring, not beating the field. A
  rule that only pays off against a field (e.g. lower QB exposure for leverage) can look worse
  here than it is.
- Salaries are synthetic for 2022-2026 W8.
- Effects are small: tenths of a point of percent-of-optimal, except M15 and SS5.

## Kenny's 11/08 list, item 7: QB with up to 3 of his own pass-catchers (`research/backtest/stack3.py`)
Same method as above (294 real slates 2021-2026, 2026 weighted x3, 2025 x1.5; 20 lineups from 30
simulated slates; scored on what really happened). Every other hard rule stays on.

| | Best lineup, % of optimal | 3rd-best | 2026 only (23 slates) |
|---|---|---|---|
| Up to 3 allowed vs max 2 | +0.05 ± 0.08 | +0.03 ± 0.05 | -0.11 / +0.17 |

- No measurable difference. When allowed, triples were picked in only 3% of lineups.
- The real best lineup had QB + 3 pass-catchers on only 2% of slates.
- **Verdict: keep the hard max of 2** (workbook M2). This test can't see ownership; triples are
  rarely played by the field, but they also rarely win.

## Item 6: QB 25% / DST 15% caps
Already tested above as M15: **-1.48 ± 0.25** points of % of optimal (hurts). The test can't see
ownership, so the leverage the caps aim for isn't measured. Kenny's standing setting keeps them on
(data/settings/standing.yaml); his call.
