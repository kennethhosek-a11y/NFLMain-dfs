# Ownership research (Kenny, 11/08/2026)

- `NFL_main_slate_ownership_projection.md`: the report on how providers project ownership,
  where models miss, and what our NFL Main system should do.
- `NFL_Ownership_Projection_Research_2026-11-08.xlsx`: the same as a workbook.
  - The **Recommendations** sheet (#1-#15) is our plan. Kenny's yellow Use? (Y/N) and Notes
    columns win.
  - The other sheets: Key Numbers, Field Habits, News Timing, Providers, Ownership Drivers,
    Sources.

## Status
| # | Recommendation | Status |
|---|---|---|
| 1 | Save every DK contest download (our contests + Milly Maker), every Sunday night | **In force**: `python -m nflmain results`. Files go in `data/contests/<date>/` and habits in `data/contests/field_habits.csv` |
| 2-15 | Position totals, softmax per position, public projections, recency, chalk sharpening, ... | Waiting for Kenny's Y/N. Weak items (#8 numbers, #14, #15, exact #5/#6 features) are starting guesses to fit from the downloads |

## Why #1 matters, and its limits
- The contest download has every entry's full lineup plus each player's %Drafted.
- It's the only source for the field's real stack rate, FLEX split, salary use and duplicates.
- DK keeps it for 10 days.
- It sits behind Kenny's login, so he downloads and uploads it; Claude never logs in for him.
- It exists only for contests he entered, so the Milly Maker needs at least one entry.
