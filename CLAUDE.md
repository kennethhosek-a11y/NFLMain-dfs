# Rules for any Claude session working in this repo

This repo (`nflmain-dfs`) is Kenny's **DraftKings NFL Classic** system: Early, Main and Afternoon
Sunday slates and the Thanksgiving slate. Kenny is the owner. He is not a programmer, so use plain
language. **All times are Eastern.**

**Follow `research/RUNBOOK.md`** for the weekly schedule and for how to talk to Kenny.

## Chats
- **Setup chat** ("NFL Main · setup"): built the system (11/07/2026).
- **Weekly chats**: one per NFL week, titled `NFL Main · Week N · Sun Mon D`
  (e.g. `NFL Main · Week 10 · Sun Nov 15`). It runs Tuesday's early look through Sunday's last
  late swap. Thanksgiving week's chat also runs the Thursday slate.
- A weekly chat may run every `nflmain` command and `research.prep`, and commit and push **data
  only** (`data/history/`, `data/state/`, `data/overrides/`, `data/settings/`, `data/contests/`,
  `data/contests_meta/`, `data/research/history/`) to the main branch.
- **History database** (`data/research/history/`, built by `research/history/build_db.py`): it
  covers 2021 through the last completed week, and Tuesday's early look adds each new week.
  Ownership history is paid (FantasyLabs etc.): **never buy data**; ask Kenny first if it seems
  worth it.

## Only the latest NFL Main chat makes changes (owner rule)
Code, data, CLAUDE.md, the runbook, branches, pull requests and NFL Main scheduled tasks change
**only in the most recent NFL Main chat**. Any older chat, including this setup chat once a weekly
chat exists, is read-only. It may answer questions and read files, but it must not commit, push,
merge, open pull requests, or create, change or delete scheduled tasks. If asked to change
something, it tells Kenny to ask in the latest NFL Main chat.

## Code changes: branches + pull requests Kenny merges
- Code changes go on a branch (`research/<topic>`) with a pull request. **Kenny merges.** Never
  push code directly to the main branch.
- Run `python -m pytest -q tests` before opening any pull request. **All tests must pass.**
- **Never force-push and never rewrite history.** That means no `rebase` of pushed commits, no
  `reset --hard` of pushed commits, and no `commit --amend` of pushed commits. Never delete
  branches or tags you didn't create in this chat.
- Research code and workbooks go under `research/`, not `nflmain/`.
- Research reaches the build only through the overrides file `data/overrides/<YYYY-MM-DD>.csv`.
  - Columns: `Name, ProjectionAdj, Status, Ownership, Note, Slate`. Leave a column empty when
    there's no opinion.
  - `Status` can be OUT, Q, D, ACTIVE, or START (this QB starts).
  - `Slate` is optional (early / main / afternoon / thanksgiving). Leave it blank for every slate.
  - Don't edit `output/` files or the model code to inject research.
- Every build setting, its default and its safe range live in `config/settings.yaml`. Changing a
  default is a code change. Kenny's "from now on" choices go in `data/settings/standing.yaml` and
  his "today only" choices in `data/settings/<date>.yaml`. Out-of-range values are rejected, never
  applied.
- Multiprocessing must use `nflmain/parallel.py`, which always uses "spawn". Forked workers hang
  after the MILP solver has run (the NBA lesson). `tests/test_parallel.py` enforces this.
- Test changes on a realistic copy (150 entries) and time them. **Every check must finish in under
  3 minutes.**

## Kenny's standing rules (built in)
- DK Classic roster: QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE), DST. $50,000 cap. Each player locks at
  his own game's kickoff.
- **Each contest in the DKEntries file is built on its own DK slate** (Early / Main / Afternoon /
  Thanksgiving), with that slate type's settings.
- **Locked players are never moved.** A player whose game has started stays in his slot
  (`lateswap.locked_mask`).
- Initial build only: each player is in 0 lineups or at least 5 (contests with 10+ entries).
  The rule is off after the first lock.
- No player in more than 50% of lineups. No lineup used twice anywhere in the portfolio.
- Large fields are judged as a portfolio on top-1% finishes. Small fields are judged on cash rate.
  ROI is reported only.
- Swaps must be confirmed on independent simulations. Late-swap options are built in parallel
  processes.
- Stacking is built in: the QB always has at least one of his pass-catchers, bring-backs are
  tried, and nobody faces our own DST.

## Kenny's lineup-rules workbook (owner rule, 11/08/2026)
`research/rules/NFL_DFS_Lineup_Rules_<date>.xlsx` holds the starting build rules. **Every weekly
chat reads the newest one** (latest date in the file name) before Tuesday's early look and before
every build. Its sheets:
- NFL Main Rules (M1-M18);
- Short Slate Changes (SS1-SS7, for 2-4 game slates: Afternoon, Thanksgiving);
- data, correlation and source sheets.
The Showdown sheet belongs to the NFL Showdown system; read it, never act on it here.

How to use it:
- **Respect Kenny's yellow columns.** "Use? (Y/N)" and "Your setting / notes" win over the
  workbook's own Setting. If Use? is N, don't apply the rule. If the columns are blank, treat the
  rule as proposed: tell Kenny which rules are in force and which aren't yet.
- **Hard** rules are never broken. **Soft** rules are penalties or portfolio-share targets the
  simulation may override.
- **Evidence "Opinion" rules get backtested on the history database (`research/backtest/`) before
  they change the build.** Report the result in plain words.
- The workbook doesn't change the build by itself. A rule reaches lineups only through a
  `config/settings.yaml` default or a code change. Both go by pull request, which Kenny merges.
  Kenny's "today only" or "from now on" values go in `data/settings/`.
- Its public data is mostly from before 2022. Where our own backtest disagrees, say so and show
  both numbers.
- A new version from Kenny is saved as a new dated file. Never edit or delete an older one.

## Secrets and paid data
- **Never commit API keys or tokens.**
- **No Odds API for now** (Kenny, 11/07/2026). Don't use The Odds API or `ODDS_API_KEY` at all.
  - Use only free sources: DK, ESPN lines and injuries, nflverse usage, and NWS weather.
  - Sportsbook props are an optional add-on that is **switched off**
    (`nflmain/project.py: PROPS_ENABLED = False`). Turning them on needs Kenny's OK.

## Ownership model research (owner rule, 11/08/2026)
`research/ownership/` holds Kenny's ownership research: the report
`NFL_main_slate_ownership_projection.md` and the workbook
`NFL_Ownership_Projection_Research_<date>.xlsx`. Read the newest workbook before any ownership work.
- **The Recommendations sheet (#1-#15) is the plan for our ownership model.** Kenny's yellow
  "Use? (Y/N)" and "Notes" columns win.
  - Y = do it, N = don't.
  - Blank = proposed. Ask Kenny before building it, except #1, which he started on 11/08/2026.
- **Weak items are starting guesses**: the late-news damping numbers (#8), the wind flag (#14),
  the uncertainty flag (#15), and the exact features in #5/#6. Fit them from our own DK contest
  downloads and snapshots before relying on them. Say which numbers are still guesses.
- **#1 is in force: save every DK contest download.** Every Sunday night (and by Wednesday at the
  latest, since DK deletes them after 10 days), Kenny downloads the contest file for every
  main-slate contest we entered, plus the Milly Maker, and uploads it to the weekly chat.
  - Run `python -m nflmain results <files> --date <Sun date>` and commit `data/contests/`.
  - The files sit behind Kenny's DK login, so Claude never signs in for him and never asks for
    his password.
  - The download only exists for contests he entered, so the Milly Maker needs at least one entry
    from him.
- A learned ownership model may replace the baseline only after it beats the baseline on
  held-out weeks across about a full season (#7).
- Never buy ownership data (FantasyLabs etc.); ask Kenny first if it seems worth it.

## Protect Kenny's other systems (NBA, NFL Showdown, UFC)
- `kennethhosek-a11y/NBA-dfs`, `kennethhosek-a11y/NFL-dfs` (Showdown) and
  `kennethhosek-a11y/UFC--dfs` are **read-only**. Never modify them, push to them, merge into
  them, or open pull requests on them. Code reused from them was **copied** here.
- Never create, change, disable or delete their scheduled tasks, and never message their chats.
- **NFL Main scheduled tasks must have names starting with `NFL Main`.**
- This system never delays or blocks the others. Keep checks under 3 minutes and never touch
  their sessions or files.
