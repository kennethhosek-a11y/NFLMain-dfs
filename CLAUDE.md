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

## Secrets and paid data
- **Never commit API keys or tokens.**
- **No Odds API for now** (Kenny, 11/07/2026). Don't use The Odds API or `ODDS_API_KEY` at all.
  - Use only free sources: DK, ESPN lines and injuries, nflverse usage, and NWS weather.
  - Sportsbook props are an optional add-on that is **switched off**
    (`nflmain/project.py: PROPS_ENABLED = False`). Turning them on needs Kenny's OK.

## Protect Kenny's other systems (NBA, NFL Showdown, UFC)
- `kennethhosek-a11y/NBA-dfs`, `kennethhosek-a11y/NFL-dfs` (Showdown) and
  `kennethhosek-a11y/UFC--dfs` are **read-only**. Never modify them, push to them, merge into
  them, or open pull requests on them. Code reused from them was **copied** here.
- Never create, change, disable or delete their scheduled tasks, and never message their chats.
- **NFL Main scheduled tasks must have names starting with `NFL Main`.**
- This system never delays or blocks the others. Keep checks under 3 minutes and never touch
  their sessions or files.
