# Weekly runbook: one chat per NFL week

Kenny's NFL Classic process. One Claude chat runs each week, from Tuesday's early look through
Sunday's last late swap. It is titled `NFL Main · Week N · Sun Mon D`. All times are **Eastern**.
Read `CLAUDE.md` first; this file is the schedule.

Contests and entry counts come from Kenny's **DKEntries.csv**. That file is the source of truth.
If it isn't clear which contests he's entering, or how many lineups, **ask him**.

## Talking to Kenny
- **Files go at the END of every reply.**
- **Every file gets a new name** (date + build time), so a re-download is never the same file:
  - workbook: `NFLMain_Prep_<date>_<slate>_<MMDD-HHMM>.xlsx` (`research.prep` names it);
  - lineups: `output/send/DKEntries_<MMDD>_<HHMM>_<what>.csv` (`nflmain` names it). `<what>` is
    build / inactives / t20 / lateswap.
- **Push notification (PushNotification) whenever a file is ready**, leading with the action:
  "Upload DKEntries_1115_1242_t20.csv - 3 entries changed (Nacua OUT)".
- Plain words. Say what changed and why in one or two lines per change.
- `output/DKEntries_current.csv` is the last file sent. Assume Kenny uploaded it unless he says
  otherwise. Always pass it explicitly: `--entries-file output/DKEntries_current.csv`.

## Commands
```bash
git pull origin main && pip install -r requirements.txt     # at every wake-up
python -m nflmain slates                                    # this week's Classic slates (type, locks, contests)
python -m nflmain run --slate main                          # projections for one slate
python -m research.prep --date <Sun date> --slate main      # Excel workbook (also --slate early / afternoon)
python -m nflmain build --entries-file <Kenny's DKEntries.csv>   # full build, every contest on its own slate
python -m nflmain news --entries-file output/DKEntries_current.csv --what t20   # news / inactives / late swap
python -m nflmain news ... --out "Name A,Name B" --q "Name C" --active "Name D" # news Kenny or a beat writer had first
python -m research.prep --export <workbook Kenny sent back> # his opinions -> data/overrides/<date>.csv
```
`news` refreshes projections (DK tags, ESPN injuries, lines, weather and research overrides). It
reports what moved since the build, then re-evaluates **every entry against every unlocked
player**. Live scores condition the sims for games already played.
- A swap happens only when it's confirmed on independent sims.
- An entry holding a player who won't play always changes.
- Locked players never move.
- It writes a new file only if something changed.

## Schedule

### Tuesday (once DK posts the next Sunday's salaries, usually Monday night or Tuesday)
- `python -m nflmain slates` to get the slate shape: which slates, how many games, lock times.
- Run the workbook for each slate Kenny usually plays, then send it at the end of the reply.
- Summarize:
  - opening lines (totals, spreads, implied team totals);
  - injury watch: who's on the injury report and when the practice reports come out;
  - salary notes and early chalk;
  - games to stack;
  - weather to watch.
- Wake-ups with `send_later` (`[America/New_York]` times): Wed 6:00 PM, Thu 6:00 PM, Fri 6:00 PM
  (after practice reports), Sat 6:00 PM, Sun 8:00 AM, 10:00 AM, 11:30 AM, 12:40 PM, 2:40 PM and
  3:45 PM. Add T-20 before any other window, e.g. 4:05 or 4:25.

### Wednesday / Thursday / Friday ~6 PM: practice reports + line moves
- WebSearch for practice participation (DNP / limited / full), injury designations on Friday, and
  line and total moves.
- Re-run the workbook (new file name). Lead the reply with **what changed since the last update**.
- Write research calls into `data/overrides/<Sun date>.csv` with a Note on each.

### Saturday ~6 PM: final research + weather
- Final injury designations (OUT / Doubtful / Questionable), weather (wind 15+ mph, rain, snow)
  and lines.
- Final overrides. Workbook at the end.

### Sunday
- **8:00 AM: news update.** Check overnight news (inactive leaks, weather), then re-run the
  workbook. Remind Kenny the build is at 10.
- **~10:00 AM: ask Kenny for his DKEntries.csv.** When it arrives, run
  `python -m nflmain build --entries-file <file>`. Send the file and a push notification.
  - Check the output: every contest built on its own slate, all entries filled, exposure, stacks.
  - Contests on a slate this system doesn't build (Primetime, Sun-Mon) are listed. Tell Kenny.
- **~11:30 AM: inactives (1:00 games).** Run `news --what inactives`. Rebuild only if it helps
  (`news` decides by confirmed gain).
- **12:40 PM: T-20 for the 1:00 lock.** WebSearch late news for the 1:00 teams, then run
  `news --what t20`.
  - If starters or news are still pending, re-check at **T-10 (12:50)** and **T-5 (12:55)**.
  - Each check must finish **in under 3 minutes**, news to file. If it can't, tell Kenny right away.
- **Between windows (1:00 games in progress): live late swap for the 4:00 players.**
  `news --what lateswap` conditions the sims on the live scores. In each changed entry, say
  whether it **leans into or away from chalk** and why:
  - an entry that's behind needs players the field doesn't have, so it leans away;
  - an entry that's ahead protects with chalk, so it leans in.
  `news` prints where each entry stands (points so far, projected final, top-1% chance).
- **~2:35-2:55 PM: inactives for the 4:00 games**, then their **T-20 (3:45 for 4:05, 4:05 for
  4:25)**, plus T-10 / T-5 if news is pending.
- After the last lock:
  - commit and push the day's data;
  - final reply on how entries stand;
  - remind Kenny to upload contest results (DK contest standings CSV) on Monday.

### Sunday night (~8:30 PM, after the last main-slate game): save the contest downloads
DK deletes contest downloads after 10 days, and they are our only ownership training data
(research/ownership, recommendation #1).
- Push a notification:
  "Save DK contest files: Play-Action, First Down, mini-MAX + Milly Maker - upload them here".
  Give Kenny these steps:
  1. On draftkings.com: **My Contests → Completed**. Click each NFL Main contest.
  2. In the contest page, click **Export Lineups to CSV** (some screens say "Download standings").
     It downloads `contest-standings-<id>.zip`.
  3. Do the same for this week's **Milly Maker**. It needs at least one entry from him.
  4. Upload all the files to this chat (zip or csv both work).
- When they arrive: `python -m nflmain results <files> --date <Sun date>`.
  - Reply with the field's habits in plain words: how often the field stacked, double stacks,
    bring-backs, FLEX split, salary used, duplicates, top-owned players.
  - Compare our projected ownership with the real %Drafted for the top 15.
  - Commit and push `data/contests/`.
- If nothing has arrived by Monday 6 PM, remind him once more. The files must be saved by
  Wednesday.

### Monday / after the week
- Contest results go in `data/contests/`. Ingesting them to retrain ownership is planned; see the
  HANDOFF "Next" list.
- **Start next week's chat**: `create_session` with
  - `source_url` = `https://github.com/kennethhosek-a11y/NFLMain-dfs`
  - `title` = `NFL Main · Week N · Sun Mon D`
  - a prompt that names the Sunday date (`YYYY-MM-DD`) and says: follow CLAUDE.md and this
    runbook, do Tuesday's early look, schedule the week's wake-ups, and list open items
    (unmerged pull requests, contest results, anything unfinished).
  - Once the "NFL Main weekly" scheduled task exists (below), it does this instead.

## Ownership research (read before any ownership work)
`research/ownership/` holds the report and the newest `NFL_Ownership_Projection_Research_<date>.xlsx`.
- Its Recommendations sheet is the plan. Kenny's Y/N column wins, and a blank means ask first.
- #1 (save every contest download) runs every Sunday night, see above.
- Weak items are starting guesses, refitted from the saved downloads.
- On Tuesdays, list which recommendations are in force, which are waiting for Kenny's Y/N, and
  which starting guesses the data has replaced so far.

## Scheduled task (after the first real slate works)
Create **one** scheduled task named **`NFL Main weekly`**. Every Tuesday at 9:41 AM Eastern
(`CRON_TZ=America/New_York 41 9 * * 2`), it starts a fresh session on this repo titled for that
week, with a standalone prompt as above. Never touch any scheduled task whose name doesn't start
with "NFL Main".

## Slate types (decided by each contest's DK draft group)
| Type | Games | Late swap | Settings |
|---|---|---|---|
| early | 1:00 only | no (one lock) | `early.*` |
| main | 1:00 + 4:00 | yes, on the 4:00 players | `main.*` |
| afternoon | 4:00 only (and Turbo) | 4:05 vs 4:25 only | `afternoon.*`: heavier game stacks, light gate |
| thanksgiving | 3 games, Thursday | three windows (~12:30, ~4:30, ~8:20) | `thanksgiving.*` |

## Thanksgiving (Thu Nov 26, 2026)
- The chat for Week 12 (`NFL Main · Week 12 · Sun Nov 29`) starts Monday Nov 23 or earlier and
  also runs the Thursday slate.
- **Dry run working by Sun Nov 22. Warn Kenny by Fri Nov 20 if it's at risk.**
- Each game gets its own inactives check (~90 min before kickoff) and its own T-20:
  ~11:10 AM inactives, 12:10 PM T-20; ~3:00 PM inactives, 4:10 PM T-20; ~6:50 PM inactives,
  8:00 PM T-20 (adjust to DK's real kickoffs).
- Between windows: live late swap on the later games, same as Sunday.
