# Henry's Match Engine

Static site deployed on Render. The live product (index.html) is a four-league
football predictor -- Premier League, La Liga, Bundesliga, Ligue 1. Six tabs:
Today (overview), Best Picks, Player Picks, Parlay, Tables, Grades, plus each
individual league's own fixtures/table view. Owner: John (guchhenry91).

Repo also holds `worldcup.html`, the completed World Cup 2026 tournament
(daily picks, group projections, title odds, full graded knockout bracket).
It is an ARCHIVE, kept exactly as it finished -- final record 64-26 of 90
(71%), champion Spain -- and should not need to change again.

## Layout
- `index.html` — the SITE ENTRY POINT: the unified predictor (Today, Best Picks,
  Player Picks, Parlay, Tables, Grades, and each league's own view). Reads
  `data/leagues/*.json`. Single self-contained file -- no build step, no external
  stylesheet (the whole "Midnight" glass UI is inline `<style>`; `app.css` is a
  leftover from the pre-redesign UI and is no longer referenced by anything live).
  World Cup is intentionally NOT a nav tab -- reachable only via a quiet footer
  link, since the tournament is over and shouldn't compete for attention with the
  live product.
- `worldcup.html` — the completed World Cup 2026 tournament, preserved verbatim:
  daily picks, group projections, title odds and the full graded knockout bracket.
  Reads `data/predictions.json` only. Final record **64-26 of 90 (71%)**, champion
  Spain. Links back to `index.html` via a fixed footer anchor. It is an ARCHIVE --
  the tournament is over, so this page should not need to change again.
- `predict.py` — prediction engine (pure stdlib). Run `python predict.py` to regenerate `data/predictions.json` from `data-raw/`.
- `data-raw/schedule.json` — all 72 group matches (do not change ids).
- `data-raw/ratings.json` — Elo + FIFA per team (baseline; predict.py applies result-based Elo deltas itself — do not manually edit after tournament start).
- `data-raw/news.json` — per-team form, injuries, key players, headlines.
- `data-raw/players.json` — per-team top attacking threats (3–4 each) used for scorer predictions: player, club, pos, club goals/apps, pens.
- `data-raw/results.json` — final scores keyed by match id as STRING.
  - **Group matches (ids 1–72):** `{"5": {"home_goals":2,"away_goals":0}}`.
  - **Knockout matches (ids 73–104:** R32 73–88, R16 89–96, QF 97–100, SF 101–102, 3rd 103, Final 104): record the ACTUAL teams and the advancer, because the projected slotting won't always match FIFA's real draw and pen shootouts must be captured: `{"74": {"home":"Germany","away":"Paraguay","home_goals":1,"away_goals":1,"winner":"Paraguay"}}`. Map an actual result to a bracket id by its HOME team (a group **winner/runner-up**, which is reliable); the projected AWAY team (a third-placed side) may be wrong, so trust the web result for the real opponent. `winner` = the team that advanced (crucial when a tie is level after extra time and decided on penalties). `knockout.py` then displays the real teams and grades the model's pick against `winner`.
- `data-raw/bracket.json` — knockout bracket structure (slots, dates, venues). Static; do not edit.
- `predict.py` runs a 20k-sim Monte Carlo each time → `knockout` section in predictions.json (title odds + projected bracket). Set env `WC_SIMS` to change sim count.
- `data-raw/picks_log.json` — **auto-managed pick tracker** (do not hand-edit). predict.py locks each match's pick before kickoff and freezes it once a result lands, so the win/loss record grades the genuine pre-match pick (never a hindsight re-computation). It is committed each run so the record persists. `record` in predictions.json = `{correct, wrong, total, pending, by_confidence}`.

## Publishing (IMPORTANT)
Never run predict.py + git + deploy hook by hand in the tasks. After editing any data file, run **`python deploy.py "<message>"`** — one atomic step that re-runs the model, grades locked picks, commits, pushes, and triggers the Render deploy. It self-heals a stale publish (re-runs the model so unpublished results get caught) and always re-triggers the deploy, so the live site can never lag the repo. `status.py` is the gate helper: prints `finished_unrecorded` + `upcoming_4h` so a task knows whether there's anything to do.

## Daily update procedure (run every morning)
1. Web-search final scores of all WC matches played yesterday (and any missed earlier); add them to `data-raw/results.json`.
2. Web-search overnight team news: injuries, suspensions, lineup news for teams playing TODAY and TOMORROW. Update those teams' entries in `data-raw/news.json` (update `form` strings with yesterday's results too, and set top-level `updated`).
3. If any player listed in `data-raw/players.json` for a team playing today/tomorrow is newly ruled OUT, remove them (so scorer picks stay accurate).
4. Run `python predict.py`. Verify it prints "Wrote 72 matches" and no errors.
5. Commit all changes ("daily update YYYY-MM-DD: results + news") and push to main.
6. Render is connected by public Git URL and does NOT auto-deploy on push — trigger a redeploy by POSTing the deploy hook stored at `C:\Users\John\.claude\worldcup-deploy-hook.txt` (PowerShell: `Invoke-RestMethod -Method Post -Uri "<hook>"`).

Rules: never invent scores or injuries — only verified info. Team names must exactly match the names in schedule.json. Keep reasons concise.

---

# Leagues engine (the live product: PL, La Liga, Bundesliga, Ligue 1)

The predictor living in `leagues/`. It shares the repo and `deploy.py` with
the archived World Cup app but **touches none of its files**: the WC
engine stays pure-stdlib, the league engine needs pandas/scipy/penaltyblog.

- `index.html` — the unified UI (see Layout above; self-contained, no `app.css`).
- `python -m leagues.publish` — the one command. Fits the model, sims the season,
  builds player props, locks picks, builds the parlays, writes `data/leagues/*.json`
  atomically.
- `python -m leagues.tune` — the match-model gate (walk-forward vs de-vigged
  closing odds). It SWEEPS xi/xg_weight on the earlier seasons, then re-scores the
  winner on a held-out final season, and promotes a challenger over the shipped
  config only if a paired 95% bootstrap CI of holdout RPS lies entirely below
  zero. Rejected candidates are written to the report. It emits TWO files:
  `backtest_report.json` (per-league scores + a `_pooled` block of tier hit rates)
  and `release_policy.json` (the per-league xi/xg_weight publish.py actually
  runs). Both are generated -- never hand-edit either, and note that a
  single-league invocation deliberately leaves release_policy.json alone so it
  cannot drop the other three leagues.
- The walk-forward backtest **runs the model that ships**, second-tier promoted
  priors included, with the prior season derived from each cutoff
  (`second_tier.feeder_season`) so it stays strictly causal. It previously fitted
  without priors, so promoted clubs raised KeyError and their fixtures were
  silently dropped -- excluding the hardest games in the league and measuring a
  model publish.py never used. The reported RPS got WORSE when this was fixed;
  that is the fix working.
- `python -m leagues.props_backtest` — the props gate. **It validates the per-90
  RATE estimation only, NOT the probabilities the board publishes.** It is handed
  each player's actual minutes, so it is blind to minutes-projection error (a
  dominant driver of live prop accuracy), and it does not exercise `match_props`'s
  rescale or `HOME_SHOT_FACTOR`. Its bar is `mae < baseline_mae` on two metrics --
  no minimum sample, no calibration check, no significance test. The live record is
  the first real test of the published numbers. Accepted gap, not an oversight:
  Understat cannot support an honest per-match anytime-scorer curve, and a gate
  that looks rigorous without being so would be worse than a stated limitation.

## Data sources
- Results + closing odds: football-data.co.uk. Team xG: Understat.
- **Players: Understat season stats + shot events** — NOT FBref. soccerdata's
  FBref player-match reader drives a headless Chrome per match page (~4/min):
  five seasons of one league is ~8 hours. Understat gives the same signal in
  seconds.
- **Penalties are unlabelled**: soccerdata maps Understat's "Penalty" situation
  to NA. Match on NA, not on the string, or every club silently gets no penalty
  taker (see the regression test in `tests/leagues/test_players.py`).
- **Promoted-club priors come from second-tier form**, NOT ClubElo (which was a
  single third-party point of failure — down for days). Each promoted club's prior
  is derived from its actual second-division season (football-data.co.uk E1/SP2/D2/F2)
  via a calibrated linear map: attack carries a mild signal, defence none (promoted
  clubs concede ~+0.19 above average regardless). See `leagues/second_tier.py` and
  `scripts/calibrate_level_gap.py`. A club that can't be resolved in the second-tier
  feed falls back to the weakest-side seed with a `data_warnings` note.
- **soccerdata shot-events bug**: `read_shot_events` crashes on GER-Bundesliga (a
  match roster returns as a list, not a dict). Both `fetch_player_logs` and
  `team_shot_context` guard it and degrade (SOT via league-average ratio, neutral
  opponent factors) rather than sink the league.
- **`HOME_SHOT_FACTOR` (leagues/props.py)** -- home sides shoot ~23% more than away
  sides (measured, not guessed: `scripts/calibrate_home_shot_factor.py`, 7,007
  matches across 5 seasons and all four leagues, using the HS/AS columns already
  present in the football-data.co.uk CSVs `leagues/history.py` fetches). Applied
  symmetrically to a player's shot budget for the shots/SOT props markets, which
  are NOT covered by the goals-side rescale (see props.py's own docstring) and so
  needed their own venue-awareness channel.

## The published boards
- `data/leagues/best.json` — cross-league **match-winner** picks at p>=0.65.
- `data/leagues/player_picks.json` — cross-league **player** picks in three markets:
  anytime goalscorer, 2+ shot attempts, 1+ shot on target. Bars are PER MARKET
  (`PLAYER_PICK_MIN_PROB` in publish.py): goalscorer **0.40**, shots **0.70**, SOT
  **0.62**. Goalscorer is lower because a team scores ~1.5 goals and one man takes
  a share, so the best anytime price in any of these leagues is ~50% -- a 0.70 bar
  there would publish an empty section forever. SOT was lowered from 0.70 to 0.62
  for the same reason discovered later: "1+ shot on target" needs ~1.2 expected
  SOT to clear 70%, which only an elite-volume shooter reaches, so 0.70 published a
  one-name board. If a market bar keeps publishing near-empty, that is a sign the
  bar sits at the market's real ceiling, not that the model is being appropriately
  strict -- check the actual probability distribution before assuming the bar is
  right.
- `data/leagues/parlays.json` (`leagues/parlays.py`) — model-built accumulators
  from the Best Picks + Player Picks boards' OWN already-published probabilities,
  never a fresh guess. Two sections: all-four-leagues and Premier-League-only.
  ONE LEG PER MATCH so the combined probability is a genuine product of
  independent legs. Ungradeable legs (Bundesliga player props -- see below) are
  excluded from every parlay, since a leg that can never settle would leave the
  whole parlay stuck "pending" forever. Frozen and graded all-or-nothing with the
  same discipline as every other pick (see `lock_parlay`/`grade_parlay`).
- **Tables tab** -- the REAL league standings from results played so far
  (`actual_standings()` in publish.py: points, GD, W/D/L), distinct from the
  Monte Carlo-projected `table` field used elsewhere. Pre-season this is every
  club on zero, alphabetical; UCL/relegation shading only appears once games have
  actually been played, so a zero-point alphabetical list never misleadingly
  implies real standings.
- Best Picks and Player Picks are graded **separately**, and player picks are also
  graded per market: a 45% goalscorer and an 80% shots pick are both near their
  market's ceiling, so pooling them yields a headline number describing neither.
  Parlays carry their own separate record too. The Grades tab shows all tiers, plus
  a calibration chart (pools `by_confidence` across both boards) comparing stated
  confidence to actual hit rate -- a trust/transparency feature, not a model-
  accuracy one; it doesn't change how any pick is made.

**Player picks are graded from shot events**, not from `fetch_player_logs` (which is
one row per player-SEASON and cannot say whether a man scored in a given fixture).
`players.match_player_stats()` counts goals/shots/SOT per player per game, INCLUDING
penalties (an anytime pick wins on a penalty; `np_goals` would grade that a miss) and
EXCLUDING own goals. A player with no shot row grades **wrong, not void** — the feed
cannot separate "didn't play" from "played, never shot", so we take the harsher
reading deliberately: it can only understate the record, never inflate it.

**That harsh reading applies ONLY to a fixture the feed actually covers.** A pick
is graded only when the shot feed holds the player's own side on that date
(`_covered_sides` in publish.py); otherwise it stays PENDING and is listed under
`awaiting_data`. This is not a nicety. On 2026-08-23 Understat had published
nothing for 2026-27 — newest row 2026-05-24, the previous May — while 26 fixtures
had been played. The frame was nowhere near empty (26,401 rows for the PL alone),
so the "is the feed available" guard passed, every lookup missed, and the board
published **0 correct / 18 wrong**. Not one of those losses was real, and it read
as a 0% hit rate against a stated 74.4%. Absence of the match is not evidence
about the player: where the feed is silent, the honest answer is "not yet", and a
fabricated loss in an append-only record cannot be taken back.

**API-Football is the fallback player feed** (`scripts/sync_player_stats.py` →
`data-raw/leagues/player_stats.json`, read by `players.api_match_stats`). It
supplies goals/shots/SOT per player per fixture, but **only where Understat has
not filed the match** — Understat is shot-event derived and identifies penalties,
so it wins wherever it has the game, and the merge applies it last so it
overwrites. One request per fixture, only for played fixtures that carry a frozen
pick and are still uncovered, so it costs nothing once Understat catches up.
Player names are joined across the two feeds by `players.resolve_squad_name`,
which reuses the roster rescue's guards and is constrained to ONE CLUB IN ONE
FIXTURE; a player it cannot match confidently is left pending rather than
guessed, because a wrong join does not mislabel a player, it settles a bet
against a stranger's shot count.

**Bundesliga: gradeable, but its SOT market stays withheld.** Two different
questions used to share one answer, because `shots_ok` answered both. They are now
separate:

- **Can the RATES be measured?** (`players.shot_events_available`) — still NO for
  Bundesliga: its Understat shot events crash upstream, so the on-target ratio
  would be a league average, an assumption dressed as a measurement. The **SOT
  market is therefore still withheld there entirely**, and the fallback cannot
  help — rates are built from SEASONS of history, and API-Football only covers
  fixtures it is asked to fetch.
- **Can a pick be SETTLED afterwards?** (`players.grading_feed_available`) — now
  YES, because API-Football reports Bundesliga goals/shots/SOT per fixture like
  any other league. So Bundesliga **goal and shots picks now grade, count in the
  record, and are eligible for parlays**.

Collapsing these two cost the league its entire player record: every pick there
published `gradeable: false` and was excluded from the record and from every
parlay, on the strength of a fact that only ever bore on the rates. `gradeable` is
a forward-looking claim, so `grading_feed_available` checks the fallback is
genuinely usable (league known, `API_FOOTBALL_KEY` present) rather than assuming
it — without a key, a league with no shot feed is still correctly ungradeable.

**`MIN_SQUAD_FOR_PROPS = 6`** — a team with fewer players in the rates table gets NO
props. The sigma-lambda rescale forces a team's players to sum to the match lambda,
so a near-empty squad hands one man the whole team's goals: promoted Schalke had a
single player with top-flight history and published as a **72.8% anytime scorer**
when nothing else in four leagues beat 50.8%. One player is more dangerous than
none — none is visibly a hole, one looks like the best pick on the board.

## Roster verification
`python -m scripts.sync_rosters` snapshots every current 2026-27 club and player
from the ESPN league/team roster feeds into `data-raw/leagues/rosters.json`.
`python -m scripts.roster_integrity_check` verifies league membership, duplicate
player IDs and visibly reports thin/incomplete source rosters. The snapshot is
dated and provisional while the summer registration window remains open. It is
an identity/eligibility source only; Understat remains the performance-rate
source, so a player with no usable history is never assigned invented scoring or
shot rates. The snapshot CORROBORATES, it does not convict. Where a club's roster is
COMPLETE (>=18 listed) it is authoritative: it reassigns a player's club, and a
player absent from it is dropped as departed. Where a roster is THIN, MISSING or
STALE we keep the existing attribution and warn on the page, because absence from
incomplete evidence is not evidence of absence.

That distinction was learned the hard way. Treating thin rosters as proof deleted
Real Madrid, Barcelona, Atletico, PSG, Marseille and 14 of 18 Bundesliga clubs --
Mbappe and Raphinha among them, 70% of La Liga and Ligue 1 -- because the free feed
happened to list fewer than 18 names for them. A surname rescue also runs, but only
within the club a player is ALREADY at, so it can never invent a transfer; without
it Understat's "Thiago" vs the feed's "Igor Thiago" deleted a real Brentford player
over a spelling difference. The match model is unaffected either way.

## Data JSON validation
`scripts/validate_data_json.py` parses every hand/agent-edited data file
(`data-raw/leagues/transfers.json`, `news.json`, etc. -- NOT `data/leagues/*.json`,
which is machine-written by `json.dump` and can't be malformed this way) and
reports the exact file/line/column on a parse error. Runs FIRST in
`leagues.yml`, before the slow model fetch, so a bad file fails in seconds
instead of after an hour. `scripts/hooks/pre-commit` (install via
`scripts/hooks/install.sh`) runs the same check on the STAGED blob before a
local commit, so the class of bug that froze the pipeline for a full day on
2026-08-02 (one missing comma in transfers.json) can't happen again -- caught
at commit time, not discovered by a failed run hours later.

## Telegram notifications
Two separate channels, both gated on `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`
being set (silently skip otherwise -- a missing notifier must never fail the
run): failure alerts (`leagues.yml`'s own "Alert on failure" step, fires on
ANY failed step) and `scripts/telegram_picks.py`, which sends each Best Pick
and Player Pick to Telegram exactly ONCE, the run it first locks (never
early, while the number could still move; never repeated). Runs BEFORE
"Commit refreshed league data" deliberately -- its dedup memory
(`data-raw/leagues/telegram_sent.json`) has to be written before that step's
`git add` or it never persists across scheduled runs. A send failure is
caught and logged, never marked sent, so it just retries next run.

## Deploy dedup
Two deploy paths exist and both hit the same Render hook, so a guard exists
to stop them double-firing on one push: `.github/workflows/deploy-on-push.yml`
skips commits authored by `leagues-bot` (leagues.yml already deploys those
itself) AND commits whose message contains `[auto-deployed]` (which
`deploy.py` appends automatically, since it POSTs the hook itself after
pushing). Render's plan meters deploys, so a redundant one isn't free -- this
was a real, recurring issue before the guard existed.

## Scheduled jobs
`ops/leagues_weekly.py` and `ops/leagues_matchday.py` are manual wrappers around
the four-league publish path. They pass `--league-data` to `deploy.py`, so a league
refresh never regenerates or stages World Cup files.

**DO NOT REGISTER THEM.** This instruction is superseded and following it would
cause an incident. `.github/workflows/leagues.yml` already runs the same job on the
same crons (`0 6 * * 2`, `0 23 * * 6,0`). Registering these locally too means two
processes publishing, both committing to the same repo, both triggering Render —
git conflicts and double deploys, on a schedule.

GitHub Actions wins because it does not need the laptop on or the app open, which
was the whole point of moving there. Keep `ops/leagues_weekly.py` and
`ops/leagues_matchday.py` as MANUAL commands (`python -m ops.leagues_weekly` for a
publish-and-deploy on demand) — just never on a schedule.

The one scheduled task that IS correct to have locally is `leagues-matchday-news`
(cron `0 8-22 * * *` — HOURLY, 08:00-22:59, all seven days since 2026-08-14, when
La Liga and the PL began scheduling midweek fixtures): it needs judgement about
confirmed XI vs rumour, which is why it cannot live in Actions.

**Its cadence is NOT what protects the pick lock.** Freezing a pick is
`leagues.yml`'s `0,30 11-22 * * *` cron in GitHub Actions, which is every 30
minutes precisely because it must be narrower than the 45-minute lock window.
This local task has its own, much wider gate — fixtures 45 minutes to 3 hours
out, a 2h15m window — so hourly still gives every fixture about two chances to
be researched before it locks. Do not "restore" this to 30 minutes on the belief
that the lock depends on it; if voids ever appear in the record, the cadence to
tighten is the Actions one.

**It ABORTS on a dirty working tree** (SKILL.md STEP 0.5) and must never be
"fixed" by stashing. It once improvised `git stash` to get a blocked
`git pull --ff-only` through and silently swallowed two hours of uncommitted
engine work. Worse, it runs `leagues.publish` and `deploy.py`, so on a dirty tree
it would ship someone's half-finished model to the live site. If you are working
in this checkout with uncommitted changes, expect the sweep to skip and say so --
that is the guard doing its job, not a failure.

Both abort rather than deploy if a fetch fails — never ship a stale-but-fresh-
looking file.
