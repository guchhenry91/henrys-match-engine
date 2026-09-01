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

# Leagues engine (the live product: PL, La Liga, Bundesliga, Ligue 1, Serie A)

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

## Serie A (added 2026-08-30)
The fifth league. **Every feed was checked before the config entry was written,
not after**: football-data `I1` has all five fitting seasons at 380 matches with
B365 closing odds, `I2` (Serie B) supplies the promoted-club priors,
fixturedownload's `serie-a-2026` has 380 fixtures, and Understat's `ITA-Serie A`
returns team xG at **100% coverage**. Its Understat shot events work, unlike
Bundesliga's, so no market is withheld.

- **Head-to-head tiebreak**, like La Liga. Serie A separates clubs level on points
  by their meetings, not goal difference; `gd` would render a table that disagrees
  with the official one at exactly the positions people care about.
- **Canonical spelling is football-data's**, as everywhere else. The alias map was
  built by reading all 27 club names each feed actually uses across six seasons
  rather than from memory — only three differ (`AC Milan`→`Milan`, `Parma Calcio
  1913`→`Parma`, fixturedownload's `Internazionale`→`Inter`). All 30 distinct
  source names were then asserted to resolve, because one unmapped name raises
  `UnknownTeam` mid-fit.
- **The gate has been run.** Holdout n=380, accuracy 52.1%, **RPS 0.2020** —
  in line with PL 0.2073, La Liga 0.2002, Bundesliga 0.1972, Ligue 1 0.2039. It
  retained the incumbent xi/xg_weight like all four others, and the market beats it
  by 0.0051 RPS, which is the expected direction. Tier hit rate 75.0% at p>=0.65
  on n=100.
- **`release_policy.json` has no SERIEA entry yet** — a single-league tune
  deliberately leaves that file alone so it cannot drop the other four. No
  practical effect, since Serie A retained the same defaults publish falls back to;
  the entry appears on the next full gate run.
- **No roster evidence yet.** ESPN's `ita.1` feed does serve all 20 clubs and
  `sync_rosters` now covers it, but the sync is currently answering **403 for every
  league** — a pre-existing outage, not a Serie A problem. Until it clears, Serie A
  publishes the "no current-roster evidence" data_warning and attribution falls
  back to last season plus transfer overrides.

**Adding it found the four-league list written out in eleven places.** That is the
real lesson: the config was the easy part. `tests/leagues/test_seriea.py` now
asserts every live-path mapping against `config.LEAGUES`, so a sixth league fails
loudly rather than silently losing its board, its lock, or its parlay section. The
UI derives its slug list from one map (`LGSLUGS`) instead of five copies, and
`test_publish_multi` derives its expectations from `FILE_FOR` rather than naming
files by hand.

**`roster_integrity_check.audit` now separates absent from contradictory.** A
league the snapshot does not cover at all is a WARNING; a league present but wrong
is still an ERROR. That is the same absence-of-evidence distinction the engine
already draws for thin club rosters, and without it a newly added league — or the
current ESPN outage — fails the whole audit as though the data were wrong rather
than missing.

## Exact-score board (bet365 6 Scores Challenge)
`leagues/six_scores.py`. Correct score is the hardest common market and this
engine does not beat the trivial strategy in the Premier League: **12.84% on a
holdout against ~13.0% for the best any lambda-based model can do**, measured by
an empirical lookup allowed to fit ON the answers. The board is AT its ceiling --
there is no accuracy headroom, and anything promising 2+ of 6 weekly (33% a
fixture) is promising 2.6x a cheating upper bound.

**`data-raw/leagues/score_calibration.json`** corrects a real, stable bias in the
fitted grid: over 1,900 PL matches it over-predicts 0-0 by 35% and 1-1 by 10%,
and under-predicts 2-2 by 20% -- Dixon-Coles' low-score correction pushed too far
for this league. Generated and gated by `scripts/calibrate_scorelines.py`
(corrections learned on the earlier seasons, scored on a held-out tail), never
hand-edited.

It buys VARIETY, NOT ACCURACY, and must never be sold as the latter: holdout hit
rate moves -0.21pp (noise) while 1-1 falls from **89% to 74% of fixtures** and
distinct scorelines rise from 4 to 7. A board that says the same thing six times
is one nobody can use; that is the entire trade, and the gate accepts a small
accuracy cost (`TOLERANCE_PP`) only because variety is the point.

**SCOPE IS LOAD-BEARING.** These factors are applied when choosing which
SCORELINE to display -- `top_scorelines` and `score_for_outcome` -- and nowhere
else. Never to `p_home`/`p_draw`/`p_away`, never to the match pick, never to
anything the record grades. The match model is calibrated and working (stated
48.3% vs 56.0% actual over its first 25 picks) and is not retuned to tidy a
scoreline board.

The board also publishes what its record will LOOK like before it happens:
`expected_of_six` (0.71), `odds_of_none_pct` (~47%) and `odds_of_two_plus_pct`
(~15%). Zero correct is the single most likely week -- more likely than one -- so
a 0/6 is the model behaving as measured, not misfiring. Picks are ranked by
confidence (they range ~12-14.5%) and each card shows the runner-up score with
`gap_to_next_pp`, the cost in probability points of playing it instead.

---

# NFL engine (`nfl/`, tab: NFL)

A second sport, deliberately kept in its own package. The soccer engine is a
Dixon-Coles model over two low-count goal processes; nothing about that survives
translation to a sport scoring 20-30 points from drives. What IS shared is the
discipline: walk-forward validation on seasons the model never saw, a release
gate that WITHHOLDS rather than caveats, and evidence published beside the picks.

- `python -m scripts.nfl_backtest` — the gate. Writes `data-raw/nfl/backtest_report.json`.
- `python -m nfl.publish` — the board. Writes `data/nfl/board.json`.
- `.github/workflows/nfl.yml` — daily publish, Tuesday re-gate. Separate from
  leagues.yml because the NFL slate is weekly and fitting five models takes
  minutes; bolting it onto a 15-minute football cron would multiply that by 96 a
  day and let either sport's failure take the other down.

## Data: nflverse, NOT the API-NFL key
Flat CSVs, no daily cap, so a six-season backfill is free. This matters: the
API-Football account hit its daily limit twice in one week and a backfill is
exactly the shape of job that causes that. **The current release path is
`stats_player/stats_player_week_{season}`** — the older `player_stats/player_stats_{season}`
still serves 2020-2024 and 404s for 2025, which looks like "no recent data"
rather than "wrong URL". `recent_team` was renamed `team`; normalised in data.py.

## What is measured, and against what
EIGHT seasons load (2018-2025), FOUR are scored (2022-2025). 2018-2021 are
burn-in and never graded -- widening that window is more evidence before the
test, not a change to the test.

| Market | n | Brier | Baseline | ECE |
|---|---|---|---|---|
| anytime_touchdown | 11,703 | 0.1850 | 0.1894 | 0.014 |
| receiving_yards | 9,337 | 0.2321 | 0.2430 | 0.017 |
| rushing_yards | 3,725 | 0.2278 | 0.2378 | 0.027 |
| passing_yards | 2,061 | 0.2319 | 0.2405 | 0.037 |
| team_winner | 1,087 | 0.2231 | 0.2473 | 63.6% accuracy |

All sixteen prop season-market combinations beat their baseline. Team winner
beats home-advantage (53.3%) in every season.

## Decisions that are load-bearing
- **The line is the player's own entering MEDIAN**, not his mean and not a
  sportsbook price. Yardage is right-skewed, so a line at the mean is beaten only
  36% of the time -- a market a model looks clever in by always saying "under",
  against a baseline flattered the same way. `MIN_LINE` then floors it: nobody
  quotes "over 0.5 receiving yards", and the floor applies in the BACKTEST too,
  so the gate measures the product on screen.
- **`MIN_OPPORTUNITY` requires a role**, or a receiver with a median of zero
  carries acquires a rushing line.
- **Ensemble, never selection.** Candidate feature sets are all fitted and
  AVERAGED. Selecting the best on an inner split made markets flip in and out of
  release as candidates were added -- that was the selector's variance, not model
  quality, and averaging removes the choice.
- **Cross-fitted calibration.** Every training row gets an out-of-fold
  prediction, then the model refits on all of them. A held-out tail cost 30% of
  the data and, on the first fold, degenerated into calibrating on rows the model
  had memorised.
- **Isotonic only above 5,000 rows**, Platt below: isotonic carves step functions
  out of noise on small samples.
- **The calibration bar is `max(0.04, null_95)`** -- practically calibrated OR
  statistically indistinguishable from perfect at that sample size. A bare
  constant punishes small markets for being small; a bare null test punishes large
  ones for being measurable.

## API-NFL: availability only
`nfl/api.py` + `scripts/sync_nfl_injuries.py` -> `data-raw/nfl/injuries.json`.
The key is used for the ONE thing nflverse cannot supply: who will actually play.
32 requests a day, one per team.

A player reported OUT is removed from the board entirely -- his last five games
look exactly as good as anyone's right up until he is inactive. A DOUBTFUL player
stays with the doubt shown, because dropping him hides a real pick while hiding
the flag misleads. **Absence from the report means NOT REPORTED, never confirmed
fit**: a quiet file because the sync failed is indistinguishable from a quiet file
because everyone is healthy, and treating the first as the second is how a
ruled-out player gets published at full confidence.

The client counts its own requests, reads the account's remaining allowance from
the rate-limit headers, and prints both every run. `data-raw/nfl/_quota.json` is a
dated circuit breaker: once the API reports the daily limit reached, every caller
no-ops for the rest of that UTC day. Both exist because of the API-Football
incident -- 7,500 calls gone in four hours, traced only by reading thirty-one
workflow logs since two of four scripts reported nothing, and every run afterwards
still firing doomed requests at a wall.

## Rosters: who is actually on each team (`nfl/rosters.py`)
Box scores say where a man last PLAYED; the roster says where he IS. Those agree
all season and disagree all summer, which is when the board must be right.
**Source is nflverse `roster_{season}.csv` -- free, 91.6 players a team, and it
carries `gsis_id`, the SAME key as `player_stats.player_id`, so clubs are joined
by IDENTITY and name matching never enters into it.**

**This was built against API-NFL first and it failed in the worst possible way.**
Its player endpoint returned 43-71 plausible names a team -- passing every
count-based "complete roster" check -- and those names did not include Patrick
Mahomes, A.J. Brown, Alvin Kamara or Austin Ekeler. Philadelphia came back as Andy
Dalton, Britain Covey and Danny Gray. The board dropped **177 current players** as
having left the league, halved itself, and every guard reported success. The
lesson is in `rosters.corroborates`: a completeness check that counts rows cannot
tell a full roster from the first slice of one, and "all 32 teams complete" is the
most reassuring possible way to be wrong.

- `ACT`/`E14` only. `CUT`, `RET` and `RES` are exactly the people the board used
  to keep projecting.
- **Corroboration gate**: the file may only overrule the box scores if it
  recognises >=60% of the players we independently know were active. nflverse
  agrees 86%; API-NFL managed 49% and was refused wholesale.
- A player listed by two teams mid-camp is placed by neither -- guessing is how a
  projection lands on the wrong team while looking certain.
- Every card carries `club_source`, so a confirmed roster spot is distinguishable
  from an inference off last season.

Live effect on the week 1 board: 98 confirmed, **10 real reassignments** (Isaiah
Likely BAL->NYG, Jauan Jennings SF->MIN, Rico Dowdle CAR->PIT, Wan'Dale Robinson
NYG->TEN, Tank Bigsby JAX->PHI and others), nothing wrongly dropped.

## Depth charts: who starts, and who was cut (`nfl/depth.py`)
**The "no depth charts" hole is closed.** It was not hypothetical: on the 2026
week 1 board, **six of nineteen passing picks were backup quarterbacks**. Marcus
Mariota (WAS QB2) was the second-highest passing pick on the whole board at 62%,
and Cleveland published its QB2 *and* QB3 while its starter appeared nowhere.

The trap is mechanical. A backup's line is his own entering median, set in relief,
so it sits at 150-180 passing yards against a starter's 210-265. "Over" looks easy,
and the model is right that he would beat it **if he played** — what it cannot see
is that he will not take a snap.

It also answers a staleness question the roster cannot. Four days after the cut to
53, nflverse's season roster still listed **90 active players a team**, and so did
its weekly file; neither could say who had been cut. The depth chart is
republished continuously and only holds players a team is actually carrying.

- **`MAX_DEPTH_RANK`** (nfl/config.py): passing **1**, the other three **3**.
  Passing is winner-take-all — one quarterback takes essentially every drop-back,
  so a QB2 is usually *none* of that market rather than a smaller share. The other
  markets genuinely share: a WR3 and an RB2 play real snaps, so only deep reserves
  are cut. Rank 4+ removed Mack Hollins (WR4) at 74.7%, then the highest receiving
  pick on the board.
- **Removed, not flagged** — the same reason a player reported OUT is removed.
- **Absence from the chart KEEPS a player.** Dropping on absence is exactly how
  177 current players were deleted in August. Only an explicit rank too far down
  removes anyone.
- **The corroboration gate applies** (`MIN_DEPTH_COVERAGE = 0.80`): a chart that
  does not recognise the board's own players is ignored wholesale, and
  `depth_check.applied` on the board says whether the filter actually ran.
- Every card publishes `depth_label` ("QB1", "WR3"), which is the context that
  tells a reader whether a low line is a soft spot or a warning.

Live effect on week 1: 108 picks -> 93, every passing pick now a QB1.

## nflverse caching has a MAX AGE
`_read_csv` used to cache forever, and CI caches the download directory between
runs, so a file fetched once in pre-season was still being served weeks later. The
roster cache was **68 hours old** and the schedule **93 hours old** eleven days
before kickoff. Completed seasons still cache indefinitely (they cannot change);
rosters, depth charts, the schedule and the current season's player weeks get a
few hours. A failed refetch falls back to the cached copy with a **loud warning**,
never silently.

## Book prices: what the API actually returns
Established 2026-08-30 by `scripts/probe_nfl_odds.py`, and worth stating precisely
because the previous answer was reached with a broken query.

- **bet365 IS visible** (bookmaker id 4), and all four markets exist as bet types:
  Player Passing Yards **210/336**, Player Rushing Yards **236/328**, Player
  Receiving Yards **266**, Anytime scorer **47** (the catalogue calls it "Anytime
  Goal Scorer"). Recorded in `odds.PLAYER_PROP_BETS` so nothing is guessed later.
- **No prices are served.** Asking for the week-1 opener's odds BY GAME ID returned
  zero records from bet365 and from every book.
- The earlier "no odds" finding filtered on a **`date` parameter the endpoint does
  not have** — the API was answering "The Date field do not exist." So the absence
  is only now properly established, by game id.
- The board publishes `odds.checked_at`, so "0 priced" cannot be confused with
  "never asked" — the same distinction the injury report draws between "not
  reported" and "confirmed fit".

## Known holes, stated on the board rather than hidden
**Lines are each player's own entering median, not a sportsbook
price**, so "over" means a better day than his typical one and the board makes NO
claim to beat a bookmaker's number. `ACTIVE_WITHIN_SEASONS` keeps long-retired
players off -- without it the week 1 board filled with Alfred Blue, C.J. Anderson
and Colt McCoy, each carried forward from a final season years back.

## NFL locking and grading (`nfl/picks.py`)
The NFL board was the last one in the repo publishing picks with **no log, no
freeze and no record**. It now freezes and grades like the others, reusing
`leagues.picks` for the freezing itself (`data-raw/nfl/picks_log.json`).

**Every kickoff on the board used to be midnight.** `nfl/data.py` read `gameday`
— a DATE — and nothing read `gametime`, which had been in the feed all along. A
pick frozen against midnight is frozen ~20 hours before a 20:20 kickoff, long
before the inactives report, or is frozen late and voided. Kickoffs are now built
through the `America/New_York` zone rather than a fixed -4, because the season
spans the November DST change and a fixed offset puts every late-season kickoff an
hour wrong in the direction that makes a lock late.

**Picks join on `game_id`**, the nflverse id (`2026_01_NE_SEA`), which encodes
season and week so settling needs no second lookup.

**Grading mirrors `nfl/features.py` exactly** — `touchdowns > 0` and
`yards > line`, strictly greater. The release gate measured the board against that
definition, so a record grading anything else would report on a product the gate
never validated. Lines are quoted on the half yard, so no result can land on one
and there is no push to handle. **The line is frozen with the pick**: it is the
player's own entering median and moves week to week, so grading against a later
line would settle a bet nobody made.

**A tie is VOID, not a loss.** The gate scores a tie 0.5 — half a win to each side,
which is what it is — and a win/loss record has no half. Scoring it a loss would
understate the model against its own measurement; a moneyline pushes for the same
reason.

**A prop settles only where the player feed holds his OWN TEAM in that week**
(`covered_games`). This is the 0-18 guard, ported: the soccer board once graded 18
picks as losses against a feed that had published nothing for the season, and
every one of those losses was fabricated. Where the feed is silent the pick stays
PENDING. nflverse also 404s `stats_player_week_{season}` until a season starts, so
`grading_stats` degrades to an EMPTY frame rather than to something that looks
like coverage.

Week 1 has not been played, so none of this can be proven by watching the live
board. `tests/nfl/test_picks.py` carries the load.

## Roster verification
`python -m scripts.sync_rosters` snapshots every current 2026-27 club and player
from **API-Football** into `data-raw/leagues/rosters.json`.

**THE ESPN FALLBACK IS GONE, and removing it made the sync more honest rather
than less robust.** It could not have rescued anything: ESPN answers **403 to
every request**, for all five leagues, on a plain call. A fallback that cannot run
is not redundancy, it is a false sense of it — and it hid the real state, which
was that rosters had simply stopped refreshing. It also mixed two id schemas, so a
league it "rescued" silently changed identity space mid-file; that is why those
leagues were deliberately never stamped `_league_verified_at`, which is a lot of
machinery around a source not trusted enough to stamp.

What replaced it:
- **A failed league keeps its previous snapshot**, unstamped, and is due again on
  the next run. There is nothing to fall over to, and nothing pretends otherwise.
- **If every league fails, the file is not rewritten at all** — rewriting it would
  move `_source` and the modification time while the squads are untouched, which
  is a file that looks freshly written and holds nothing new.
- **A missing `API_FOOTBALL_KEY` now fails loudly** (exit 1). It used to fall
  through to ESPN, so a run with no key looked like a successful refresh.
- **The retry moved, it was not dropped.** ESPN's fetcher retried four times with
  backoff and was the only retry in the roster path; `api_football.Client` now
  retries transient transport failures — but never a *permanent* API error, since
  the endpoint answered and asking again just spends the allowance twice on the
  same "no".

**THE DAILY ALLOWANCE WAS NEVER THE BINDING CONSTRAINT — the per-minute one is.**
The first single-source refresh spent **94 requests of 7,500** and still had La
Liga come back with *"You have exceeded the limit of requests per minute of your
subscription"*, because 94 requests went out in about ten seconds. Nothing in the
client knew a per-minute ceiling existed.

So requests are now **paced** (`PACE_SECONDS`), and a rate-limit error is the one
API error treated as transient: it waits out the window once and retries, where a
bad league id still fails immediately. The new behaviour was visible in that same
run and worked — La Liga kept its previous snapshot, stayed due, and no other
league was affected.
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

---

# Champions League engine (`ucl/`, tab: UCL)

The 2026/27 league phase: 36 clubs, 8 matches each. Strengths are fitted from
**sixteen seasons of European results** (`data-raw/ucl/history.json`, one
API-Football request per season -- the past never changes, so only the current
season is refetched).

- `python -m scripts.sync_ucl_history` — history + the drawn fixture list.
- `python -m scripts.ucl_backtest` — the gate (walk-forward, RPS vs a home-
  advantage baseline). Writes `data-raw/ucl/backtest_report.json`.
- `python -m ucl.publish` — the board. Writes `data/ucl/board.json`.

**Qualifying rounds are included on purpose.** They are where the small clubs
actually play: excluding them would leave exactly the clubs with the least
evidence with even less.

**Ninety-minute scores only.** A tie settled in extra time or on penalties is a
DRAW as a football match; recording the winner's scoreline would teach the model
these clubs score more than they do.

**Seeded vs fitted is published per fixture.** Two of the 36 have no Champions
League history in the window and are seeded at the weakest sides' strength; seven
more carry a thin-history flag. A number fitted from 182 matches and one seeded
from none render identically unless the board says which is which, and the second
is the one a reader would most want to discount.

## UCL locking and grading
The board published predictions for weeks with **no picks log, no freeze and no
record** — nothing stopped the displayed pick moving after kickoff. It now uses
`leagues.picks` wholesale (`data-raw/ucl/picks_log.json`), rather than a second
implementation of rules that are not football-specific and would be a second place
for the record to drift.

- **Joined on the API fixture id, never on club names.** The board renders
  "Internazionale" where the draw list says "Inter Milan"; a name join settles a
  frozen bet against whichever club the string happened to match.
- **A fixture with no id or no kickoff is NOT locked.** `sync_ucl_history` now
  carries the full kickoff timestamp (it previously truncated to a bare date). A
  pick frozen against a *guessed* kickoff is either frozen hours early, before any
  team news, or marked tainted and voided — the La Liga failure of 2026-08-15.
- **Grading sweeps the LOG, not the board.** `upcoming_fixtures()` publishes only
  unplayed games, so a match leaves the board the moment it finishes. Grading
  driven off the board would skip every match on the one run that could have
  graded it, and the record would read 0-0 forever. There is a regression test.
- A result that never arrives stays **pending, never wrong** — the same rule the
  soccer board learned when Understat's silence published 18 fabricated losses.

**The league phase opens 8 September and the feed has published no fixtures yet**,
so the board shows 0 matches and the record 0-0. The machinery therefore cannot be
proven by watching it, and is covered by `tests/ucl/test_picks.py` instead.

NFL now locks and grades too — see the NFL section above.

---

# NBA engine (`nba/`, tab: NBA)

A third sport. The tab is live and shows **evidence, not picks**: no fixture feed
is wired in yet, so there is nothing to project onto, and `status: evidence_only`
plus a banner say so before any number appears. That was the middle path between a
tab that looks live and does nothing — which this codebase treats as worse than no
tab — and hiding fifteen seasons of finished validation.

`tests/leagues/test_sidebar_props.py` holds the line: NBA sidebar links may exist
exactly when `data/nba/board.json` does. When the fixture feed arrives, `games`
and `props` fill in and nothing about the payload's shape changes.

- `python -m scripts.nba_backtest` — the gate. Writes `data-raw/nba/backtest_report.json`.
- `python -m nba.publish` — the board. Writes `data/nba/board.json`.
- `--props-only` reuses the stored team-winner result instead of repeating a
  240-combination Elo grid search that takes most of an hour.

## Data: stats.nba.com, one request per season
`leaguegamelog` returns an entire season at once — 26,651 player rows, 2,460 team
rows — so **fifteen seasons cost thirty requests**. It covers the current season
and reaches back two decades.

**The obvious alternative was checked and rejected.** hoopR's flat-file releases
are the NBA's nflverse: free, fast, and they **stop at season 2023**. They cannot
see 2025-26 at all. That was found by reading the file listing before writing any
engine, not after.

The headers are not decoration — stats.nba.com refuses a bare request and returns
an empty body rather than an error, which reads exactly like "no data for that
season".

## Three things the data forced
- **Neutral-venue games were silently dropped.** Five 2025-26 games have BOTH
  rows reading "@" (`DAL @ DET` and `DET @ DAL`) because neither side owns the
  court. The pairing lost them — and only in the current season, so the loss would
  have grown every year. They are kept, flagged `neutral`, and get **no home edge**.
- **The line offset had to be measured per market.** Adding +0.5 to a small
  INTEGER median means "over" needs median plus one, which pushed rebounds,
  assists and threes to a 36-40% base rate. Measured over 30,561 rows: +0.5 for
  points, -0.5 for the three count markets. Base rates became 0.494 / 0.487 /
  0.434 / 0.565. Assists and threes are not 0.50 and cannot be — a half-point line
  cannot straddle a small integer, and that is stated rather than tuned away.
- **`PropModel` and the Elo were NFL-shaped**, sorting on `week` and reading a
  `location` string. Both were generalised (date ordering, boolean `neutral`)
  rather than fabricating a fake `week` column. NFL frames still carry `week`, so
  their behaviour is unchanged and all 125 NFL tests pass.

## What is measured: 15 seasons, 11 scored
2011-12 to 2025-26 load; 2015-16 to 2025-26 are scored. The first four are burn-in
and never graded.

| Market | n | Brier | Baseline | Accuracy | ECE | Released |
|---|---|---|---|---|---|---|
| team_winner | 13,209 | 0.2158 | 0.2458 | 65.4% | — | **yes** |
| rebounds | 173,213 | 0.2225 | 0.2310 | 64.0% | 0.011 | **yes** |
| assists | 173,213 | 0.1925 | 0.2111 | 70.8% | 0.031 | **yes** |
| threes | 173,213 | 0.1858 | 0.2088 | 72.1% | 0.022 | **yes** |
| points | 173,213 | 0.2300 | 0.2311 | 61.5% | 0.063 | **WITHHELD** |

**Points is withheld** and the gate is right to: it lost to the baseline in
2023, 2024 AND 2025, and its ECE of 0.063 is above the bar. Only 2026 recovered.

**Team winner beats home court in every one of the eleven seasons**, which is a
stronger baseline in basketball than in football — the home side wins 54-59%
outright.

## THE PROP HEADLINE NUMBERS ARE FLATTERED BY THE FLOOR
`MIN_LINE` floors a line too low to quote. Measured, that floor **binds on most
rows**, so for those the "line" is a CONSTANT rather than the player's own median,
and the question becomes "will a rotation player clear a fixed number" — far more
predictable than a balanced prop.

| market | at floor | accuracy overall | accuracy where the line IS his median |
|---|---|---|---|
| points | 27% | 61.5% | 61.7% |
| rebounds | 64% | 64.0% | 63.7% |
| assists | **79%** | 70.8% | **64.9%** |
| threes | **80%** | 72.1% | **64.6%** |

The edge over the baseline shrinks with it: assists from 0.0186 Brier to **0.0031**,
threes from 0.0230 to **0.0030**. So the model has modest real skill on a
book-shaped line, and the headline overstates it. `backtest_report.json` carries
`floor_share` and an `above_floor` block on every market so this is published
rather than discovered later — the same discipline as the six-scores board, which
says plainly that it buys variety and not accuracy.

Nothing here is a claim to beat a bookmaker: no NBA prices have been fetched, and
the lines are the engine's own.

## The NBA tab shows both numbers, and shows the failure
Each market renders its headline accuracy AND its above-the-floor accuracy, plus
what share of its lines sit at the quotable floor. Points appears too, marked
**withheld** with the gate's reason on hover — hiding a market because it failed
would be the dishonest option, and the sidebar links to it for the same reason.
The per-market links genuinely filter the tab rather than labelling an unfiltered
page.

## Data JSON validation
`scripts/validate_data_json.py` parses **every** JSON file the site depends on --
the hand/agent-edited inputs (`data-raw/leagues/transfers.json`, `news.json`,
...) AND the published payloads the browser actually fetches (`data/leagues/*.json`,
`data/nfl/board.json`, `data/ucl/board.json`) -- and reports the exact
file/line/column on a parse error.

**The payloads were excluded on the reasoning that `json.dump` output "can't be
malformed this way", and that reasoning was wrong.** It describes the WRITER and
says nothing about what happens to a file afterwards. On 2026-08-26 a task
committed an unresolved autostash conflict and all eight payloads reached the live
site containing raw `<<<<<<<` markers -- every board on the page empty. The
validator ran that morning and passed, because it was not looking at those files.
It now checks them, and reports a conflict marker BY NAME rather than as the
"Expecting property name" JSON error that describes only the symptom. Runs FIRST in
`leagues.yml`, before the slow model fetch, so a bad file fails in seconds
instead of after an hour. `scripts/hooks/pre-commit` (install via
`scripts/hooks/install.sh`) runs the same check on the STAGED blob before a
local commit, so the class of bug that froze the pipeline for a full day on
2026-08-02 (one missing comma in transfers.json) can't happen again -- caught
at commit time, not discovered by a failed run hours later.

## Results update without the model (`scripts/refresh_results.py`)
Grading a played fixture and redrawing the table needs no model — the pick was
frozen days ago and the score is a fact. But both only happened inside
`leagues.publish`, which refits four leagues and takes ~18 minutes, so the visible
results were only ever as fresh as the last full refresh. On 2026-08-29 that left
the boards **four hours stale with about fifteen finished games missing**. It also
meant results were lost whenever the heavy job failed (21% of scheduled runs) or
stood down for a fresher run.

The fast path runs in the `lock` job on every trigger, in seconds, and updates
only what a score changes: grades frozen picks, redraws `standings`, recomputes
`unrecorded`, and drops finished fixtures from the upcoming list. **Predictions,
props, parlays and the projected table are left exactly as the model computed
them** — a finished match does not change what the model thinks about the next one.

- **It cannot invent a pick.** It grades the frozen log; a fixture with no entry
  stays out of the record and is reported under `unrecorded`.
- **It refuses to regress.** `fetch_fixtures` silently falls back to a snapshot
  when the feed times out (5 hours old in the run that prompted this) and returns
  it with no flag saying so. `publish` has `sanity_check` for that; this path had
  nothing and would have overwritten a 14-match table with a stale 12-match one. A
  season's played count only goes up, so a decrease is proof of stale input and the
  league is skipped.
- **`leagues/standings.py`** holds `actual_standings` and `unrecorded_fixtures`
  precisely so this can run on a **pandas-only** environment; importing them from
  `publish` would drag in penaltyblog, scipy, sklearn and soccerdata. `publish`
  re-exports them, so there is still one implementation. Same reasoning that put
  `LOCK_WINDOW_HOURS` in `config.py`.
- The `lock` job therefore **does** trigger Render now, but only when a board file
  actually changed. A freeze alone changes the record, not the page.

## The scheduled-run ration
Measured, not assumed. `0,15,30,45 11-22` asked for 48 slots a day and was honoured
55-70% (26-34 runs) until 2026-08-26. Adding `lock.yml`'s `*/10` took the repo's
total ask to ~130 slots and **everything** collapsed to 3-7 runs a day. That is far
worse than proportional, which is what makes over-asking the better explanation
than sampling.

The matchday cron is now **`17,47 11-22`** — 24 slots, both minutes away from the
hour boundary (GitHub's docs name the start of every hour as a high-load window;
`:00` and `:30` were the two worst available). **This is a hypothesis under test.**
If the ration is absolute rather than a penalty, fewer slots simply means fewer
runs — check `scripts/measure_lock_reliability.py`, which tracks runs per day
repo-wide, before concluding either way. The cost of being wrong is much lower now
that every run refreshes results on the fast path.

## The lock window adapts to how often the locker actually runs
`leagues/lockwindow.py`. `LOCK_WINDOW_HOURS` is the FLOOR; the window in force
widens to cover the gap since the last locking run, capped at `MAX_WINDOW_HOURS`
(12).

**This exists because a fixed window plus a sparse scheduler produces a BIASED
RECORD, not just a thin one.** A pick enters the record only if a locking run
happens before kickoff. On 2026-08-28 the runs were at 15:23 and 21:18, and every
fixture kicking off between 18:30 and 19:30 fell in the gap — **Bayern Munich v
Stuttgart, Lille v Paris SG, Alaves v Villarreal, Crystal Palace v Manchester
City**, plus Liverpool v Nottingham Forest the next morning. All were shown on the
board with a pick, all were played, and none left any entry: not graded, not void,
gone. The published record read 5-4 while silently omitting two played PL games.

That is a sample selected by GitHub's scheduler rather than by anything about
football, which is why it matters more than the missing count suggests.

- **What widening trades.** A pick frozen six hours out has seen less team news
  than one frozen at two. It is still frozen strictly before kickoff, so it is
  still honest — `LATE_LOCK_HOURS` stays 0.0 and nothing here relaxes it. The cost
  is measured and small (a confirmed XI moved Arsenal 77.4% → 77.1%); the
  alternative is losing the fixture from the record entirely. **Freezing early is a
  worse pick; freezing never is a worse record.**
- **The cap** stops the first run back from a multi-day outage freezing a whole
  matchweek at once on stale numbers.
- Every locking path writes the heartbeat — the fast locker AND `publish` — because
  what matters is the gap between runs that *could* have frozen something.
- A missing or corrupt heartbeat degrades to the plain floor, never wider.

## `unrecorded`: played fixtures with no frozen pick
Published per league and shown on the Grades tab. The adaptive window narrows this
set; publishing it is what stops the remainder being invisible. A reader can see
5-4 **and** see which played games are not in it. A record that hides them would
describe only the fixtures that happened to kick off near a run.

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

**Its cadence is NOT what protects the pick lock.** Freezing a pick is the
`lock` job at the top of `.github/workflows/leagues.yml` (`scripts/lock_picks.py`),
which runs before the publish on every trigger of that workflow. This local task
has its own, much wider gate — fixtures 45 minutes to 3 hours out — and does not
affect locking at all.

## Locking (`scripts/lock_picks.py`, the `lock` job in `leagues.yml`)
**Freezing a pick needs no model, only the board already published.** That is why
it runs as its own fast JOB ahead of the publish, rather than inside it. Locking
used to happen only inside `leagues.publish`,
which refits four league models and needs ~10 minutes to reach the lock step; add
GitHub's own scheduling delay — a MEDIAN of 8 minutes after the cron slot,
measured over 109 runs, 90th percentile 14 — and a nominal 18:45 run froze a pick
at ~19:05. For a 19:00 kickoff that is a late lock, which taints it, which VOIDS
it. Four La Liga fixtures, eight player picks and thirty-seven parlays went that
way in a fortnight; every one was a good pick. 29% of scheduled runs also fail
outright, and on 2026-08-26 GitHub fired nothing at all between 17:30 and 19:09.

The locker reads JSON, compares timestamps, writes JSON — seconds, not minutes —
so it finishes in about thirty seconds and the pick is frozen within a minute of
the trigger instead of ten minutes into a refit. It installs **pandas only**:
`LOCK_WINDOW_HOURS` lives in `leagues/config.py` (pure stdlib) precisely so this
job need not import penaltyblog/scipy/sklearn to learn one float.

**It freezes the NFL board too.** The NFL board cannot lock itself in time: its own
workflow publishes at 09:00 and 16:00 UTC while the Sunday slate kicks off at
17:00, 20:05 and 00:20. Only the first ever falls inside the window from a publish
run, so every late game — Sunday night football included — would reach kickoff
unfrozen and then be frozen late by the next morning's run, tainted, and voided.
`lock_nfl` runs `nfl.picks` in lock-only mode: no results, no player feed, no
network. It is wrapped in its own try/except so a missing NFL dependency can never
take soccer locking down with it.

- It **only ever freezes what was already published**. It computes nothing, so it
  cannot introduce a number the board did not show.
- Idempotent: `lock_pick`/`lock_prop` no-op on a second call for the same key, so
  running on every trigger can never move `locked_at` forward and turn a
  well-timed lock into a late one.
- It honours `time_suspect` (now published on each match for this purpose) and
  never locks a fixture that has already kicked off.
- **No Render deploy.** Locking changes the RECORD, not the board; deploying would
  spend build minutes to change nothing a reader sees — the exact waste that
  exhausted the Render spend limit on 2026-08-22.

`publish` still locks too, as a safety net on the same window. **`LATE_LOCK_HOURS`
stays at 0.0 and must not be relaxed**: tolerating a late lock would hide the
symptom without making the pick honest, and the record's whole value is that every
entry demonstrably was made before kickoff.

**ADDING A SECOND SCHEDULED WORKFLOW COST MORE LOCKING THAN IT BOUGHT.** `lock.yml`
was created on 2026-08-27 with its own `*/10 11-23` cron. Measured two days later
(`scripts/measure_lock_reliability.py` -> `data-raw/lock_reliability.json`), it
fired **3 of 97 due slots, 3.1%, with zero failures** -- GitHub simply never
started the other 94. That looked like one greedy schedule being throttled. The
repo-wide count showed the real effect:

| date | total scheduled runs | |
|---|---|---|
| 2026-08-23 | 34 | |
| 2026-08-25 | 28 | |
| 2026-08-26 | 16 | |
| 2026-08-27 | **6** | lock.yml added |
| 2026-08-28 | 7 | |
| 2026-08-29 | 3 | |

`leagues.yml` — which publishes the boards **and** freezes their picks — fell from
27-28 runs a day to 4. Whether this is a per-repo ration or GitHub-wide load
cannot be told apart from here, but the response is the same: **asking for more
scheduled runs produced fewer.**

So `lock.yml`'s cron is **gone** (the workflow stays dispatchable for a manual
freeze) and locking now rides `leagues.yml` as a fast `lock` **job** that runs
BEFORE the publish job, on the trigger that actually fires. It installs pandas
only, freezes soccer *and* NFL, commits the picks logs, and never deploys.

`LOCK_WINDOW_HOURS` **stays at 2.0** until the consolidated setup has been
measured over several days. The floor for that constant is the measured
`worst_gap_hours`; re-run the measurement before revisiting it.

**It ABORTS on a dirty working tree** (SKILL.md STEP 0.5) and must never be
"fixed" by stashing. It once improvised `git stash` to get a blocked
`git pull --ff-only` through and silently swallowed two hours of uncommitted
engine work. Worse, it runs `leagues.publish` and `deploy.py`, so on a dirty tree
it would ship someone's half-finished model to the live site. If you are working
in this checkout with uncommitted changes, expect the sweep to skip and say so --
that is the guard doing its job, not a failure.

Both abort rather than deploy if a fetch fails — never ship a stale-but-fresh-
looking file.
