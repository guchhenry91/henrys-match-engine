# The model rebuild (agreed with the owner 2026-10-11)

## Why
On 11 Oct 2026, four independent reviews (a sharp trader, a sportsbook pricing
modeller, a player-props bettor, and a measurement auditor) read the code and the
live record. Grading was correct (903 picks re-graded, 0 mismatches). The model was not.

- **Root cause (proved offline):** every prop model was trained on lines it set
  itself from the player's own median (`features.augment_lines`), with the line as an
  input. A bookmaker's line carries information (who starts, how long a pitcher goes,
  injury news); the model read it backwards. Trained on its own lines the strikeouts
  model was perfectly calibrated (75.1% said, 75.1% landed); against an informed book
  it said 70% and 60% landed. Soccer props, whose lines are fixed (1+, 2+), were the
  only calibrated props -- the same diagnosis.
- **Selection made it worse:** picks were chosen where the model disagreed most with
  the book, i.e. its mistakes (MLB non-HR overs: model 64.7%, book 56%, landed 48.9%).
- **Layers stacked over a broken core:** shrink, trust blending, market blend, gate,
  four tier schemes, a value chain that could never fire. Later layers undid earlier
  ones (trust re-blends from the raw number, discarding shrink). None fixed the core.
- Missing inputs: MLB opposing starter / lineup slot / workload; NFL game total and
  spread (deliberately excluded); NBA usage and pace.

Full reports: `/tmp` copies were session-only; the findings above are the record.

## Goal (the owner's words: "a proving winner")
**Wins first.** When the app says 70%, about 70% win, and it publishes only the
high-chance picks. Money (price, closing line) is recorded and shown, but does not
decide what is published. Anytime scorer STAYS and is improved, not removed.

## Design: forecast the game, not the line
1. **Distribution forecasts.** For every player and game, forecast the full
   distribution of the stat (counts: Poisson / negative binomial; yards and minutes:
   continuous). P(over ANY line) is read off the distribution. The book's line is never
   a model input.
2. **The inputs that drive the stat** (data checked 11 Oct):
   - NBA: minutes forecast first, then per-minute rates; usage, pace, opponent,
     rest/back-to-back, teammates out, current rosters (`nba/rosters.py`). Needs a
     full-column re-export of the stats.nba.com history (the committed files keep only
     MIN/PTS/REB/AST/FG3M).
   - NFL: snaps/targets/carries then yards; game total and spread (nflverse schedules
     carry spread_line/total_line for every past game).
   - MLB: opposing starting pitcher, lineup slot (`b_lp` in Retrosheet batting),
     pitcher workload, park. Test on postseason too.
   - Soccer anytime scorer: confirmed XI and expected minutes, penalty taker, team
     expected goals from the match model, opponent defence, bet365's scorer price.
3. **Combine with the market for WINNING, not just value.** The book's de-vigged
   price is an input to the final chance, with a weight learned per market from the
   record and allowed to go below zero. Agreement between forecast and market is where
   picks win most.
4. **One pipeline replaces the stacked layers** (shrink, trust, market blend, tier
   schemes, value->Bet list chain). Freeze picks only after lineups / inactives / XIs
   are confirmed. At most one pick per game per market, a few per sport per day.
5. **Measure everything:** archive every bookmaker line seen (real-line test data),
   closing price on every pick, best UK price. Old model picks are tracking-only while
   the rebuild runs.

## The gate (a market goes live only if all pass)
1. Calibrated on every line in walk-forward tests (ECE <= 0.03).
2. On REAL book lines (archive + logged picks) and an informed-book simulation:
   stated chance matches landed rate.
3. On the picks it would actually publish: landed rate >= stated (selection-aware).
4. Live: labelled "proving" until 300+ graded picks confirm the stated rates.

## Order
0. Foundation (line archive, tracking-only flag, closing prices, best price).
1. NBA -- opening night 20 Oct (tracking-only if the gate is not passed by then).
2. NFL -- about two weeks after.
3. Soccer props (anytime scorer rebuild) and match picks.
4. MLB -- over the winter, for spring 2027.

## Data sources
The Odds API (essential: lines/prices), API-Football (essential for soccer: lineups,
minutes, player stats, bet365 scorer prices), API-NFL (injuries + bet365 NFL props;
check whether nflverse injuries/depth charts can replace it before cancelling). Free:
nflverse, Retrosheet + MLB StatsAPI, stats.nba.com via the Worker proxy, Sleeper,
football-data.co.uk, Understat.

## Progress
- 2026-10-11: foundation done (line archive, tracking-only banner, plan). History pulled:
  NBA full (main), NFL / MLB / Understat per-match (branch `history-data`).
- 2026-10-11: NBA forecast model built, gated and wired into the live board
  (`nba/forecast_board.py`, engine `forecast-v1`). Combination weights come from a
  SIMULATED informed book; re-fit them on the real-line archive once ~300 NBA picks
  have graded.
- 2026-10-11: NBA teammates-out feature and real-line weight re-fit (300 graded) live.
- 2026-10-11: NFL forecast model built, gated and wired into the live board
  (`nfl/forecast.py`, `nfl/forecast_board.py`, `scripts/nfl_forecast_gate.py` ->
  `data-raw/nfl/forecast_report.json`). Findings, stated plainly:
  - The forecast alone is badly overconfident against a book that knows usage (said
    ~0.77, landed 0.52-0.62). Books set yardage lines at their MEDIAN (bet365's 2026
    fair over averages 0.500), so a free weight on the book's chance explodes; the
    combination now holds the book's weight at 1 and learns only the forecast's pull
    (`tracking/combine_fit.fit_offset`): a = 0.17 passing, 0.06 rushing, 0.18
    receiving, 0 TD.
  - Published at 62%+ in the 2022-25 walk-forward: passing 180 picks said 67% landed
    76%; receiving 270 said 65% landed 72%; rushing 5. On real 2026 bet365 lines
    (63/69/143) nothing clears the bar -- yardage main lines are near coin flips, and
    the board says so by publishing little.
  - Anytime TD: the forecast adds nothing over a book that knows touches (a = 0), so
    a TD pick is the book's own favourite that the forecast agrees with ("yes" only).
  - Next for NFL: an injury-report-based teammates-out feature, and the alternate
    ladders (higher-probability rungs) once the line archive has real NFL alt lines.
- 2026-10-11: NFL teammates-out (official pre-game report, Out only -- the live feed
  folds Doubtful into Questionable) gated A/B and ON: the forecast's pull rose
  (passing a 0.17 -> 0.21, receiving 0.18 -> 0.21), published picks roughly doubled
  and stayed calibrated (passing 290 said 67.6% landed 68.3%; receiving 544 said
  64.8% landed 65.4%). TD unchanged (a = 0).
- 2026-10-11: NFL alternate lines ON (`nfl/forecast_board.ALT_*`): the book's lower
  rungs, the highest one at 70%+ combined where the forecast is at least as sure as
  the book, for players without a main-line pick, max 2 a game. Simulated rungs:
  passing said 77.0% landed 83.0% (n=1,141), rushing 76.6/85.5 (2,733), receiving
  74.1/76.2 (6,566) -- at or above the stated rate. Real alt lines are being archived.
- 2026-10-11: SOCCER PROPS REBUILT (`leagues/forecast.py`, `leagues/forecast_board.py`,
  gate `scripts/soccer_props_gate.py` -> `data-raw/leagues/props_forecast_report.json`).
  Per player per match from Understat's per-match rows (`data-raw/leagues/understat_matches/`,
  kept current by `scripts/sync_understat_matches.py`, run by the board itself): a role
  model (start / bench / out) and per-market outcome models given the role, with
  penalty share, per-90 form, team and opponent xG form, home. Not appearing is a loss.
  Walk-forward 2022-25, five leagues, published (top 3 a match per market):
  anytime scorer at 40%+: 1,106 said 46.6% landed 47.7% (old model: said ~42%,
  landed 16-30%); 2+ shots at 62%+: 5,076 said 69.4% landed 70.3%; 1+ on target at
  62%+: 2,477 said 67.8% landed 68.4%. With the confirmed XI, about twice the picks,
  still calibrated. No after-the-fact recalibration; `record_rebuilt` on the board.
  Still to do for soccer: the bet365 scorer price as a second opinion once real
  graded prices accumulate (combine_fit), and a historical price check.
- 2026-10-11: SOCCER: a player who does not play is now VOID (owner's decision, as
  bet365 settles), decided from Understat's match lineups (`forecast.appearances`);
  with no lineup and no stat row a pick stays pending. Applies to the whole record,
  old picks included. The model's chances are now IF HE PLAYS, published only for a
  likely or confirmed starter. Re-gated: scorer 1,583 said 47.4% landed 46.6%; 2+
  shots 7,993 said 70.9% landed 71.8%; on target 4,549 said 68.9% landed 68.2%.
- 2026-10-11: SOCCER: bet365's anytime price is the scorer's second opinion. Every
  quoted player's price is logged with the forecast (`data-raw/leagues/priced_scorer_lines.json`),
  graded from the match rows (did not play = void), and at 300 graded the blend
  weights are fitted and adopted only if better out-of-fold
  (`data-raw/leagues/scorer_combine_weights.json`). Until then the forecast stands
  alone and bet365's chance is shown beside it. Only 15 past picks carried a price,
  too few to test a blend on history.
- 2026-10-12: MLB REBUILT (`mlb/forecast.py`, `mlb/forecast_board.py`, gate
  `scripts/mlb_forecast_gate.py` -> `data-raw/mlb/forecast_report.json`). One
  gradient-boosted classifier over the COUNT itself per market (0, 1, 2 ... hits; 0-12+
  strikeouts), so any line is read off the full distribution with no assumed shape.
  Inputs: per-PA rates (short/long), PAs and usual lineup slot, the opposing (probable)
  starter's K rate and depth, the opponent lineup's K rate vs starters, both teams' run
  form, park factors. Walk-forward 2021-25: calibrated at every line (ECE 0.005-0.008).
  Rules per market:
  - hits, RBI -- the model's own chance, the book must agree on the side (the simulated
    book did not know the opposing starter or park, so no blend weight from it is
    trusted): hits said 66.8% landed 66.3%; RBI 71.3/71.5.
  - strikeouts, team runs -- blended with the book (book weight 1): Ks said 67.8% landed
    67.5%; team runs 65.4/63.2.
  - H+R+RBI and home runs HELD: H+R+RBI landed 45% of 22 real lines stated at 66%
    (one day, but a red flag); HR "yes" never reaches 62%. Both stay priced and logged.
  Every priced line is logged (`data-raw/mlb/priced_lines.json`) and graded from box
  scores (did not start = void); at 300 graded a market's weights are re-fitted and,
  if adopted, it moves to the blended rule. The real-line check reads the live log
  copied to /tmp; it covers only the last regular-season day (~270 lines).

## Four-reviewer audit of the rebuild (2026-10-12) and what changed
Sharp trader, modeller, props trader and risk auditor (reports kept outside the repo)
agreed: the redesign is right and calibrated against real results, but the gates'
simulated books were weaker than real ones, real lines leaned the wrong way, and
"wins first" without a price check still loses money. The owner approved all fixes:
1. Trust the book by default (`combine_fit.prior`: book weight 1, forecast pull <= 0.2,
   no constant) in every sport until real lines are adopted.
2. Real lines are the proof: `tracking/proving.py` panel per market on every page
   ("unproven" until 300 graded at the stated rate). No real money before that.
3. Strict re-learning: adopt only a real (95% bound, grouped by game) gain whose
   would-publish picks landed at their stated rate; never change a publishing rule.
4. PRICE CHECK (`tracking/pricecheck.py`): published only if break-even or better at
   bet365's price (or an estimate); every pick shows break-even odds and expected return.
5. NFL alternate lines paused (rung margin floored at 10%); NFL TD and MLB strikeouts
   tracking-only.
6. Bugs: NFL weather no longer an input; NBA teammates-out off (trained after the fact);
   soccer next match per team only; void only with no stat row; 72h pending alert;
   tests gate the deploy; a missing library fails the job.
7. Football: 75% start chance, and new picks wait for the confirmed XI while runs are
   frequent. One pick per player per game everywhere.
8. Closing-line value shown where a close exists; a pre-start odds check for props
   would cost credits and awaits the owner's approval.
9. Missing inputs: NBA pre-game Elo gap (stand-in for the spread, which is not on
   hand historically) built and A/B-gated -- no change (PTS MAE 4.900 vs 4.901, the
   rest identical), so it stays off. NBA gate re-run with teammates-out off to match
   live. MLB lineup slot/handedness/bullpen and football team-goal inputs wait until
   the basics are proven on real lines.

## Third review (2026-10-12) and what changed
All four reviewers scored the rebuild 6/10: "ready to prove, not ready for real money".
Every point they raised is addressed:
1. **Shrink, don't stack.** The prior was sigmoid(a*logit(forecast) + 1*logit(book)), which
   ADDED the forecast's confidence to the book's: a forecast agreeing with a 72% book read
   75.6%, so every favourite looked like value. Weights are now (a, 1-a, c) -- the book
   moved a fraction a toward the forecast (`combine_fit.prior`, `shrink`, `fit_offset`).
   NBA and the soccer scorer re-fit in the same form.
2. **Price check** (`tracking/pricecheck.py`): +3% expected return (was 0), bet365's
   margin estimated at 7.5% on two-way props and 12% on one-sided markets (was 5%), and
   no pick where the forecast is more than 15 points from the book (`MAX_GAP`). One pick
   per player keeps the highest stated chance, not the highest return. Soccer is judged
   on the forecast pulled toward the market (`leagues/forecast_board.checked_p`), and that
   chance is what is published and frozen. CONSEQUENCE, stated plainly: with the default
   pull of 0.2 a model pick can rarely clear +3% at bet365, so boards will publish few
   or no model picks until a market earns more weight on real lines.
3. **Re-learning** judges the picks that would actually publish under the price check
   (`combine_fit._tail`), needs 100 of them (was 30), re-tests only after 50 new graded
   lines, and drops adopted weights that the book alone later beats. Every priced line
   (published or not) is logged and graded, including soccer shots / on target
   (`priced_shot_lines.json`); boards publish `learning` (n of 300 per market).
4. **Proving panel**: "on target" also needs the top of its interval to reach the stated
   rate; "bettable" (proven at bet365) needs profit >= 0 at the approved prices AND a
   profit lower bound above zero or closing-line value z >= 2. Profit and CLV are scored
   at the bet365 price each pick was approved at (`price_check` frozen with every pick).
5. **Closing check** compares a line move with the SAME book's line before the lock
   ("unknown" otherwise); CLV only on our exact line, at the bet365 price.
6. **Site honesty**: a board not built by the rebuilt model, or 26h+ stale, says so; an
   empty panel says "Unproven ... Do not stake"; old and rebuilt records are shown apart;
   price edges show their own record and stay at <= 0.5% stakes until 300+ beat the
   close; shown-but-never-frozen soccer player picks are counted. The health check fails
   a board without the rebuilt engine and rewrites health.json every run.
7. **Locks**: the lock job polls confirmed XIs and the locker never freezes a benched
   player (`lock_picks._xi_status`); NFL props drop players the news file ruled OUT after
   the board was built; NBA teammates-out reads the official report only, and is off for
   a run with no report today (`injury_report.fetch_today`, board `injury_report`).
8. **Ops**: jobs run the `tested` branch, moved by the deploy workflow only after the
   tests pass; NFL grading on the lock job refreshes `record_rebuilt`/`proving` and waits
   30 minutes between feed downloads; NFL publish no longer crashes on string player ids.

## Value and proving picks (owner's decision, 2026-10-12)
The strict price check left every board empty (MLB: 131 lines priced, best -3.5% at bet365),
so no market could ever build the record it needs. Boards now carry two kinds of pick
(`tracking/pricecheck.qualifies`):
- **value**: +3% expected return at bet365. Eligible for the Bet list once its market is proven.
- **proving**: not value at this price, but a real contender -- at the chance bar, the model
  at least 2 points SURER than the book's fair chance (`MIN_BACKING`) and within 15 points of
  it. Frozen and graded so the market can prove itself; shown "Proving pick - not a bet";
  never a value verdict or stake. A model merely equal to the book is the book's pick and
  is not published. Soccer picks with no price yet are proving picks on the model alone
  (it was checked on 1,583 real outcomes) and now freeze like any other.
- MLB RBI unders are no longer published (juiced side; live record 43-29 vs the book's 68%).
