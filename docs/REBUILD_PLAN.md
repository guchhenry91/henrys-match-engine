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
