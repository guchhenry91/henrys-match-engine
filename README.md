# Henry's Match Engine

A multi-sport betting predictor: football (Premier League, La Liga, Bundesliga, Serie A, Ligue 1 and the Champions League), NFL, NBA and MLB. It makes team picks and player-prop picks, checks each one against the bookmaker's price, freezes it before kick-off and grades it against the real result. Every record on the site is built from those frozen picks, never from a re-score after the fact.

**Live site:** [henrys-match-engine.henrys-edge-engine.workers.dev](https://henrys-match-engine.henrys-edge-engine.workers.dev)

Also included: the archived World Cup 2026 predictor (`worldcup.html`), kept exactly as it finished (64-26 of 90, 71%, champion Spain).

## How a pick is made

Every sport goes through the same five steps.

1. **Forecast.** A statistical model forecasts the outcome: a team's chance to win, or a player's whole stat distribution (not just an average), so it can give a chance for any line.
2. **Combine with the bookmaker.** Bookmakers are usually better informed than the model, especially about playing time. So the chance the site shows is the **bookmaker's chance moved part of the way toward the model's**, at most 20% until real results prove the model deserves more. The model's own number is kept, but on its own it has tested as overconfident against realistic lines, so it is never the headline.
3. **Price check (bet365).** A pick is either:
   - a **value pick**: at least +3% expected return at bet365's price, with the model within 15 points of the bookmaker; or
   - a **proving pick**: the model is at least 2 points more confident than the bookmaker, but the price isn't good enough to bet. Proving picks are shown, tagged *not a bet*, and graded so each market can prove itself.
4. **Freeze.** Picks freeze before the start: soccer and NFL inside the lock window (2 hours, widening if the scheduler has been slow), NBA and MLB on the last run before tip-off or within 2 hours. A frozen pick is never rewritten, and nothing is frozen after kick-off.
5. **Grade and earn trust.** Graded picks are scored at the price they froze on, with closing-line value where available. A market only reaches the **Bet list** once its own record shows profit at bet365's prices. Until then its picks are tracking-only.

## What each model does

### Soccer (five leagues)
- **Match picks:** a Dixon-Coles model (`leagues/model.py`) fitted on results and expected goals from football-data.co.uk and Understat, weighted toward recent form and shrunk toward the league average. Promoted clubs are seeded from their real second-division season. Best Picks are match winners at 65%+.
- **Player picks** (anytime scorer, 2+ shot attempts, 1+ shot on target): a model of each player's chance to start or come on, plus his chance of the outcome if he plays, trained on Understat's per-match player data (`leagues/forecast.py`). It reads team news and the confirmed XI, and a player benched or missing from the XI is never frozen. Bars: 40% scorer, 62% shots and on target. Prices come from bet365 (scorer) and US books (shots).
- **Grading:** from Understat shot events, with API-Football as the fallback. A player who didn't play is void.

### Champions League
- **Match picks:** a model fitted on sixteen seasons of European results, checked against bet365's 1X2 odds and leaning on bet365 when they disagree (`ucl/market.py`).
- **Player props:** only for matches between two clubs from the big five leagues, priced from the players' league form (`ucl/props.py`), graded from API-Football and ESPN.

### NFL
- **Team picks:** an Elo rating model.
- **Player props** (receiving, rushing and passing yards; anytime TD): a forecast of each player's yards or touchdown chance from his role, snaps, usage, the betting spread and total, the opponent, weather and **teammates on the official injury report** (`nfl/forecast.py`). Players returning from missed games are scaled down by a measured correction (they were overrated by 17% on receiving yards). Lines are bookmakers' lines only.
- **Proving bar:** 60% on the model's own chance (owner's choice). In testing these picks landed 52-58%, so each one shows *"Tested: picks like this landed N%"*.
- **Anytime TD** is priced and logged but not published.

### NBA
- **Team picks:** an Elo rating model.
- **Player props** (points, rebounds, assists, threes): a forecast of each player's stat distribution from minutes, usage, pace, opponent and teammates out (`nba/forecast.py`). On a player's first game of a season it uses his longer-run level, not the last games before the summer.
- **Main board:** proving picks qualify on the blended chance at 55%+ (points 54%+, since points lines sit near 50/50; in testing those landed 54.4%, just above bet365's break-even). Value picks need 62%, or 55% on a near 50/50 line. One pick per player per game across rebounds, assists and threes; a points pick can sit beside it. Both rules were tested against a simulated sharp bookmaker; rebounds, assists and threes land at or above what they state, points about 2 points below.
- **Custom plays:** milestone bets ("25+ points", "10+ rebounds") on a team's three focal players, from the bookmakers' alternate lines. Overs only; the model must give 70%+, at odds of 1.40+.
- **Monster nights:** big-night long shots (30+ points, 12+ rebounds, 10+ assists or 5+ threes, well above the player's average). Listed only when the price is long enough to pay for the misses; at most 3 a night.
- **Playoffs** load and grade like the regular season, with a measured playoff correction: stars hold their points, role players are scaled down.
- **Rosters** come from current rosters (Sleeper, checked against box scores) and are cross-checked daily against ESPN.

### MLB
- **Team picks:** an Elo rating model.
- **Player props:** count models per market (`mlb/forecast.py`). Hits, RBIs and team runs are published when combined with the bookmaker. Strikeouts, hits + runs + RBIs and home runs are priced but held back.
- Postponed games replayed on another day are voided.

## What is and isn't proven

- **Soccer** has the longest live record. **NBA and NFL props are new**: they're tested on past seasons but have few or no graded live picks, so they're tracking-only until their records say otherwise.
- **There are no historical bookmaker prices for player props**, so backtests measure whether picks land at the rate they claim, not whether they make money. Profit only shows in the live record.
- **The model alone is overconfident against sharp lines** in the NBA and NFL. That's why the site shows the blended chance.
- **The Bet list stays small on purpose.** Most picks are proving picks until a market shows profit at bet365's prices over 200+ bets.

## How it runs

- **Hosting:** one Cloudflare Worker (`cloudflare/worker.js`) serves the site and queues jobs on a schedule. The jobs run in a container (`cloudflare/container/`) on the `tested` branch.
- **Data:** the live boards and pick logs are kept in Cloudflare D1, not in the repo. The repo's `data/` files are not the live ones.
- **Deploys:** a push to `main` deploys the site and Worker once tests pass (`.github/workflows/deploy-cloudflare.yml`).
- **Monitoring:** a health check (`scripts/health_check.py`) runs every few hours and alerts on the site and by Telegram. It covers stale boards, failed jobs, empty prop boards before games, picks frozen late and ungraded picks.

## Running it locally

```bash
pip install -r requirements.txt
python -m leagues.publish PL               # one soccer league
python -m nba.publish                      # NBA board
python -m scripts.nba_forecast_gate        # NBA model test (official injury report by default)
python -m scripts.nfl_forecast_gate        # NFL model test
python -m pytest tests/ -q                 # tests
```

API keys (`API_FOOTBALL_KEY`, `ODDS_API_KEY`) unlock lineups and bookmaker prices. Without them the pipelines publish less rather than guess. Paid API use has hard daily and monthly caps (see `CLAUDE.md`).

## Stack

Python (pandas, scikit-learn, scipy, [penaltyblog](https://github.com/martineastwood/penaltyblog), [soccerdata](https://github.com/probberechts/soccerdata)) for the models; a single-file vanilla JS/CSS frontend (`index.html`); Cloudflare Workers, Containers and D1 for hosting.
