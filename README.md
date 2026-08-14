# Henry's Match Engine

A statistical football predictor for the Premier League, La Liga, Bundesliga, and Ligue 1 — match outcomes, player props, and model-built parlays, all graded against real results with a frozen, no-hindsight pick log.

**Live site:** [worldcup-nnxg.onrender.com](https://worldcup-nnxg.onrender.com)

Also included: an archived World Cup 2026 tournament predictor, kept exactly as it finished (final record 64-26 of 90, 71%, champion Spain).

## What this is

A Dixon-Coles / xG-blended statistical model — not machine learning, not a black box. Team strengths are fitted from five seasons of results and expected-goals data, time-weighted so recent form matters more, and shrunk toward the league mean so a small sample never produces an extreme number. Promoted clubs are seeded from their actual second-tier form rather than a generic fallback.

Every published probability is graded honestly:
- A pick **locks 45 minutes before kickoff** and is never rewritten, even if the model would call it differently by full time. The published record reflects the genuine pre-match call, not a hindsight re-score.
- The match model is **walk-forward backtested against de-vigged closing bookmaker odds** — the honest benchmark for "does this actually know something." Right now it doesn't reliably beat the closing line, and that's stated plainly rather than hidden. A [calibration chart](#the-boards) tracks whether a stated confidence (e.g. 70%) actually lands that often once results come in.
- Player props are graded from actual shot events, not season averages — a market cannot be settled by data it wasn't measured against.

## The boards

| Board | What it is |
|---|---|
| **Best Picks** | Cross-league match-winner picks at ≥65% model confidence |
| **Player Picks** | Anytime goalscorer, 2+ shot attempts, 1+ shot on target — graded per market, since a 45% goalscorer pick and an 80% shots pick are both near their market's ceiling |
| **Parlays** | Model-built accumulators, one leg per match so the combined probability is a genuine product of independent legs, not a guess |
| **Tables** | Actual league standings (points, goal difference) from results played so far |
| **Grades** | The full track record per market, plus a calibration chart comparing stated confidence to real hit rate |

## Honest limitations

- **The model does not currently beat the closing market.** A blend of the model with the market's own line puts zero weight on the model — verified, published, not spun.
- **Bundesliga player props cannot be graded.** The upstream shot-event feed crashes for that league, so its player picks publish but are excluded from the record rather than faked or silently dropped.
- **The props gate validates rate estimation, not the probabilities on the board.** `leagues/props_backtest.py` tests whether the per-90 rates are estimated well, scored as season-total MAE against a baseline — and it is handed each player's *actual* minutes. The board publishes per-match probabilities, which are those rates run through a minutes projection, a penalty adjustment and a rescale to the match model's lambda. So the gate is blind to minutes-projection error, which is a dominant driver of live prop accuracy, and its bar is `mae < baseline_mae` on two metrics with no minimum sample, no calibration check and no significance test. **The live record is the first real test of the published probabilities.** This is a known gap, not an oversight: Understat cannot support an honest per-match anytime-scorer curve, so the alternative would be a gate that looks rigorous and isn't.
- **Free data sources have real limits.** Roster and transfer data come from free feeds (ESPN, Understat) with documented reconciliation logic to avoid mis-attributing a departed player — see `leagues/players.py` for how thin/incomplete sources are handled without guessing.

## How it's built

- **Match model**: Dixon-Coles bivariate Poisson (`leagues/model.py`), fit on results + xG from football-data.co.uk and Understat, with empirical-Bayes shrinkage and a home-advantage term.
- **Promoted clubs**: seeded from calibrated second-tier form (`leagues/second_tier.py`) rather than a single third-party rating source.
- **Player props**: per-90 rates shrunk toward positional priors, rescaled to the match model's own expected goals so the player numbers can never disagree with the team prediction (`leagues/props.py`).
- **Season simulation**: Monte Carlo over the fitted model for projected final standings, title/top-4/relegation odds (`leagues/sim.py`).
- **Parlays**: combines the model's own already-published pick probabilities, one leg per match for independence (`leagues/parlays.py`).
- **Validation**: walk-forward backtest vs. de-vigged closing odds (`leagues/backtest.py`, `leagues/tune.py`). The backtest runs the model that ships, promoted-club priors included, so the score covers the whole fixture list rather than quietly excluding the hardest games. Parameters are swept on the earlier seasons and the winner is re-scored on a **held-out final season**, and a challenger only replaces the shipped config if a paired 95% bootstrap interval puts it entirely ahead — rejected candidates are written to the report, not discarded. The gate emits `release_policy.json`, which `publish.py` reads, so the shipped parameters and the evidence for them are produced by one run and cannot drift apart.
- **Publishing**: `python -m leagues.publish` fits the model, runs the simulation, builds props and parlays, locks picks inside the kickoff window, and writes the four league files plus the cross-league boards atomically.
- **CI**: a scheduled GitHub Actions pipeline republishes through the season, validates hand-edited data files before touching the model, and deploys to Render on every push.

## Running it locally

```bash
pip install -r requirements.txt
python -m leagues.publish PL          # one league, for quick iteration
python -m leagues.publish             # all four leagues
python -m pytest tests/leagues/ -q    # test suite
```

No API keys are required for the core pipeline; a free [API-Football](https://www.api-football.com/) key enables confirmed-lineup fetching (`API_FOOTBALL_KEY`) but the pipeline degrades gracefully without one.

## Stack

Pure Python (pandas, scipy, [penaltyblog](https://github.com/martineastwood/penaltyblog), [soccerdata](https://github.com/probberechts/soccerdata)) on the backend; a single-file, no-build-step vanilla JS/CSS frontend (`index.html`). No framework, no bundler.
