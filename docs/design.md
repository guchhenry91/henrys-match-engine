# World Cup 2026 Predictor — Design

Date: 2026-06-11 · Approved by John

## Purpose
A daily-updated web app predicting every 2026 World Cup group-stage match (72 matches, 48 teams, 12 groups), so John can pick winners each day.

## Architecture
- **Static site** deployed on Render, auto-deployed from GitHub (`guchhenry91/worldcup`).
- `index.html` — single-page Apple liquid-glass UI (vanilla JS). Views: Today, Schedule, Groups.
- `data/predictions.json` — single data file the UI reads: matches + predictions + team intel + standings.
- `data-raw/` — source data: `schedule.json`, `ratings.json` (Elo + FIFA), `news.json`, `results.json`.
- `predict.py` — prediction engine. Reads data-raw, writes `data/predictions.json`.

## Prediction model
1. Base: World Football Elo ratings, blended with FIFA ranking points.
2. Home advantage: +80 Elo for a host nation playing in its own country.
3. News adjustment: −15 Elo per key player out (cap −60), −8 if doubtful.
4. Win/draw/win: Elo diff → expected goals via λ = base · 10^(±dr/600), Poisson grid → P(H)/P(D)/P(A) and most likely score.
5. Confidence 1–5 from max probability.
6. After each matchday: results feed Elo updates (K=60, goal-diff multiplier), picks get graded, accuracy shown in app.

## Daily automation
Scheduled cloud agent every morning (~7am ET): fetch yesterday's results + overnight injury/news, update data-raw, run `predict.py`, commit + push → Render redeploys.
