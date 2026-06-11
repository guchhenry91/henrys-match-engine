# World Cup 2026 Predictor

Static site (index.html + data/predictions.json) deployed on Render, predicting every WC26 group-stage match. Owner: John (guchhenry91).

## Layout
- `index.html` — UI, reads `data/predictions.json` only. No build step.
- `predict.py` — prediction engine (pure stdlib). Run `python predict.py` to regenerate `data/predictions.json` from `data-raw/`.
- `data-raw/schedule.json` — all 72 group matches (do not change ids).
- `data-raw/ratings.json` — Elo + FIFA per team (baseline; predict.py applies result-based Elo deltas itself — do not manually edit after tournament start).
- `data-raw/news.json` — per-team form, injuries, key players, headlines.
- `data-raw/results.json` — final scores keyed by match id as STRING: `{"5": {"home_goals":2,"away_goals":0}}`.

## Daily update procedure (run every morning)
1. Web-search final scores of all WC matches played yesterday (and any missed earlier); add them to `data-raw/results.json`.
2. Web-search overnight team news: injuries, suspensions, lineup news for teams playing TODAY and TOMORROW. Update those teams' entries in `data-raw/news.json` (update `form` strings with yesterday's results too, and set top-level `updated`).
3. Run `python predict.py`. Verify it prints "Wrote 72 matches" and no errors.
4. Commit all changes ("daily update YYYY-MM-DD: results + news") and push to main. Render auto-deploys.

Rules: never invent scores or injuries — only verified info. Team names must exactly match the names in schedule.json. Keep reasons concise.
