# Multi-League Football Predictor — Design Spec

**Date:** 2026-07-13
**Author:** John + Claude
**Status:** Draft for review

## 1. Goal

Extend the existing World Cup app (`C:\Users\John\worldcup`, live at worldcup-nnxg.onrender.com) into a general **football predictor** that forecasts the 2026-27 club season for four top-flight leagues, with a genuinely research-grade model — "every match carefully calculated, not random." Reuse the WC app's proven infrastructure (glass UI, `deploy.py`, locked-pick tracking, scheduled agents) and its Monte Carlo simulator.

**Success = per-match forecasts within a hair of the bookmakers** (target ~52–55% 1X2 accuracy, RPS ≈ 0.19–0.20, validated walk-forward vs de-vigged closing odds) **plus** season-long insight (title / top-4 / relegation odds).

## 2. Scope

**In (v1):**
- **4 leagues:** Premier League (E0, 20 teams), La Liga (SP1, 20), Bundesliga (D1, 18), Ligue 1 (F1, 18). All 2026-27 fixtures already released.
- Per-match: 1X2 %, predicted scoreline, confidence, **player props (top-3/team: anytime scorer, shots, shots on target)**, locked + graded pick.
- Per-league: projected final table + title / top-4 (European) / relegation odds via season Monte Carlo, refreshed weekly.
- Backtest report per league (accuracy, RPS, Brier vs market).
- Integrated into the existing app behind a **competition switcher** (World Cup stays as-is; leagues added).

**Out / deferred:**
- **Champions League & FA Cup** — deferred until their draws (CL league-phase ~late Aug; FA Cup round-by-round). Engine will be format-agnostic so they slot in later with only a config + fixtures.
- Second divisions, other leagues, live in-play, betting/staking advice.

## 3. Architecture

One **competition-agnostic engine**; each competition is a config + a dataset. New modules live alongside the WC files in the same repo.

```
worldcup/                      (repo root — becomes the multi-sport app)
  index.html                   extended: competition switcher, league views
  data/predictions.json        WC (unchanged)
  data/leagues/<lg>.json       per-league published output (PL, LALIGA, BUNDESLIGA, LIGUE1)
  predict.py                   WC engine (unchanged)
  deploy.py, status.py         reused as-is (deploy.py already atomic)
  leagues/                     NEW package
    __init__.py
    config.py                  per-league: code, teams, fixtures src, format, dates
    data.py                    acquisition (soccerdata + football-data.co.uk + ClubElo + football-data.org)
    model.py                   fit (penaltyblog Dixon-Coles), priors, calibration
    scorers.py                 player anytime-scorer model
    sim.py                     season Monte Carlo -> table + title/relegation odds
    backtest.py                walk-forward validation vs de-vigged odds
    run.py                     orchestrator: build one league's <lg>.json
    names.py                   cross-source team-name normalization map
  data-raw/leagues/<lg>/       cached raw data (results, xg, players, fixtures, elo)
```

**Dependencies (new):** `penaltyblog`, `soccerdata`, `pandas`, `scipy`, `numpy`. These are heavier than the WC's pure-stdlib engine, so the league engine runs **locally** to produce `data/leagues/<lg>.json`; the deployed site stays static (serves the JSON). A `requirements.txt` pins versions.

## 4. Data layer (`data.py`, `names.py`)

Accuracy-first, all free/authoritative:

| Data | Source | Access | Used for |
|---|---|---|---|
| 5 seasons results + shots + **closing odds** | football-data.co.uk (`mmz4281/{season}/{div}.csv`) | direct CSV / soccerdata `MatchHistory` | model fit + odds benchmark |
| Team & player **xG/npxG**, goals, assists, minutes, pens | **FBref (StatsBomb)** via soccerdata; **Understat** cross-check | soccerdata (cached) | strengths + scorer model |
| Cross-league / promoted-team strength prior | **ClubElo** (`http://api.clubelo.com/{date}`) | direct http CSV | priors + shrinkage |
| 2026-27 fixtures + **authoritative live results** | **football-data.org** free API (JSON, 10 req/min) | keyed REST | fixtures + result recording |

- **Name normalization** is the main integration cost: football-data.co.uk, FBref, Understat, ClubElo, football-data.org each spell clubs differently. `names.py` holds one canonical→source alias table per league; build once, unit-tested against all sources.
- **Caching:** soccerdata caches to disk; football-data.co.uk CSVs re-downloaded weekly. FBref is rate-limited/fragile → pull slowly, cache hard, weekly cadence only.
- **Authoritative results** from football-data.org replace the WC app's error-prone web-search recording (fixes the pen-shootout mistakes).
- **Paid-API seam:** `data.py` exposes an `injuries()` / `lineups()` interface returning empty by default; a future API-Football adapter can fill it without touching the model.

## 5. Match model (`model.py`) — the core

Built on **penaltyblog** (MIT, maintained through 2026) so we compose tested code, not hand-rolled MLE.

1. **Response variable:** for each team-match, `y = 0.75·xG + 0.25·actual_goals` (xG is the more repeatable signal; keep 25% actual to retain genuine finishing skill & robustness to xG-model error). Tunable 0.7–0.8 by CV.
2. **Fit `penaltyblog.models.DixonColesGoalModel`** on the last 5 seasons of blended goals, with **exponential time-decay weights** `w = exp(-ξ·days_ago)`, **ξ ≈ 0.003/day** (half-life ~230 days). ξ tuned per league by walk-forward RPS. Yields per-team attack/defence, home advantage, and the low-score `rho`.
3. **Cold-start priors + shrinkage:** promoted teams and low-data clubs are shrunk toward a prior built from **ClubElo (blended with prior-season strength)** — SPI-style. Prior weight decays as real 2026-27 games accumulate. This is the #1 robustness fix.
4. **Rating ensemble:** also compute **pi-ratings** (`penaltyblog.ratings.PiRating`, home/away-separated) and blend the Dixon-Coles 1X2 with the pi-ratings 1X2 (small, cheap accuracy/robustness gain + sanity check).
5. **Calibration:** apply **isotonic recalibration** of the blended 1X2 fitted on a held-out season. If/when odds are trusted, optionally blend toward de-vigged market probabilities.
6. **Output per fixture:** `p_home/p_draw/p_away`, most-likely scoreline (from `.grid`), confidence (from max prob), and the same **DRAW-pick rule** as WC (pick draw when no side > ~40%). Reuses WC pick semantics so the UI/tracking are identical.

Everything is `.fit()` once per league per update (seconds), then `.predict()` per fixture.

## 6. Player props model (`scorers.py`) — goals **and shots**

A real upgrade over the WC's `goals/apps × share`. Same machinery drives three player props: **anytime scorer**, **shots**, and **shots on target**.

**6a. Anytime scorer**
- **Per-90 rate** blending realized and expected: `rate90 = w·(npG/90s) + (1-w)·(npxG/90s)`, `w`≈0.6 for high-minute players, ~0.4 for low-sample. Uses **non-penalty** xG/goals.
- **Empirical-Bayes shrinkage** to position priors for new signings/few-minutes: `rate90 = (m·obs + K·prior_pos)/(m+K)`, K≈5–10 nineties; position priors (FW ~0.45, W ~0.25, AM ~0.20, DF ~0.05 g/90) from our own 5-season pool. Cross-league transfers adjusted by ClubElo league-strength ratio.
- **Season decay:** weight each historical 90 by `α^(seasons_ago)`, α≈0.7.
- **Penalties separate:** only the designated taker gets `λ_pen = ExpPens_team × 0.76`.
- **Tie to the match model:** rescale players so `Σλ_i = Λ_team` (fitted team goal expectation), then **anytime P = 1 − exp(−λ_i)**.

**6b. Shots & shots on target** (new — per the "player shots / shot attempts" requirement)
- **Expected shots per match:** `s_i = shots90_i × (exp_minutes_i/90) × opp_volume_adj`, where `shots90_i` is the player's decay-weighted, shrinkage-corrected **total shots per 90** (FBref `Sh`), and `opp_volume_adj` scales by how many shots the opponent typically concedes vs league average (a team is more/less likely to generate shots by matchup). Same empirical-Bayes shrinkage to **position shot priors** (FW ~2.5, W ~1.8, AM ~1.4, MF ~0.9, DF ~0.4 shots/90) for low-sample players.
- **Shots on target:** `sot_i = s_i × player_sot_ratio_i` (career shots-on-target ÷ shots, shrunk to a ~0.35 league prior).
- **Prop probabilities** (Poisson on the expected counts): **P(1+ shot) = 1−exp(−s_i)**, **P(2+ shots) = 1−exp(−s_i)(1+s_i)**, **P(1+ shot on target) = 1−exp(−sot_i)**. Also surface the **expected shot count** itself (e.g. "Saka — 3.1 shots, 78% 2+").
- Team's overall shot volume is anchored to the match model's expected goals (more expected goals ⇒ proportionally more expected shots via the league's goals-per-shot conversion), keeping shots and the scoreline internally consistent.

**Shared:** top-3 players/team per prop; penalty-taker flagged; **"DOUBT"** flag when injured/doubtful (reusing WC logic); everything backtested against historical player shot/goal logs where feasible.

## 7. Season simulation (`sim.py`)

Reuses the WC Monte Carlo pattern, pointed at a round-robin. Per league: for each of N=10,000 runs, sample every remaining fixture's scoreline from the fitted model, tally points (with real results locked in), build the final table with proper tie-breakers (league-specific: PL = GD then GF; La Liga = head-to-head; etc.), and record each team's finishing position. Aggregate → projected table, **title %, top-4/European %, relegation %**. Refreshed weekly as fixtures resolve.

## 8. Pick tracking & records

Identical to the WC's honest system: **lock each pick pre-kickoff** into `picks_log`, grade the frozen pick on the authoritative result, **void** any late-locked pick. Per-league record `{correct, wrong, total, void, by_confidence}` shown as a W–L chip. Because results come from football-data.org (authoritative), the pen-shootout / mis-record problems from the WC don't recur.

## 9. Backtest / credibility (`backtest.py`)

Before trusting picks, **walk-forward** validate: train on matches up to date t, predict the next round, roll forward across the last 1–2 seasons. Report per league: **1X2 accuracy, RPS, Brier, log-loss**, and the same metrics for **de-vigged bookmaker closing odds** as the benchmark, plus a calibration curve. A "Model Performance" view surfaces these live (this also delivers the WC stats-page you approved). Honesty rule: if a league backtests poorly, we say so rather than ship vibes.

## 10. UI (`index.html`)

- **Competition switcher** (top-level): World Cup · Premier League · La Liga · Bundesliga · Ligue 1. Each league loads its `data/leagues/<lg>.json`.
- **Per-league views** (reusing WC card components): **Matchweek** (this round's fixtures with picks + scorers), **Table** (projected final table + title/relegation/Euro odds bars), **Fixtures/Results** (full season, ✓/✗ graded), **Form/Team sheets**, **Performance** (backtest + live record).
- Same gold + team-colour glass aesthetic; team colours extended to ~90 clubs (a colour/crest map — a build task).

## 11. Ops

- **Weekly job** (extend scheduled tasks or a new one): refresh current-season data (soccerdata + football-data.co.uk), pull authoritative results (football-data.org), re-fit each league, re-sim, regenerate `data/leagues/<lg>.json`, publish via `deploy.py`.
- **Match-day job:** record just-finished results (authoritative), re-grade, redeploy.
- The league engine needs the Python deps installed locally (`pip install -r requirements.txt`), so these run on the local machine (like the WC tasks), not on Render.

## 12. Risks & mitigations

- **FBref rate-limiting** → cache hard, weekly cadence, Understat as backup xG.
- **Cross-source name mismatches** → `names.py` mapping table, unit-tested; fail loudly on an unmapped team.
- **Cold-start (no 2026-27 games early season)** → ClubElo + prior-season priors carry the model until real games accumulate; backtest confirms early-season behaviour.
- **Transfer window churn** (open to ~Sep 1) → squads/ratings update weekly; player priors shrink new signings sensibly.
- **Scope creep** → v1 is 4 leagues only; CL/FA Cup explicitly deferred.
- **Heavier stack** (pandas/scipy vs WC stdlib) → isolated in `leagues/`; WC engine untouched; site stays static.

## 13. Build phases (for the implementation plan)

1. **Data layer + name maps** — pull & cache 5 seasons for all 4 leagues; verify against known tables.
2. **Match model + backtest** — fit Dixon-Coles + xG + priors + calibration; walk-forward validate each league; tune ξ and xG weight. **Gate: must beat baseline & approach market before proceeding.**
3. **Player props model** — anytime scorer **+ shots + shots on target**.
4. **Season simulation.**
5. **Orchestrator → `data/leagues/<lg>.json`** (contract mirrors WC `predictions.json` where possible).
6. **UI: competition switcher + league views + performance page.**
7. **Pick tracking + weekly/match-day ops + deploy.**
8. **Backfill the WC "Model Performance" view** (same component).

## 14. Open decisions (resolved)

- Data: **free, accuracy-first** (football-data.co.uk + FBref/StatsBomb + Understat + ClubElo + football-data.org). No paid API in v1; injury-feed seam left for later.
- Scope: **top flight only**, 4 leagues, integrated into the app, CL/FA Cup deferred to their draws.
- Model: **time-weighted Dixon-Coles on xG-blended goals + priors + pi-rating ensemble + isotonic calibration** (via penaltyblog), backtested vs market.
