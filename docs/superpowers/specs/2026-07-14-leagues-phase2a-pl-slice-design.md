# Multi-League Predictor — Phase 2a: Premier League Vertical Slice

**Date:** 2026-07-14
**Parent spec:** `docs/superpowers/specs/2026-07-13-multileague-predictor-design.md`
**Predecessor:** `docs/superpowers/plans/2026-07-13-leagues-phase1-data-and-model.md` (data layer + match model + backtest gate)

---

## 1. Goal

Take the Phase 1 match model end-to-end for **one league (Premier League)**: player props, season simulation, a published JSON file, a live page, pick-tracking plumbing, and the scheduled jobs that keep it fresh. Prove the whole chain on one league before plan 2b copies it four times.

**Why a vertical slice:** the JSON contract and the UI components are the parts most likely to be wrong, and they are cheapest to change before three more leagues depend on them.

**Deadline that matters:** the 2026-27 seasons kick off in **mid-August 2026** — roughly four weeks out. The slice must be picking Matchweek 1 by then, which is also when cold-start (no current-season matches) is at its worst.

---

## 2. Scope

**In:**
- Player props: anytime scorer, shots, shots on target (all three — they share one machine).
- Props backtest gate over 2025-26 (the only honest check available in the off-season).
- Season Monte Carlo: projected table, title / top-4 / relegation percentages.
- `data/leagues/pl.json` orchestrated by `publish.py`.
- `leagues.html` — a standalone PL page, linked to/from the WC app.
- Pick-lock + record plumbing (contract and honesty rules baked in, even though nothing grades until August).
- Weekly + match-day scheduled jobs.

**Out (deferred to plan 2b):**
- The other three leagues (La Liga, Bundesliga, Ligue 1).
- The unified in-app competition switcher inside `index.html`.
- Backfilling the "Model Performance" view onto the World Cup app.

**Hard constraint:** the World Cup app is live and in its most-watched week (final: **2026-07-19**). This slice does **not** touch `index.html`, `predict.py`, or the WC data files. The league page is a separate file; the two are joined by a plain link. Unification happens in 2b, after the WC winds down.

---

## 3. Gate (blocks everything below)

Phase 2a is **gated on the Phase 1 backtest passing for PL** under the current `model.py` (which carries two fixes made after the original gate report: scale-consistent xG blending, and prior shrinkage for thin-data teams).

Confirmed for PL under the fixed model:

| | accuracy | RPS | Brier |
|---|---|---|---|
| model | 53.7% | 0.2001 | 0.5838 |
| de-vigged market | 54.9% | 0.1939 | 0.5706 |
| **gap** | -1.2pp | **+0.0061** | +0.0132 |

Within the pass band (accuracy 50-55%, RPS 0.19-0.21, gap +0.005 to +0.02). A *negative* gap would have meant lookahead leakage, not brilliance.

**Props inherit this model.** A broken match model silently corrupts every player number, because fixture-specific information reaches the props only through the Σλ rescale (§5.6). If the match model regresses, stop.

---

## 4. Architecture

Six new modules in the existing `leagues/` package, plus one page. One direction, no cycles:

```
players.py ─┐
            ├─> model.py (Phase 1) ──> Λ per fixture ─┬─> props.py ─┐
dataset.py ─┘                                         │             ├─> publish.py ──> data/leagues/pl.json ──> leagues.html
                                                      └─> sim.py ───┤
                                                          picks.py ─┘
```

| Module | Responsibility | Network? |
|---|---|---|
| `players.py` | **One source only:** FBref per-player match logs via `soccerdata` (minutes, npG, npxG, shots, SOT), 5 seasons, names through `names.py`. Returns a clean frame. | yes (cached) |
| `props.py` | Rates → shrinkage → minutes → penalties → **Σλ rescale** → Poisson props. Pure function of frames. | **no** |
| `props_backtest.py` | Walk-forward 2025-26: scorer log-loss + calibration, shots/SOT MAE, vs a position-prior baseline. | no |
| `sim.py` | Season Monte Carlo, 10k runs, PL tie-breakers. | no |
| `picks.py` | Lock pre-kickoff, grade the frozen pick, void late locks. | no |
| `publish.py` | The **only** module that knows the JSON contract. Fits, sims, props, picks → writes the file, publishes via `deploy.py`. | via deploy |

`model.py` and `props.py` being pure functions of dataframes is what makes them testable without a network. `publish.py` owning the contract alone is what makes plan 2b a loop rather than a rewrite.

---

## 5. Player props model (`props.py`)

Worked through one player (Haaland, City home to Brentford):

1. **Raw:** per-appearance rows — minutes, npG, npxG, shots, SOT.
2. **Per-90 rate, blended:** `rate90 = w·(npG/90s) + (1-w)·(npxG/90s)`; `w ≈ 0.6` high-minutes, `≈ 0.4` low-sample (goals are near-pure noise at low volume). Seasons decay by `α ≈ 0.7` per season back.
3. **Empirical-Bayes shrinkage:** `rate90 = (m·obs + K·prior_pos) / (m + K)`, `K ≈ 5-10` nineties. Position priors (goals/90): FW ~0.45, W ~0.25, AM ~0.20, MF ~0.10, DF ~0.05. A 30-nineties player barely moves; a one-nineties player is dragged back to "generic forward." *Same bug class as the promoted-team problem fixed in `model.py`: a rating fitted on almost no data is confidently wrong.*
4. **Minutes:** scale by expected minutes (recent starts, rotation, DOUBT flag).
5. **Penalties separately:** everything above is non-penalty; only the designated taker gets `λ_pen = expected_penalties × 0.76` added back.
6. **The Σλ rescale (load-bearing):** the player model's team total (say 2.4) disagrees with the match model's fitted `Λ_team` (say 2.1). The match model wins — it knows the opponent. Multiply every player's λ by `Λ_team / Σλ_i`, forcing `Σλ_i = Λ_team`. **This is the only channel by which opponent strength, home advantage, and the predicted scoreline reach the player numbers.** Without it, Haaland's number is identical against Brentford and Liverpool — the tell of a season-long rate table wearing a costume.
7. **To a probability:** `P(anytime) = 1 - exp(-λ)`. λ=0.95 → 61%.
8. **Shots & SOT:** same rates/shrinkage/minutes on total shots (position priors ~2.5 FW down to ~0.4 DF). Fixture adjustment is *not* the goal rescale but an **opponent shot-volume adjustment** (deep-sitting teams concede volume). `sot = shots × player_sot_ratio` (shrunk to a ~0.35 league prior). Props: `P(1+ shot) = 1 - exp(-s)`, `P(2+ shots) = 1 - exp(-s)(1+s)`, `P(1+ SOT) = 1 - exp(-sot)`. Team shot volume anchors to the match model's expected goals via the league's goals-per-shot conversion, so shots and the scoreline cannot drift apart.
9. **Publish:** top 3 per team per prop; penalty taker flagged; DOUBT flag on injured/doubtful.

**All constants (w, α, K, position priors) are provisional** — taken from the parent spec's research, not from our data. The backtest tunes them.

---

## 6. Props gate (`props_backtest.py`)

Walk-forward over 2025-26, training only on matches before each cutoff:

- **Anytime scorer:** log-loss and a calibration curve (players predicted at 20% should score ~20% of the time).
- **Shots / SOT:** MAE of expected vs actual counts.
- **Baseline:** position-prior-only (no player history, no fixture adjustment).

**Pass:** beats the baseline on every metric, and the scorer calibration curve is monotone and near-diagonal.
**Fail:** tune the constants. If it still fails, the props do not ship — per the parent spec's honesty rule, we say so rather than ship vibes.

---

## 7. Season simulation (`sim.py`)

10,000 runs over the remaining fixture list. Each run samples every fixture's scoreline from the fitted model, tallies points with real results locked in, and builds the final table. **PL tie-breakers in order: points → goal difference → goals for → head-to-head.** Aggregate finishing positions → projected table with `title_pct`, `top4_pct`, `relegation_pct`.

Pre-season this is the headline of the page (and the hardest test of the model's team strengths, since it compounds them over 38 matchweeks).

---

## 8. Pick tracking (`picks.py`)

Ports the WC honesty rules verbatim:
- Lock each pick into `data-raw/leagues/pl/picks_log.json` **before kickoff**.
- Grade the **frozen** pick against the authoritative result — never hindsight.
- A pick locked after kickoff is `tainted` → **void**, shown but excluded from the record.
- `record = {correct, wrong, total, void, pending, by_confidence}`.

Nothing grades until August. The rules go in now so they are not retrofitted later — retrofitting is how the WC app ended up voiding 14 picks.

---

## 9. JSON contract (`data/leagues/pl.json`)

Mirrors WC `predictions.json` wherever the card components are reused:

| Field | Notes |
|---|---|
| `updated`, `record` | Same shape as WC → the W-L chip is a straight port. |
| `matches[]` | `id, date, matchweek, home, away, prediction{p_home,p_draw,p_away,pick,pick_type,score,confidence,reasons[]}, result, graded, void` — same names as WC, so the fixture card needs no changes. |
| `matches[].props[]` | Per team, top 3 per prop: `{player, position, anytime_pct, exp_shots, p_shots_2plus, exp_sot, p_sot_1plus, penalty_taker, doubt}`. A **superset** of the WC scorer block. |
| `table[]` | `team, played, points, proj_points, title_pct, top4_pct, relegation_pct`. |
| `backtest` | Phase 1 gate numbers + props gate numbers — the page shows its own credibility. |

**Deliberate divergence:** club crests/colours (20 clubs) live in a static `data/leagues/clubs.json`, not in the predictions payload — they change once a season, not weekly.

---

## 10. Ops

- **Weekly job:** refresh current-season data, re-fit, re-sim, re-props, republish, deploy.
- **Match-day job:** record authoritative results, re-grade frozen picks, redeploy.
- Both run **locally** (like the WC scheduled tasks) — the league engine needs pandas/scipy/penaltyblog, and Render only serves the static site.
- Publish is **abort-on-failure** (reusing `deploy.py`'s lockfile + atomic write): a failed fetch must never ship a stale-but-fresh-looking file.

---

## 11. Error handling

| Failure | Response |
|---|---|
| Unmapped team/player name | **Fail loudly.** `names.py` raises rather than inventing `UnknownTeam`; `players.py` inherits this. |
| Thin-sample player | Shrinkage to position prior (§5.3). The backtest proves it worked. |
| FBref rate-limiting (likeliest op failure) | Cache hard, weekly cadence, Understat as xG backup. On failure: **abort the publish**, do not ship stale. |
| Missing xG for a fixture | Degrade to the goals-only channel. Safe *because* of the re-centring fix — previously a team with no xG became a superteam. |
| Cold-start (no 2026-27 matches yet) | ClubElo + prior-season strengths carry the model until real games accumulate. |

---

## 12. Testing

Unit tests with fixed payloads and no network (matching the 25 existing Phase 1 tests). Pinned by test:

- **Σλ invariant:** after rescale, `Σλ_i == Λ_team` for both sides.
- **Shrinkage direction:** a 1-nineties player lands nearer the position prior than a 30-nineties player.
- **Penalty split:** only the designated taker carries `λ_pen`; non-takers' rates are non-penalty throughout.
- **PL tie-breaker order:** points → GD → GF → head-to-head.
- **Pick voiding:** a pick locked after kickoff is void and excluded from the record.
- `players.py`: parse test against a captured payload + a manual live smoke test (as the Phase 1 loaders did).
- `props_backtest.py` is the integration-level check.

---

## 13. Risks

- **Props constants are guesses** until the backtest tunes them (§5, §6). This is the biggest open technical risk.
- **Four weeks to kickoff.** If the slice slips, the fallback is to ship the match model + season sim without props — the sim and the picks stand alone.
- **FBref availability** is outside our control; Understat is the backup for xG, but *player shot logs* have no equally good free backup. If FBref blocks us, shots/SOT are the props that die, not the scorer prop (Understat carries shots per player).
- **WC app collision:** mitigated by touching no WC file (§2).
