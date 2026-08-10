# Correct-Score Engine Audit — Final Report

**Status: research complete, nothing applied to production.** Every script referenced
here lives under `scripts/`, writes only to `data-raw/leagues/*.json`, and has been run
against real historical data with the same walk-forward, causal, paired-bootstrap
discipline used elsewhere in this codebase. `leagues/`, `index.html`, and `data/leagues/`
are unchanged except for two additive, backward-compatible column captures in
`leagues/history.py` (Over/Under 2.5 odds, Bet365-specific odds) that nothing in
production reads yet.

---

## 1. Root-cause audit (Phase 1)

**Accepted finding, not re-litigated:** the frequent 1-1 grid-mode is a genuine,
thin-margin property of a well-differentiated Poisson-based model applied to low-scoring
football — not compression, not a rho bug, not stale data. Evidence:

- **Attack/defence spread is real.** Log-scale std 0.18–0.23 across leagues; zero
  promoted clubs fell back to the crude "weakest team" prior — every one resolved via
  real second-tier-informed priors (`correct_score_audit_phase1.json`).
- **Rho is already fit independently per league**, and genuinely varies: PL −0.105,
  La Liga **+0.019** (positive — actually *suppresses* 1-1), Bundesliga −0.057,
  Ligue 1 −0.058. This directly answers "should rho be per-league" — it already is, and
  it's doing real, differentiated work (La Liga has the lowest near-term 1-1 rate and
  best correct-score accuracy of the four).
- **The 1X2 model never picks a draw** across the full remaining season (1,372
  fixtures, all 4 leagues) — not a bug. Home advantage structurally pushes the leading
  side's probability above the draw's; even the single closest fixture found
  (Everton v Crystal Palace, H 37.6% / D 29.0% / A 33.4%) has the draw trailing by 8.5
  points. Confirmed independently in the draw investigation (§6) — the real betting
  market shows the identical pattern.
- **The margin by which 1-1 wins is thin**: mean top1-vs-top2 gap 1.4–3.4 percentage
  points across leagues, with some literal ties (min gap 0.0pp in Bundesliga/Ligue 1).
  It's winning a close race among several similarly-likely low scores, not dominating.

Fixture-level diagnostic table (all 38 near-term fixtures + all 1,372 remaining-season
fixtures, with home/away lambda, rho, top-6 scores, gap, 1X2 probs, pick, grid mode,
pick-conditional score): `data-raw/leagues/correct_score_audit_phase1.json`. Reproducible
via `python -m scripts.correct_score_audit`.

---

## 2. Leakage-free benchmark (Phase 2/3)

`scripts/correct_score_benchmark.py` — walk-forward, strictly causal (train on
`date < cutoff` only), weekly steps, all 4 leagues + combined. Primary metric: full-grid
log-loss. Full results: `data-raw/leagues/correct_score_benchmark_phase2.json`.

| League | Existing model log-loss | Actual 1-1 rate | Predicted 1-1 pick rate |
|---|---|---|---|
| PL | 2.9686 | 11.44% | 63.56% |
| La Liga | 2.7986 | 14.42% | 44.77% |
| Bundesliga | 3.0734 | 11.18% | 81.53% |
| Ligue 1 | 2.9788 | 10.26% | 75.22% |
| **Combined** | **2.9425** | 11.98% | 64.35% |

**Rho-removed / simple independent Poisson** (mathematically identical under this
codebase's grid construction — Phase 2's candidate #2 and Phase 3's "simple independent
Poisson" baseline are the same thing here, implemented once): **not statistically
distinguishable from the existing model in any league.** Removing rho *does* shift which
score wins the argmax a lot (Bundesliga's predicted-1-1 rate swings from 82% to 64%)
without moving aggregate log-loss — rho reshuffles a few cells, it doesn't change overall
quality.

**Per-scoreline calibration is good** — predicted vs. actual for 0-0/1-0/0-1/1-1/2-1/1-2
mostly within 1–3 percentage points.

**The one real, consistent, actionable finding:** total goals are mildly
under-predicted in every league (combined: 2.81 predicted vs. 2.89 actual; worst in
Bundesliga, 3.069 vs. 3.174 mean goals; 4+-goal matches under-predicted by up to 5.2
points there).

---

## 3. Total-goals under-prediction: mechanism and correction (explicitly requested)

### Mechanism audit (`scripts/total_goals_mechanism_audit.py`)

Isolated each candidate cause. Result: **no single mechanism is the culprit.**

- **xG blend**: Understat xG runs *above* actual goals in every league (opposite of
  the "conservative xG" hypothesis) — PL +0.16, La Liga +0.23, Bundesliga +0.05,
  Ligue 1 +0.16. Sweeping `xg_weight` from 0 (pure goals) to 1 (pure xG) moves the
  predicted level by only 0.02–0.09 goals, because the blend is deliberately
  re-centered on the goal model's own level by design — confirmed working as intended.
- **Shrinkage**: contributes only 0.02–0.03 goals — not the driver.
- **Home advantage**: recent-1-year `home_adv` is *higher* than the full-history fit in
  every league (e.g. PL 0.182 → 0.214), not lower — rules out a stale/underestimated
  home-advantage hypothesis.
- **Season-by-season scoring trend**: noisy, no consistent monotonic uptrend across all
  four leagues.
- **The real signal**: the gap is far larger in true walk-forward (causal, held-out)
  prediction than in an in-sample fit on the same data. This points to **real-time
  forecasting conservatism** — the decay/shrinkage bias-variance tradeoff paying for
  lower variance with a small, consistent bias in live prediction — not a fixed
  structural bug anywhere in the pipeline.

### Correction test (`scripts/total_goals_correction_experiment.py`)

Tested a causal, recent-window (120-day, pre-cutoff only) recalibration factor, both as
a uniform total-scale and separate home/away scales.

**Confirmed working mechanically** — mean predicted total moves substantially closer to
actual everywhere (e.g. La Liga 2.511 → 2.640 vs. actual 2.654). **But grid log-loss
does not improve significantly in any league**, and is slightly worse in
Bundesliga/Ligue 1. **Answering the explicit question: no, a naive total-goals rescale
does not solve the under-prediction in a way that improves calibration** — because it
scales lambdas without re-deriving rho/grid-shape for the new scale, distorting how the
low-score correction interacts with the rescaled lambdas.

This is *why* the market-blend candidate (§5) works where naive rescaling doesn't: it
re-derives total **and** direction jointly, self-consistently, through the model's own
rho — not a bolt-on afterward.

---

## 4. Alternative distribution families (Phase 4b/4c) — all negative

| Candidate | Combined log-loss delta vs. existing (95% CI) | Significant? |
|---|---|---|
| Bivariate Poisson (shared-shock, causal MLE) | [−0.0018, 0.0036] | No |
| Negative binomial marginals (causal MLE) | [−0.0017, 0.0022] | No |
| Dynamic strength (xi/half-life re-tuned) | PL: [−0.0156, −0.0028] **(default wins, significantly)** | No — and PL regresses if changed |
| CMP | Not implemented — normalizing constant has no closed form, judged impractical inside a per-match walk-forward MLE loop given the time budget; documented, not forced |

Bivariate Poisson's fitted shared-shock component (λ₃) is genuinely near-zero for
PL/Bundesliga (mean 0.002–0.004) but real for La Liga (mean 0.124) — a real, league-specific
correlation signal exists, but even where it's substantial it doesn't produce a
significant log-loss gain. **The goal distribution family / correlation structure is not
the bottleneck.** Full results: `data-raw/leagues/bivariate_poisson_experiment.json`,
`negative_binomial_experiment.json`, `dynamic_strength_experiment.json`.

---

## 5. Market-calibrated model (Phase 4a) — the one positive finding

`scripts/market_combined_experiment.py`, `scripts/market_model_ab_report.py`.

**Method**: de-vig Over/Under 2.5 → market-implied total goals. De-vig 1X2 → market's
own (p_home, p_draw, p_away). Solve for a split ratio that reproduces the market's 1X2
via *this model's own* rho/grid shape (not the model's attack/defence ratio). Blend
weight fit on the earlier half of held-out matches, evaluated on a separate later half.

| League | A: Independent Henry | B: Market-calibrated | 95% CI (Henry − Market) | Significant? |
|---|---|---|---|---|
| PL | 2.8955 | 2.8871 | [−0.0078, 0.0236] | **No** |
| La Liga | 2.7805 | 2.7604 | [0.0063, 0.0347] | **Yes** |
| Bundesliga | 3.0728 | 3.0459 | [0.0069, 0.0462] | **Yes** |
| Ligue 1 | 3.0324 | 3.0041 | [0.0095, 0.0479] | **Yes** |

**Important honesty check, stated explicitly per the request: this is not "Henry beats
the market."** The optimal weight on Henry's own team-strength estimates is 0.00 in
every league. It means: using the market's own total+direction, filtered through this
model's rho/grid shape, produces better-calibrated *correct-score probabilities* than
Henry's own strength estimates alone. A real, externally-validated accuracy gain for
what gets published — not alpha over the market.

**Bookmaker stability**: Bet365-only direction vs. multi-book consensus direction are
**statistically indistinguishable in every league** (all CIs include zero) — the signal
isn't an artifact of one bookmaker's pricing.

**Constrained blend — the "don't quietly replace Henry" answer**: a fixed-weight
tradeoff curve (0/10/20/30/40/50/100% retained on Henry) shows:
- **PL**: nearly flat near the low end; 20–30% Henry is *marginally better* than 0%,
  not worse (2.8854 at 20% vs. 2.8871 at 0%).
- **La Liga/Bundesliga/Ligue 1**: monotonically best at 0% Henry, but 10–20% Henry costs
  only a small fraction of the full 0-to-100% range (e.g. La Liga: full range 0.0201
  nats; 20% Henry costs 0.0008 of that).

**Recommendation: a ~20% Henry / 80% market blend**, not the literal zero-weight
optimum — near-optimal everywhere, keeps genuine model independence, and is free-to-marginally-better
in PL specifically.

**Correct-score market comparison: not possible.** API-Football's own pre-match odds
(including its "Exact Score" market) returned empty for every fixture tested this
session — past, upcoming, and a Champions League final — despite being listed in the
account's package features. Documented, not resolved.

**⚠ Data-provenance / leakage caveat (critical for any production decision):** the odds
used are football-data.co.uk's historical archive — **closing** (or near-closing) lines.
This is NOT the same as odds available at this pipeline's actual publish timestamp
(`MATCHWEEKS_AHEAD=1` means a pick can be generated days before kickoff). This backtest
validly proves "market information helps," but does **not** simulate the live publish
cadence. A production deployment would need odds captured *at publish time* — thinner,
less mature than closing lines — and would likely show a **smaller** benefit than
reported here. This is the single biggest open risk before promoting this candidate.

---

## 6. Draw investigation (explicitly requested)

`scripts/draw_investigation.py`. Confirmed by direct code read
(`leagues/publish.py:382`, `pick = max(probs, key=probs.get)`): the pick-selection code
is a correct, unbiased argmax over the three raw model probabilities. `Calibrator` in
`leagues/model.py` is defined but **never called anywhere** in `leagues/` or `scripts/`
(repo-wide grep) — production ships raw, uncalibrated probabilities.

| League | Actual draw rate | Model mean p(draw) | Market mean p(draw) | Model draw-picks | **Market draw-picks** |
|---|---|---|---|---|---|
| PL | 24.56% | 23.31% | 23.57% | 0 | **0** |
| La Liga | 25.95% | 25.83% | 26.28% | 3 | **9** |
| Bundesliga | 25.75% | 24.77% | 23.39% | 0 | **0** |
| Ligue 1 | 23.69% | 24.93% | 24.84% | 0 | **0** |

Model's mean draw probability tracks the actual draw rate within ~1–1.3pp everywhere,
and tracks the market's own mean draw probability just as closely. Bucket-level
calibration is noisy but not systematically biased (gaps within 1–2 standard errors at
n≈150–230/bucket — unlike the total-goals finding, which was clean and consistent).
**Most decisive: the real market also picks draw as its own 1X2 favorite almost never**
— 0/0/0 in three leagues, 9 of 1,137 in La Liga, closely matching the model's own
0/3/0/0. **The model is not systematically underpricing draws.**

---

## 7. Answering the required questions directly

1. **Best independent model (no odds)**: the existing production Dixon-Coles + xG blend.
   Every architectural challenger tested (no-rho, bivariate Poisson, negative binomial,
   re-tuned xi, naive total-goals rescale) failed to beat it with significance anywhere,
   and PL significantly *regressed* under re-tuned xi. No changes recommended to the
   independent model itself.
2. **Best market-informed model**: the combined 1X2+O/U split-solve blend
   (`market_combined_experiment.py`), constrained to ~20% weight on Henry's own lambdas
   rather than the literal zero-weight optimum.
3. **Best model per league**:
   - PL: existing model (market blend directionally favorable, not significant — and
     PL is the league this matters most for, so the higher bar is appropriate).
   - La Liga, Bundesliga, Ligue 1: constrained (~20% Henry) market-calibrated blend,
     significant improvement.
4. **Does the total-goal correction solve the under-prediction?** Mechanically yes (moves
   the predicted mean much closer to actual), but a naive rescale **does not** improve
   calibration — only the market-informed joint total+direction correction does, and
   only in 3 of 4 leagues with significance.
5. **Does any challenger significantly beat production?** Yes — the market-calibrated
   blend, in La Liga, Bundesliga, and Ligue 1. No architectural challenger (distribution
   family, dynamic strength) does, anywhere.
6. **Computation/data requirements**: the market-blend candidate needs `odds_h/d/a` and
   `odds_over25/under25` at **publish time**, not the closing lines this backtest used
   (see §5's leakage caveat) — this is a real, unresolved gap, not a computation cost
   issue. Compute cost itself is trivial (one `scipy.optimize.minimize_scalar` call per
   fixture at publish time, sub-millisecond).

---

## 8. Proposed patch (NOT applied — for review only)

### If the market-calibrated blend is approved for La Liga/Bundesliga/Ligue 1:

**Files that would change:**
- `leagues/history.py` — already additive/merged (Over/Under + Bet365 columns
  captured; nothing reads them in production yet). No further change needed here.
- `leagues/model.py` — add a new function (not a `LeagueModel` method, since it needs
  market odds as an input) implementing the split-solve blend from
  `scripts/market_combined_experiment.py`'s `solve_split()`, callable as
  `market_calibrated_lambdas(model, home, away, odds_h, odds_d, odds_a, odds_over25, odds_under25, henry_weight=0.2)`.
- `leagues/publish.py` — in `build()`, after computing `pred = model.predict(home, away)`,
  conditionally call the new blend function IF live odds are available at publish time
  for that league (gate on `league in {"LALIGA", "BUNDESLIGA", "LIGUE1"}` per the
  per-league recommendation above), replacing `pred["lambda_home"]/["lambda_away"]`
  and rebuilding the grid before the rest of `build()`'s pipeline runs. Must add a new
  `data_warnings` entry when odds are unavailable for a fixture (fall back to the
  independent model silently degrading, never erroring).
- **NEW dependency**: a live odds source at publish time. This is the actual blocker —
  football-data.co.uk does not provide a real-time feed, only historical archives.
  Would need either (a) API-Football's odds endpoint once pre-match odds coverage is
  confirmed working (currently returns empty — see §5), or (b) a different odds
  provider. **Do not implement the publish.py wiring until a live odds source is
  confirmed working end-to-end.**

### Presentation change (prepared, not deployed — Phase 5's exact spec)

Current `index.html` (`openMatch()`, lines 497–534) already implements ~80% of the
requested structure post this session's earlier `09e48fc` fix: the true grid-mode score
is the headline ("best guess"), the pick-conditional score shows as a note **only when
it differs** from the grid mode, and a top-3 "Most likely scorelines" section exists.

Gap vs. the exact 4-part spec requested:
1. No explicit "Match Pick: Team (P%)" label currently rendered in the detail view
   (only the H/D/A split bar, without a text pick line).
2. "Projected Score If Pick Wins" is currently conditional (`condDiffers`) — shown only
   when it disagrees with the raw mode. The spec wants it always shown, labeled plainly.
3. Headline label "best guess" → rename to "Most Likely Raw Score" per spec, with the
   existing honest caveat text kept.

Proposed diff (reference only, not applied to `index.html`):

```diff
@@ index.html openMatch(), inside the .dhead block @@
     <div class="glass dhead"><div class="lg">${esc(LG[lk])} · MW${m.matchweek}</div><div class="vs">${esc(m.home)} v ${esc(m.away)}</div>
+      <div style="font-size:12px;font-weight:700;color:var(--sub)">Match Pick</div>
+      <div style="font-size:15px;font-weight:800">${esc(p.pick)} <span style="color:var(--sub);font-weight:600">(${pc(p.p_pick)}%)</span></div>
-      <div class="score"><b>${esc(sc[0])}</b><span>best guess${bestPct!=null?" · "+bestPct+"%":""}</span><b>${esc(sc[1]||"")}</b></div>
+      <div class="score"><b>${esc(sc[0])}</b><span>Most Likely Raw Score${bestPct!=null?" · "+bestPct+"%":""}</span><b>${esc(sc[1]||"")}</b></div>
       <div class="split"><div class="sh" style="flex:${ph||1}">${ph}%</div><div class="sd" style="flex:${pd||1}">${pd}%</div><div class="sa" style="flex:${pa||1}">${pa}%</div></div>
       <div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:8px">${esc(m.home)} win · draw · ${esc(m.away)} win</div>
-      ${condDiffers?`<div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:4px">If ${esc(p.pick)} win${p.pick===m.home||p.pick===m.away?"s":""}, most likely: ${esc(p.score)}</div>`:""}</div>
+      <div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:4px">Projected Score If Pick Wins: ${esc(p.score)}${condDiffers?" (differs from the raw mode above)":""}</div></div>
```

(`tops` section, already labeled "Most likely scorelines", already matches "Other
Plausible Scores" in substance — cosmetic rename only if desired, not required.)

---

## 9. Risks and rollback plan

**Risks of promoting the market-calibrated blend:**
1. **Live-odds timing gap (critical, §5)** — backtest used closing lines; live publish
   cadence needs earlier odds, which may show a smaller or no benefit. Must be
   re-validated with genuinely-timed odds before shipping, not assumed to transfer.
2. **Correct-score market itself unreachable** (§5) — cannot cross-check the blend's
   output against real correct-score market prices, only against 1X2/O-U markets used to
   *build* it. Reduces confidence in the specific exact-score numbers even where the
   1X2/O-U-derived blend is statistically validated.
3. **PL excluded from the recommendation** — the one league with real money on the line
   for this user did not reach significance. Shipping the blend only for the other three
   leagues avoids overclaiming there, but creates a per-league behavioral split to
   maintain and explain.
4. **New runtime dependency** — publish.py would need network access to a live odds
   source at every publish run, with graceful degradation already required (never error,
   never silently ship stale odds as fresh).

**Rollback plan**: the proposed change is additive and gated (§8) — a
`henry_weight` parameter and a per-league feature flag. Reverting is a one-line change
(set `henry_weight=1.0` or remove the league from the gated set) with no data migration,
no schema change to `data/leagues/*.json` beyond the already-existing fields, and no
effect on `leagues/model.py`'s core `LeagueModel` class (the blend lives alongside it,
not inside it). The independent model remains fully intact and is the automatic fallback
whenever live odds are unavailable.

**Promotion criteria checklist** (per the explicit request):
- [x] Improves untouched chronological exact-score log-loss — yes, La Liga/Bundesliga/Ligue 1
- [x] Survives paired bootstrap — yes, those three leagues
- [ ] Does not materially regress any league — **PL is neutral-to-favorable but not
      significant; recommend excluding PL from the gated rollout rather than treating
      this as a regression**
- [ ] Does not damage 1X2 probabilities — **not yet directly re-verified against the RPS
      gate** (`leagues/tune.py`); the blend's 1X2 output is by construction close to the
      market's own de-vigged 1X2, which prior work (`market_gap_experiment.json`) already
      showed is at least as good as the current model's RPS — reasonable inference, not
      a direct re-test
- [x] All results reported before any production modification — this document

**Not promoted, no further action needed**: bivariate Poisson, negative binomial, CMP
(not attempted), re-tuned xi/dynamic strength, naive total-goals rescale, rho removal.
All tested rigorously, all negative, all safe to leave as closed investigations.

---

**Waiting for approval before touching any production file.**
