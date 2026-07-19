# Leagues Team News — Design Spec

**Date:** 2026-07-19
**Status:** scoped, not built
**Parent:** `docs/superpowers/specs/2026-07-13-multileague-predictor-design.md` (§14 left the injury feed as an explicit seam)

---

## 1. The gap

The leagues model consumes **no team news whatsoever**. There is no injury, suspension or lineup input anywhere in `leagues/`; the `doubt` field in `props.match_props` is hardcoded `False` and never set. So:

- A striker ruled out on Friday still appears in Saturday's scorer props.
- A rotated or benched key player is priced as if starting.
- A red-card suspension is invisible.

The World Cup app does not have this gap. Its `worldcup-prematch-update` task runs every 2h and, for fixtures within ~4h, web-searches "confirmed/predicted XI, late fitness, a player ruled out/benched, suspensions", updates `data-raw/news.json`, and removes newly-OUT players from `data-raw/players.json`. That mechanism is precisely what moved the France–Spain semi-final at lock time.

**This is the largest remaining accuracy gap in the leagues app.** Every other avenue explored (Weibull-copula, opponent-adjusted xG, O/U and BTTS markets, selective correct-score) was measured and rejected. This one has not been tried, and the WC app is live evidence that it moves predictions.

## 2. Why it cannot simply copy the World Cup approach

| | World Cup | Four leagues |
|---|---|---|
| Fixtures needing news | ~2–4 per day | **~37 per weekend** |
| Runs on | Claude scheduled task | GitHub Actions (no LLM) |
| Cadence | every 2h | weekly + match-day |

Two hard constraints follow:

1. **It cannot live in the GitHub Actions job.** Reading team news requires judgment — distinguishing "doubtful" from "ruled out", a rotation rumour from a confirmed XI. That is not deterministic code, and the whole point of the Actions workflow is that it runs without an LLM.
2. **Full coverage is unaffordable.** ~37 fixtures a weekend at WC-level rigour is roughly ten times the WC's workload, every week, for nine months.

## 3. Scope: Best Picks only

News is gathered **only for fixtures on the Best Picks board** (p ≥ 0.65, ~18 a matchweek across four leagues, often fewer).

Rationale:
- Those are the picks a reader actually acts on, and the ones carrying the 77.4% billing. A wrong Best Pick costs far more trust than a wrong 41% coin-flip.
- It bounds the work to a tractable ~18 fixtures.
- It concentrates effort where a single absent striker most changes the answer.

Explicitly **out of scope**: news for all other fixtures, and any attempt to model tactical/formation changes.

## 4. Design

**Data file:** `data-raw/leagues/news.json`, mirroring the WC's shape.

```json
{
  "_verified_on": "2026-08-21",
  "PL": {
    "Arsenal": {
      "out": ["Player A"],
      "doubt": ["Player B"],
      "note": "Player A suspended (red card MW1)",
      "checked": "2026-08-21T09:00:00Z"
    }
  }
}
```

**Consumption, in two places:**

1. `players.load_news(league)` → applied in `props.match_props`:
   - a player listed `out` is **removed** from that fixture's props entirely (his λ is redistributed across the remaining squad by the existing Σλ rescale, so the team total stays equal to the match model's Λ — no mass is lost);
   - a player listed `doubt` keeps his place but has expected minutes reduced and `doubt: true` set, which the card already has a slot for.

2. **No effect on the 1X2 pick in v1.** Team news adjusts *player props only*. Moving the match model on team news needs a defensible mapping from "a starter is out" to a change in team λ, and inventing one unvalidated would be exactly the kind of unmeasured change this project has rejected four times today. Left for v2, gated on evidence.

**Freshness:** `sanity_check` fails the deploy if a Best Picks fixture kicks off within 24h and its two clubs have no `checked` timestamp inside the preceding 48h — the same fail-loud discipline as the transfer overrides.

## 5. Who runs it

A Claude scheduled task, `leagues-matchday-news`, mirroring `worldcup-prematch-update`:
- **Cadence:** Friday morning and Saturday morning in season (most fixtures fall Sat–Sun).
- **Input:** reads `data/leagues/best.json` for the upcoming board.
- **Action:** for each club in those fixtures, web-search confirmed/predicted XI, injuries, suspensions; write `data-raw/leagues/news.json`; re-run `python -m leagues.publish`; `python -m scripts.sanity_check`; deploy.
- **Rule:** verified information only, never rumour — identical to the transfers standard.

This reintroduces a Claude dependency for *this feature only*. The core engine stays fully autonomous in GitHub Actions; if the news task does not run, publishing continues and the page simply reports that squads were not news-checked.

## 6. Risks

- **Cost/attention.** ~18 fixtures × 2 clubs, twice a weekend, for a full season. If it proves too heavy, tighten the board to p ≥ 0.70 (~9 fixtures) rather than degrade the quality of each check.
- **Unmeasurable benefit at first.** There is no backtest for this: historical team news is not in any feed we have. The honest evaluation is forward-looking — compare Best Picks hit rate before and after the feature over a meaningful sample, and be willing to conclude it did nothing.
- **Silent staleness**, mitigated by the freshness gate in §4.
- **Scope creep into the match model** — explicitly deferred in §4.

## 7. Success criterion

Stated in advance, to avoid grading ourselves generously later: the feature earns its place if, over ≥100 settled Best Picks, the board's hit rate with news is at least as good as the 77.4% backtested baseline **and** no Best Pick is ever published featuring a player confirmed out before kickoff. The second condition is the real point — the first is unlikely to be statistically separable within one season.
