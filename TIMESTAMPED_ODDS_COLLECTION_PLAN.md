# Timestamped early-odds collection — design only, nothing implemented

**Status: not built, not scheduled, not wired into any workflow.** This is the plan for
review. No code in this document runs anywhere until approved.

## Why this exists

Every market-blend number in `CORRECT_SCORE_AUDIT_REPORT.md` §5 was backtested against
football-data.co.uk's historical **closing** odds — the best price right before kickoff.
This pipeline's actual publish cadence (`MATCHWEEKS_AHEAD=1` in `leagues/publish.py`,
plus the 30-minute matchday refresh cycle in `.github/workflows/leagues.yml`) generates
picks up to a week before kickoff, when odds are thinner, less mature, and can move
significantly before close. Closing-line backtests cannot honestly stand in for that. The
only way to know whether the market blend helps at the moment Henry actually publishes is
to record odds at that moment, for real, going forward, and backtest against *that*
archive once enough fixtures have accumulated.

## Current blocker (must be resolved before collection can start)

API-Football is the only currently-connected odds source. Its pre-match odds endpoint
(`/odds?fixture=X`) returned **empty for every fixture tested this session** — an
upcoming Premier League match, an already-played match from last season, and a
Champions League final — despite "Pre-match Odds" being listed in the account's
package features. Only in-play (live match) odds actually returned data, which is
useless for a pre-kickoff pipeline. Per the standing plan, this gets re-checked once a
real matchday arrives (a match a few hours from kickoff, when pre-match markets should
be most active if they're going to appear at all). **Collection cannot meaningfully
start until this is confirmed working end-to-end** — collecting nothing but empty
responses would be a wasted quota spend, not evidence of anything.

## Design (once the blocker clears)

### What gets recorded, every capture

| Field | Source | Notes |
|---|---|---|
| `provider` | Fixed string, e.g. `"api-football"` | Whichever source is actually live |
| `retrieval_timestamp` | `datetime.now(timezone.utc).isoformat()` at the moment of the API call | Not the publish run's start time — the literal moment this specific odds call returned |
| `kickoff_timestamp` | From the fixture data already in the pipeline | So "hours before kickoff" is always derivable, never estimated |
| `bookmaker_update_timestamp` | Only if the provider supplies one (API-Football's odds objects may include an `update` field per bookmaker — needs confirming once the endpoint actually returns data) | Recorded as `null` if not supplied, never backfilled or guessed |
| `bookmaker` | Per-bookmaker, not just a consensus average | So a later analysis can still ask "was this specific book's price fresh" |
| `odds_1x2` | `{home, draw, away}` raw prices | Raw, not de-vigged — de-vig at analysis time, not capture time, so the raw archive stays reusable if the de-vig method changes |
| `odds_over_under_2_5` | `{over, under}` raw prices | Same raw-not-devigged rule |
| `missing_data_status` | One of `"complete"`, `"partial_1x2_only"`, `"partial_ou_only"`, `"empty"` | Every capture attempt gets a row, even a failed one — silence is itself data (tells us how often the market simply isn't quotable yet at that lead time) |
| `fixture_id`, `league`, `home`, `away` | From the existing fixture pipeline | Joins cleanly to everything else already in `data-raw/leagues/` |

### Storage

A new, append-only file: `data-raw/leagues/timestamped_odds_log.json` (or one per
league, `timestamped_odds_log_{PL,LALIGA,BUNDESLIGA,LIGUE1}.json` — decide once volume
is known). Append-only: never overwritten, never deduplicated by rewriting — each
capture is a permanent, immutable row, since the whole point is having an honest
record of what was actually seen at each point in time. Written atomically (temp file +
`os.replace`, same discipline as every other write in `leagues/publish.py`) so a crash
mid-write can't corrupt the archive.

### Capture cadence

Piggyback on the pipeline's EXISTING publish cadence rather than inventing a new
schedule — capture odds at the same moments `leagues/publish.py` would actually use
them, so the archive directly answers "what would the blend have seen, for real, at
each real publish run":
- Every scheduled `leagues.yml` run that has an upcoming fixture in its window (the
  existing `0 6 * * 2` weekly refit and the `0,30 11-22 * * *` matchday cadence) makes
  one capture attempt per upcoming fixture, alongside the existing fetch steps — not a
  new separate cron.
- This means a single fixture accumulates MULTIPLE timestamped snapshots over the days
  leading to kickoff (e.g. first seen 6 days out at the Tuesday refit, then every 30
  minutes once inside the matchday window) — which is actually more valuable than one
  snapshot, since it lets a later analysis pick exactly "how many hours before kickoff"
  to test at, rather than being stuck with whatever a single arbitrary capture time gave.

### What this does NOT do

- Does not feed anything into `leagues/publish.py`'s actual predictions. This is a
  read-only, side-channel logging step. Zero effect on any published pick, any
  `data/leagues/*.json` field, or any user-visible behavior.
- Does not touch `leagues/history.py`'s existing closing-odds columns (those stay as
  the walk-forward backtest's data source; this is a new, separate, forward-looking
  archive that only starts accumulating from whenever it's turned on).
- Does not retroactively backfill. Per instruction: never approximate early odds using
  closing prices. If a fixture wasn't captured in real time, it simply isn't in this
  archive — there is no substitute.

### When there's enough data to re-run the evaluation

No fixed date — driven by volume. A rough target: at least 300-400 fixtures per league
with a genuine early-window capture (not just the closing-window one) before re-running
the walk-forward-style comparison from §5, since that's roughly the smallest sample
this session's paired-bootstrap CIs stayed informative at. Given ~9-10 fixtures/league/
week, that's several months of real collection, not something available quickly. This
is the honest tradeoff for insisting on real timestamps instead of closing-line
approximation — flagged clearly so the timeline is understood going in, not discovered
partway through.

## Status

Design only. Nothing scheduled, nothing implemented, no code changes anywhere. Waiting
for (a) confirmation the odds blocker has cleared, then (b) approval to implement the
capture step itself as a small, additive, read-only logging addition to `leagues.yml`.
