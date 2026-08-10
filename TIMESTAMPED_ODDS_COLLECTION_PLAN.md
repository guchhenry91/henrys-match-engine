# Timestamped early-odds collection — design v2, nothing implemented

**Status: not built, not scheduled, not wired into any workflow.** Revised after
review — v1 was too vague on storage persistence, call volume, and reconciliation.
This version fixes those with concrete numbers and reuses this codebase's own existing
patterns rather than inventing new ones.

## The blocker, unchanged

API-Football's pre-match odds endpoint (`/odds?fixture=X`) returned **empty for every
fixture tested this session** — an upcoming PL match, an already-played match, a
Champions League final. Only in-play (live) odds returned data. Collection cannot
meaningfully start until this clears (re-check pending, next real matchday). Everything
below is the design to implement once that's confirmed, not something starting now.

## 1. Persistent storage — corrected

GitHub Actions workspaces are ephemeral: any file written during a run and not
committed back to the repo is gone when the run ends. This project already solves
exactly this problem for every other piece of state it keeps
(`data-raw/leagues/rosters.json`, `telegram_sent.json`, etc.) the same way: **commit the
file back to git at the end of the run**, same atomic-write + `leagues-bot` commit
discipline already used everywhere in `leagues/publish.py` and the sync scripts. No new
infrastructure (no database, no cloud bucket) — just the pattern this repo already runs
on, applied to a new file.

**Format: JSONL, one capture per line**, per instruction — not a single growing JSON
array. This matters concretely for a git-backed store: appending a line to a `.jsonl`
file is a one-line diff; appending to a JSON array means rewriting the closing
bracket/comma, which is a much worse diff and a real merge-conflict risk if two
matchday runs land close together (the existing cadence already runs every 30 minutes
during match windows — back-to-back commits are normal, not an edge case).

Path: `data-raw/leagues/odds_log/{league}.jsonl`, one file per league (keeps individual
file size and diff size manageable; a single combined file would grow to thousands of
lines by mid-season and make every commit's diff noisy for leagues that didn't even
have a capture that run).

## 2. Expected API call volume — computed, not guessed

- 4 leagues, matchweek fixture counts: PL 10, La Liga 10, Bundesliga 9, Ligue 1 9 → **38
  fixtures/week combined**.
- 4 lead-time buckets per fixture (below) → **at most 4 odds calls per fixture across
  its whole lifecycle**, not per day, not per 30-minute tick.
- `/odds?fixture=X` returns every market and every bookmaker in ONE call (confirmed in
  this session's diagnostic) — no per-market or per-bookmaker multiplication.
- Fixture-ID reconciliation (§4) batches by date via `client.get("fixtures", date=...)`
  — one call covers every fixture on that date, reused across all matches sharing it,
  same caching already used in `scripts/sync_lineups.py`'s `by_date` dict.

**Math**: 38 fixtures/week × 4 buckets = 152 odds calls/week + roughly 10–15
date-batched reconciliation calls/week (one per distinct fixture date, not per fixture)
≈ **165–170 calls/week ≈ 24 calls/day average**, with a realistic worst-case burst
(a full Saturday slate of ~8–10 fixtures all crossing the same bucket in the same
30-minute run) of ~10 calls in one run.

**Against the subscription**: the $20/mo plan is 7,500 requests/day, 300/min. 24/day
average is **0.3% of daily quota**; even a 3× pad for rate-limit retries and burst
clustering stays under 1%. This also sits alongside the existing roster/lineup sync's
~90 calls/day (established earlier this session) with no meaningful contention — total
combined usage still under 2% of quota. No plan upgrade needed for this specifically.

## 3. Lead-time capture buckets (replaces "every 30 minutes")

Piggybacks on the EXISTING cron cadence (`leagues.yml`'s `0,30 11-22 * * *` matchday
run and `0 6 * * 2` weekly refit) — no new schedule. At each existing run, for each
upcoming fixture, check whether time-to-kickoff has crossed into a bucket not yet
captured for that fixture:

| Bucket | Window | Rationale |
|---|---|---|
| `first_publish` | Whenever the fixture first enters the published window (`MATCHWEEKS_AHEAD` boundary, ~5–7 days out) | The earliest point Henry could plausibly use a live blend — the actual case this whole investigation is about |
| `24h` | ≤24h and >18h before kickoff | Standard "day before" market maturity checkpoint |
| `6h` | ≤6h and >3h before kickoff | Matchday-morning line, well before team news typically lands |
| `1h` | ≤60min and >30min before kickoff | Aligns with `LOCK_WINDOW_HOURS`, the same window this pipeline already uses to freeze picks — directly comparable to what a live deployment would actually see at publish time |

Bucket boundaries are approximate (±30 min, the tick granularity) — a fixture might be
captured at 23.6h instead of exactly 24h. Documented as a known imprecision, not
silently smoothed over.

## 4. Team/fixture reconciliation — reuses an existing, proven pattern

Not new logic: `scripts/sync_lineups.py` (lines ~38–56) already does exactly this —
fetch API-Football's `fixtures` for a date, resolve each candidate's team names through
`leagues.names.canonical()`, and match against this project's own `(home, away)` pair,
skipping any candidate that fails canonicalization (`UnknownTeam`) rather than guessing.
The odds collector reuses this verbatim rather than reinventing fixture matching:

```python
match = None
for candidate in by_date[date]:
    if candidate.get("league", {}).get("id") != API_LEAGUES[league]:
        continue
    try:
        home = canonical(candidate["teams"]["home"]["name"], league)
        away = canonical(candidate["teams"]["away"]["name"], league)
    except UnknownTeam:
        continue
    if (home, away) == (fixture["home"], fixture["away"]):
        match = candidate
        break
```

## 5. Deduplication

Key: **`fixture_id + provider + bookmaker + retrieval_bucket`** (exact fields per
instruction). Since JSONL is append-only and can't be cheaply queried for "has this key
already been written" without scanning a growing file every run, maintain a small,
separate, non-append state file per league —
`data-raw/leagues/odds_log/{league}_captured.json`, a flat map of
`{"{fixture_id}:{bucket}": true}` — checked before any capture attempt and updated
after a successful one. This file stays small (bounded by fixture count × 4 buckets,
not by bookmaker count), unlike the JSONL log itself.

## 6. Retries, rate limits, secrets

- **Retries**: reuse `scripts/sync_rosters.py`'s existing `get_json()` pattern
  (exponential backoff, capped attempts) rather than inventing a new retry style —
  that function already handles this codebase's real-world failure modes (transient
  TLS/network errors) and is proven in production.
- **Rate limits**: `leagues/api_football.py`'s `Client` class currently has NO retry or
  backoff logic at all (checked directly — a single failed call raises immediately).
  The odds collector needs this added: detect HTTP 429 / API-Football's rate-limit
  error body specifically, respect a `Retry-After` header if the response supplies one,
  otherwise back off longer than the roster sync's default retry delay before retrying
  once. Given the volume computed in §2, hitting the 300/min cap in normal operation is
  not expected — this is a defensive measure, not a response to an anticipated problem.
- **Secrets**: no new secret needed. Reuses the existing `API_FOOTBALL_KEY` GitHub
  Actions secret already wired for roster and lineup sync — same provider, same
  account, different endpoint.

## 7. Corrected collection timeline

**300–400 matches PER LEAGUE is not "a few months" — it's essentially a full season.**
PL alone plays 380 matches/season; at ~9–10 fixtures/week, reaching 300–400 fixtures
*with a genuine early-window capture* for one league takes roughly 30–38 weeks — most
of a season, not a short collection window. This was stated imprecisely before and
needed correcting.

Two honest options, not one glossed-over estimate:
- **Per-league validation** (300–400 fixtures in ONE league): ~7–9 months. Only
  realistic if this stays a multi-season, standing research track.
- **Pooled validation** (300–400 fixtures combined across all 4 leagues, same
  approach §5's original benchmark already used for the closing-odds backtest): ~38
  fixtures/week combined → **8–10 weeks** for a first pooled re-evaluation, with
  per-league numbers filled in progressively as each league's own count grows in
  parallel. This is the realistic near-term milestone; per-league significance would
  still lag behind it.

Neither starts until §0's blocker clears. This section states the honest cost of doing
this right — with genuine timestamps, no closing-price shortcuts — not a schedule
commitment.

## Status

Design only. Nothing scheduled, nothing implemented. Waiting for (a) confirmation the
odds-availability blocker has cleared, then (b) approval to implement the capture step
as a small, additive, read-only logging addition to `leagues.yml` — no effect on any
published prediction until a separate, later decision to wire the blend itself in.
