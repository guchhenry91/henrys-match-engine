# Build prompt — Henry's Match Engine

A complete specification for rebuilding this system. Written so the hard-won
decisions survive: most sections say **why**, because nearly every constraint
here exists because the obvious alternative was tried and failed in a specific,
recorded way.

---

## 1. The product

A static football prediction site for four leagues — **Premier League, La Liga,
Bundesliga, Ligue 1** — covering the 2026-27 season.

Seven views, all in one self-contained HTML file:

| View | Content |
|---|---|
| Today | Cross-league overview, top pick, live counters |
| Best Picks | Match-winner picks at p ≥ 0.65 |
| Player Picks | Anytime goalscorer, 2+ shot attempts, 1+ shot on target |
| Parlay | Model-built accumulators in three sections |
| Tables | Real standings from played results |
| Grades | Separate records per board and per market, plus a calibration chart |
| Per-league | Fixtures and predictions for each league |

Every number published must be **honest about its own uncertainty**. This is the
governing principle. A pick that is right 12% of the time must say 12%, even
when that looks weak.

---

## 2. Stack

- **Python engine** (`leagues/`), no web framework. One command publishes:
  `python -m leagues.publish`
- **Single-file UI** (`index.html`) — inline `<style>` and `<script>`, no build
  step, no bundler, no external stylesheet. Reads `data/leagues/*.json` with
  `fetch(..., {cache:"no-cache"})`.
- **Static hosting** (Render), auto-deploying on push to `main`.
- **GitHub Actions** runs the engine on a schedule — deliberately, so nothing
  depends on a laptop being awake.

```
penaltyblog==1.11.0   soccerdata==1.9.0   pandas>=3.0
scipy>=1.18           numpy>=2.0          scikit-learn>=1.9   pytest>=8.0
```

195 tests. Everything below that says "measured" has a script under `scripts/`
that produced the number and a JSON artifact under `data-raw/leagues/`.

---

## 3. Data sources, and why each one

| Need | Source | Why not the obvious alternative |
|---|---|---|
| Results + closing odds | football-data.co.uk CSVs | — |
| Team xG | Understat | — |
| Player rates | Understat season stats + shot events | **Not FBref.** soccerdata's FBref player-match reader drives a headless Chrome per match page (~4/min) — five seasons of one league is ~8 hours, four leagues is days. Understat gives the same signal in seconds. |
| Squad membership | API-Football (paid tier), ESPN fallback | The free API-Football tier has no current-season access; discovering that took a broken sync sitting stale for 9 days. |
| Confirmed lineups | API-Football | — |
| Promoted-club priors | Second-tier football-data.co.uk (E1/SP2/D2/F2) | **Not ClubElo** — a single third-party point of failure that was down for days. |
| Live 1X2 market odds | football-data.co.uk `fixtures.csv` | Same feed the backtest scores against, so the de-vig and team-name mapping are reused unchanged. Carries ~1 week ahead, so it is legitimately empty off-season. |

**Penalties are unlabelled.** soccerdata maps Understat's "Penalty" situation to
`NA`. Match on `NA`, not on the string, or every club silently gets no penalty
taker.

**Bundesliga shot events crash upstream** (`read_shot_events` returns a list
where a dict is expected). Both callers must guard and degrade — SOT via a
league-average ratio, neutral opponent factors — rather than sink the league.

---

## 4. The match model (`leagues/model.py`)

Dixon-Coles fitted via `penaltyblog`, blended with xG:

1. Fit Dixon-Coles on **actual goals** → `rho`, home advantage, attack/defence.
2. Compute xG-based attack/defence separately (log ratio to league average).
3. **Blend the deviations, not the raw values** — 75% xG / 25% goals.
4. Build the scoreline grid with the Dixon-Coles tau correction.

> **Why blend deviations:** penaltyblog pins `mean(attack)=1` and lets defence
> absorb the league scoring level (mean ≈ −0.8), while xG log-ratios centre on 0.
> Mixing the raw scales shifts every lambda down 12–18%, under-predicting goals
> and inflating draws.

**Constants** (all tuned by walk-forward sweep, not guessed):

```python
XG_WEIGHT      = 0.75    MAX_GOALS = 10
PRIOR_STRENGTH = 3.0     # empirical-Bayes pull toward the league mean
XI_PER_DAY     = 0.003   # ~231-day half-life
```

> `xi=0.0065` from the 1997 paper is per **half-week**, not per day. In days the
> sweet spot is 0.0018–0.0033.

> `PRIOR_STRENGTH` shrinkage lands on **every** team, not just thin-data ones —
> with a 231-day half-life even five full seasons gives `eff ≈ 38`. It therefore
> controls the whole league's strength spread and must be tuned, not guessed.

**Two scores per fixture, and they are different questions:**

- `top_scores[0]` — the unconditional grid mode. The single most probable exact
  scoreline.
- `score` — the most likely scoreline **given** the 1X2 pick
  (`score_for_outcome`).

They disagree often, because 1-1 is the largest individual cell in most close
matches even when one side is clearly favoured. **Verified against 7,008 real
matches: 1-1 is the most common scoreline in all four leagues** (11.0–13.5%).
Both must be published; which one leads is a product decision (see §9).

---

## 5. Player props (`leagues/props.py`)

Pipeline: rates → shrinkage → expected minutes → penalties → **rescale to the
match lambda** → Poisson → **calibration**.

```python
HOME_SHOT_FACTOR = 1.108   # measured: 7,007 matches, home sides shoot ~23% more
PEN_CONVERSION   = 0.76
MIN_SQUAD_FOR_PROPS = 6
```

**The rescale is load-bearing.** Every player's goal lambda is scaled so a
team's players sum to exactly the match model's team lambda. This is also the
*only* venue-awareness channel for goals — which is why shots/SOT needed their
own (`HOME_SHOT_FACTOR`), since they are built from season shot rates and are
not rescaled to anything.

**`MIN_SQUAD_FOR_PROPS = 6`:** a team with fewer players in the rates table gets
**no** props. The rescale hands a near-empty squad's entire goal expectation to
one man — promoted Schalke had one player with top-flight history and published
as a **72.8% anytime scorer** when nothing else in four leagues beat 50.8%.
*One player is more dangerous than none: none is a visible hole, one looks like
the best pick on the board.*

### Calibration — measured overconfidence correction

```python
PROP_CALIBRATION = {"goal": 1.437, "shots": 1.954, "sot": 1.376}   # p -> p**k
```

Published probabilities were measured **short of their stated rate in every
market**:

```
PL      goal 42.1 -> 15.6    shots 75.0 -> 50.0    sot 66.8 -> 55.6
LALIGA  goal 42.8 -> 29.5    shots 77.0 -> 60.2    sot 69.8 -> 61.1
```

This is genuine, not a grading artefact: every probability **already multiplies
by the appearance probability**, so a player who does not feature is priced in
and grading his absence as a loss is consistent with the claim.

Design rules for the correction:
- **Monotone** — it may make the board honest but must never reorder it.
- **One parameter** — ~100–150 picks per market cannot support anything richer.
- **Fitted on one league, tested on the other**, keeping the weaker of the two,
  so it is strictly better than no correction in both and cannot over-deflate.

**The bars must be re-based onto the same scale** (`new = old ** k`):

```python
PLAYER_PICK_MIN_PROB = {"goal": 0.276, "shots": 0.497, "sot": 0.511}
```

> Without this, the correction silently guts the board — it cut 144 published
> picks to 18 and emptied the goalscorer section. Selection is unchanged; only
> the labelling became truthful.

### The gate that measures this (`scripts/props_pick_calibration.py`)

The rate gate deliberately stops short of per-match calibration, because shot
events contain only players who *shot*. That is correct for a curve over all
players. It does not block the narrower question: **of the picks actually
published, what fraction won?**

**Cluster by player.** The same man is picked every week, so picks are not
independent draws. PL's goal column was 22 picks but only **four players**, two
of whom never featured — that alone produced an apparent 4.5% hit rate. A
two-proportion z-test on raw picks said the leagues differed significantly
(p=0.012); it was invalid for exactly this reason. Report
`distinct_players` alongside `n`, and **suppress the verdict below 6 distinct
players** rather than asserting what the sample cannot support.

---

## 6. Parlays (`leagues/parlays.py`)

Accumulators built from the boards' **own already-published probabilities** —
never a fresh guess. Three sections: all-four-leagues, **Shots & on target**,
Premier-League-only.

**Two rules make the combined number honest:**

1. **One leg per match.** Two legs from one fixture are correlated, so
   multiplying them is a lie. This bites hard: nine published shot-volume picks
   collapse to three usable legs, because Haaland/Semenyo share a fixture, as do
   Mbappé/Vinícius, and Bundesliga props are ungradeable. That is the honest
   count, not a cap.
2. **The same freezing discipline as every other pick.** Frozen at lock time,
   graded from the result, correct only if every leg lands, void if first locked
   after the earliest leg kicked off.

Ungradeable legs are excluded entirely — a leg that can never settle leaves the
whole parlay pending forever.

Do **not** force market variety within a fixture: a player's "2+ attempts" and
"1+ on target" are the same bet twice, so only the stronger can be used.

---

## 7. Picks, freezing and grading (`leagues/picks.py`)

```python
LOCK_WINDOW_HOURS  = 0.75   # 45 min — a confirmed XI (published ~1h out) still reaches the model
MATCHWEEKS_AHEAD   = 1
BEST_PICK_MIN_PROB = 0.65
```

- A pick **locks** on a run inside the lock window and is then frozen. The
  publish cadence must be *tighter than the window* or a fixture slips through
  unlocked, locks after kickoff, and is voided.
- Everything on the card must describe the **frozen** pick, not a fresh argmax —
  on a re-run after the model flips, the card would otherwise show one team and
  grade another.
- Boards are graded **separately**, and player picks **per market** — a 45%
  goalscorer and an 80% shots pick are both near their market's ceiling, so
  pooling them yields a number describing neither.
- **A player with no shot row grades wrong, not void.** The feed cannot separate
  "didn't play" from "played, never shot", so take the harsher reading
  deliberately: it can only understate the record, never inflate it.

---

## 8. Roster verification and name matching (`leagues/players.py`)

The snapshot **corroborates; it does not convict** — except where a club's
roster is complete (`MIN_COMPLETE_ROSTER = 18`), where it is authoritative and a
player absent from it is dropped as departed.

> Treating thin rosters as proof once deleted Real Madrid, Barcelona, Atlético,
> PSG, Marseille and 14 of 18 Bundesliga clubs — Mbappé and Raphinha among them,
> 70% of La Liga and Ligue 1 — because the free feed happened to list fewer than
> 18 names. Absence from incomplete evidence is not evidence of absence.

### Name matching is where the real bugs live

The two feeds spell players differently, and treating every difference as a
departure deleted **51 real first-team players** including Alisson, Ezri Konsa,
Ansu Fati and Kylian Mbappé. Six distinct causes, all of which must be handled:

| Cause | Example |
|---|---|
| Surname is not the last token | `Ezri Konsa Ngoyo` → `E. Konsa` |
| Mononym vs full name | `Alisson` → `Alisson Becker` |
| Name order reversed | `Woo-Yeong Jeong` → `Jeong Woo-Yeong` |
| Letters NFKD does not decompose | `Djordje Petrovic` → `Đ. Petrović` |
| Hyphenated forename abbreviated on its second half | `Gian-Luca` → `L.` |
| Short forms and one-character variants | `Josh`/`Joshua`, `Anssumane`/`Ansu`, `Yeremi`/`Yeremy` |

**Guards that must not be removed:**
- Match only **within the club the player is already at** — so it can never
  invent a transfer.
- Exactly **one** candidate at that club.
- Remaining name parts must be compatible — this is what stops a departed
  `Joao Neves` being revived by team-mate `Ruben Neves`.
- A shared token that is only **both names' first part** is a forename
  collision, not identity (`Marc Cucurella` must not match `Marc Guiu`).
- **Exclude name particles** (`van`, `von`, `der`, `del`…) from match evidence —
  `van Hecke` and `van de Ven` at the same club otherwise both look like
  candidates and the uniqueness guard silently refuses both.
- Apply the compatibility check **while filtering candidates**, not after
  choosing one, or an incompatible near-miss inflates the count and vetoes a
  genuine match.
- **HTML-unescape names at ingest** — API-Football returns `N. O&apos;Reilly`.

### Manual transfer overrides outrank the feed

`transfers.json` exists for the window the feed cannot cover: a move announced
hours ago, where the feed still lists the old club, Understat agrees, and
**nothing looks inconsistent to any automated check**.

> A data-driven sweep cannot find these by construction — it inspects players
> the reconciliation *dropped*, and a player is only dropped once the feed knows
> he left. Only news covers that window.

> And the override must **win**: reconciliation otherwise sees an exact name
> match at the old club and reassigns the player straight back, silently undoing
> it. Bruno Guimarães was set to Arsenal and still published at Newcastle.

Record what a sweep **actually covered** in a `_coverage` field. A bare
`_verified_on` date claims a completeness no sweep achieves.

---

## 9. UI rules

- **The card must not contradict itself.** "Pick: Manchester United" above a
  large "1-1" reads as broken. Lead with the pick-consistent score; keep the
  raw grid mode visible and labelled, because it is measurably the better
  exact-score bet (12.60% vs 10.69% top-1 over 3,958 walk-forward matches — on a
  six-fold that is ~1-in-249k vs ~1-in-671k).
- **Never mix calibrated and uncalibrated numbers in one row.** Showing a
  corrected "32% anytime" beside a raw "3.4 attempts" invites reading them as
  agreeing.
- **Name the team.** A colour-coded avatar is not a label if the palette covers
  10 of 76 clubs.
- **Make holes visible.** A team with no props must be named with the reason,
  not omitted — otherwise a one-sided list looks like a bug.
- **Show the market as disagreement, not edge.** A gap between model and
  bookmaker is at least as likely to mean the market is pricing team news the
  model cannot see.
- Guard `Number(null)` — it is `0`, which **is** finite, and will happily render
  "Bookmakers 0% 0% 0%" while claiming agreement.

---

## 10. Infrastructure

- **`leagues.yml`** — scheduled refresh. Validates hand-edited JSON **first**
  (fails in seconds, not after an hour of model fetching), then syncs rosters,
  news and lineups, publishes, sanity-checks, runs tests, notifies, commits,
  deploys.
- **Deploy dedup** — two deploy paths hit the same metered hook, so skip
  bot-authored commits and commits tagged `[auto-deployed]`.
- **Telegram** — failure alerts, plus each pick sent **exactly once**, the run it
  first locks. The dedup memory must be written *before* the commit step or it
  never persists.
- **Do not register the local wrappers on a schedule.** GitHub Actions already
  runs the same job; two publishers means git conflicts and double deploys.
- **Both abort rather than deploy if a fetch fails** — never ship a
  stale-but-fresh-looking file.
- Render's CDN caches for **5 minutes** (`s-maxage=300`); a deploy is not
  visible instantly. Verify a deploy by grepping the served HTML for a string
  unique to the **new** build — one that also exists in the old build gives a
  false positive.

---

## 11. What was tried and did not work

Do not re-run these hoping for a different answer without new data or a new
method. Each has a script and a JSON artifact.

| Candidate | Result |
|---|---|
| Blending de-vigged 1X2 market odds | Optimal weight on our model: **0.00**. Across 11 subsets too. |
| Over/Under 2.5 as a total-goals signal | 95% CI crosses zero |
| Combined 1X2 + O/U | Significant in La Liga/Bundesliga/Ligue 1, **not PL** — and optimal weight on our own model is still 0.00, so it is market-informed, not alpha |
| Bivariate Poisson (shared shock) | Not significant anywhere |
| Negative binomial marginals | Not significant anywhere |
| Removing rho / plain Poisson | Indistinguishable from the existing model |
| Re-tuning the recency half-life | PL **significantly worse**; the default was already right |
| Naive total-goals rescale | Fixes the level, does **not** improve calibration |
| Conway-Maxwell-Poisson | Not attempted — no closed-form normalising constant, judged impractical |

**The conclusion to internalise:** the model's *architecture* is not the
bottleneck. The wins came from data correctness (roster/name matching,
transfers) and honest calibration — not from a fancier distribution.

---

## 12. Testing philosophy

- Every hard-won lesson gets a **regression test that names the failure**, so a
  future change cannot quietly undo it.
- When a test's assertion changes, **update it with a comment explaining why the
  old invariant no longer holds** — never delete it.
- Statistical claims need a **significance check**, not a point estimate. Raw
  hit-rate deltas are noise-dominated at these sample sizes.
- **Verify your own measurement before believing it.** In this build, a first
  pass reported an 11% hit rate for Ligue 1 (the script was scoring players who
  had left years earlier), a 26,000-sigma gap (standard error computed from an
  observed rate of 0), and a "significant" league difference (a z-test on
  correlated repeated picks). All three were measurement bugs, caught because
  the numbers looked implausible.

---

## 13. Known open items

- **PL anytime-goalscorer is unmeasurable**, not proven bad — too few distinct
  players in one season. The live record will settle it.
- **Bundesliga player picks cannot be graded** (shot events crash upstream), so
  they publish `gradeable: false` and the SOT market is withheld there entirely.
- **Pre-match odds from API-Football return empty** for every fixture tested;
  only in-play works. The football-data.co.uk fixtures feed is the live source
  and populates ~a week before kickoff.
- **The market-calibrated model is not deployed** — it was validated against
  *closing* odds, and this pipeline publishes days earlier. It needs
  genuinely-timestamped early odds before it can be trusted.
