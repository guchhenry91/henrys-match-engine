# Leagues Phase 2b — Four Leagues + Unified Switcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish La Liga, Bundesliga and Ligue 1 alongside the Premier League, and merge the World Cup app and the four league pages into one competition-switcher site.

**Architecture:** The engine is already league-generic (`publish.build(league)`, `names.ALIASES` covers all four leagues). Phase 2b is a loop over leagues plus a data-driven UI merge. The unified page is built as `app.html` behind a copy and only replaces the live `index.html` after the World Cup final (2026-07-19).

**Tech Stack:** Python 3.14 (pandas/scipy/penaltyblog for the league engine; pure stdlib for the untouched WC engine), vanilla JS + CSS, pytest.

**Spec:** `docs/superpowers/specs/2026-07-14-leagues-phase2b-four-leagues-and-unified-ui-design.md`

---

## PRECONDITIONS — do not start until both hold

- [ ] **ClubElo reachable.** `curl -m 10 "http://api.clubelo.com/$(date +%Y-%m-%d)"` returns HTTP 200 with >100 rows. Every league's promoted-club priors depend on it; building during an outage ships degraded tables. (As of 2026-07-14 it is DOWN — a watch is running.)
- [ ] **Build-phase 6 (the `index.html` swap) waits until after 2026-07-19.** Build-phases 1-5 and 7 may proceed before then; only the swap is date-gated.

---

## File Structure

```
worldcup/
  leagues/publish.py            MODIFY — main() loops 4 leagues
  data/leagues/
    clubs.json                  MODIFY — becomes league-keyed, or add:
    clubs_laliga.json           NEW
    clubs_bundesliga.json       NEW
    clubs_ligue1.json           NEW
    laliga.json                 NEW (generated)
    bundesliga.json             NEW (generated)
    ligue1.json                 NEW (generated)
  app.html                      NEW — unified switcher page (becomes index.html post-final)
  app.css                       NEW — shared component styles (leagues.css + WC styles merged)
  leagues/wc_adapter.py         NEW — maps predictions.json onto the shared view contract (if a Python-side transform is cleaner than JS; else adapter lives in app.html)
  predict.py                    MODIFY (build-phase 5) — emit `backtest`/record surfacing only; no field removed
  ops/leagues_weekly.py         MODIFY — loop 4 leagues
  ops/leagues_matchday.py       MODIFY — loop 4 leagues
  tests/leagues/
    test_publish_multi.py       NEW
    test_clubs_coverage.py      NEW
```

The WC engine (`predict.py`) stays pure-stdlib; build-phase 5 only *adds* a surfaced record, verified to change no existing field.

---

## Build-phase 1: Name-map + colour completeness (all 4 leagues)

**Files:**
- Modify: `leagues/names.py` (fill gaps only)
- Create: `data/leagues/clubs_laliga.json`, `clubs_bundesliga.json`, `clubs_ligue1.json`
- Create: `tests/leagues/test_clubs_coverage.py`

- [ ] **Step 1: Write the coverage test (drives what must exist)**

```python
# tests/leagues/test_clubs_coverage.py
import json
from pathlib import Path

import pytest

from leagues import config, fixtures
from leagues.names import canonical, UnknownTeam

ROOT = Path(__file__).resolve().parents[2]
CLUBS = {"PL": "clubs.json", "LALIGA": "clubs_laliga.json",
         "BUNDESLIGA": "clubs_bundesliga.json", "LIGUE1": "clubs_ligue1.json"}


@pytest.mark.parametrize("league", list(config.LEAGUES))
def test_every_fixture_team_maps_and_has_a_colour(league):
    fx = fixtures.fetch_fixtures(league)          # live feed (cached)
    teams = sorted(set(fx["home"]) | set(fx["away"]))
    colours = json.loads((ROOT / "data" / "leagues" / CLUBS[league]).read_text("utf-8"))
    missing_colour = [t for t in teams if t not in colours]
    assert not missing_colour, f"{league}: no colour for {missing_colour}"
    # names already canonical (fetch_fixtures raises UnknownTeam otherwise) — assert shape
    for t in teams:
        assert "primary" in colours[t] and "short" in colours[t]
```

- [ ] **Step 2: Run it to see the real gaps**

Run: `python -m pytest tests/leagues/test_clubs_coverage.py -v`
Expected: FAILs listing the exact promoted/unmapped clubs per league. `fetch_fixtures` itself raises `UnknownTeam` for any name gap in `names.py` — fix those first (add aliases), then the colour gaps.

- [ ] **Step 3: Fill name-map gaps in `names.py`**

For each `UnknownTeam` raised, add the source spelling to the correct canonical entry's alias set (the tables already exist per league). Do **not** invent a canonical club — match it to the real promoted/renamed side.

- [ ] **Step 4: Create the three colour maps**

One file per league, canonical names exactly as `names.py` emits them, same shape as the PL `clubs.json`:
```json
{ "Real Madrid": {"primary": "#FEBE10", "secondary": "#00529F", "short": "RMA"}, ... }
```
Use each club's real primary/secondary kit colours. Every team the fixture feed lists must have an entry.

- [ ] **Step 5: Run to green**

Run: `python -m pytest tests/leagues/test_clubs_coverage.py -v`
Expected: 4 passed (one per league).

- [ ] **Step 6: Commit**

```bash
git add leagues/names.py data/leagues/clubs_*.json tests/leagues/test_clubs_coverage.py
git commit -m "feat(leagues): name-map + colour coverage for all four leagues"
```

---

## Build-phase 2: Generalise `publish.main()` to four leagues

**Files:**
- Modify: `leagues/publish.py`
- Create: `tests/leagues/test_publish_multi.py`

- [ ] **Step 1: Write the failing test** (pure — feeds a stub `build`)

```python
# tests/leagues/test_publish_multi.py
import json
import leagues.publish as publish


def test_main_writes_one_atomic_file_per_league(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)
    monkeypatch.setattr(publish, "build",
                        lambda lg: {"league": lg, "matches": [], "table": [],
                                    "missing_squads": [], "data_warnings": []})
    publish.main()
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == ["bundesliga.json", "laliga.json", "ligue1.json", "pl.json"]
    # no leftover temp files
    assert not list(tmp_path.glob("*.tmp"))


def test_one_league_failing_does_not_block_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)

    def flaky(lg):
        if lg == "LALIGA":
            raise RuntimeError("simulated fetch failure")
        return {"league": lg, "matches": [], "table": [],
                "missing_squads": [], "data_warnings": []}

    monkeypatch.setattr(publish, "build", flaky)
    publish.main()                         # must not raise
    written = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert "laliga" not in written         # the failing one is skipped
    assert "pl" in written and "bundesliga" in written
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/leagues/test_publish_multi.py -v`
Expected: FAIL — current `main()` writes only `pl.json`.

- [ ] **Step 3: Rewrite `main()` as a per-league loop**

```python
FILE_FOR = {"PL": "pl.json", "LALIGA": "laliga.json",
            "BUNDESLIGA": "bundesliga.json", "LIGUE1": "ligue1.json"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for league, fname in FILE_FOR.items():
        try:
            payload = build(league)
        except Exception as exc:            # one league's outage must not sink the rest
            print(f"ABORT {league}: {exc}; leaving its file untouched")
            continue
        path = OUT / fname
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)                   # atomic
        print(f"wrote {path} - {len(payload['matches'])} fixtures, "
              f"{len(payload['table'])} teams")
        if payload["missing_squads"]:
            print(f"  WARNING {league}: no player data for {payload['missing_squads']}")
```

Keep the PL-only `build("PL")` behaviour reachable for quick iteration via a `if __name__` arg, e.g. `python -m leagues.publish PL`.

- [ ] **Step 4: Run to green**

Run: `python -m pytest tests/leagues/test_publish_multi.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add leagues/publish.py tests/leagues/test_publish_multi.py
git commit -m "feat(leagues): publish all four leagues, aborting per-league on failure"
```

---

## Build-phase 3: Per-league gates (match + props)

**Files:** none new — runs existing `leagues.tune` and `leagues.props_backtest`, records reports.

- [ ] **Step 1: Re-confirm the match gate for all four**

Run: `python -m leagues.tune`
Expected: `data-raw/leagues/backtest_report.json` with all four in-band (PL +0.0061, La Liga +0.0055, Bundesliga +0.0066, Ligue 1 +0.0058 as of 2026-07-14). Any league that has drifted out of the +0.005…+0.02 RPS-gap band is held back.

- [ ] **Step 2: Run the props gate per league**

`props_backtest.run` currently defaults to PL. Run it for each key:
```bash
python -c "
import json,io
from leagues import props_backtest
for lg in ['PL','LALIGA','BUNDESLIGA','LIGUE1']:
    rep = props_backtest.run(lg)
    io.open(f'data-raw/leagues/props_report_{lg.lower()}.json','w',encoding='utf-8').write(json.dumps(rep,indent=2))
    print(lg, 'PASS' if rep['passes'] else 'FAIL', 'goals', rep['goals_lift'], 'shots', rep['shots_lift'])
"
```
Expected: each beats its baseline (positive `goals_lift` and `shots_lift`). **A league that fails does not go in the switcher** — report it, do not ship it.

- [ ] **Step 3: Resolve the Bundesliga `xg_weight=1.0` question**

Bundesliga tuned to pure xG. Confirm it is signal, not a join bug:
```bash
python -c "
from leagues import dataset
d = dataset.build_matches('BUNDESLIGA')
print('xG coverage', d['home_xg'].notna().mean())
"
```
Expected: coverage ~100%. If coverage is low, the pure-xG win is an artefact of a bad join — fix the join (name mismatches, season codes) before trusting the number.

- [ ] **Step 4: Commit the reports**

```bash
git add data-raw/leagues/backtest_report.json data-raw/leagues/props_report_*.json
git commit -m "test(leagues): per-league match + props gate reports"
```

- [ ] **Step 5: STOP — report the four-league gate table to the user** before building UI on top.

---

## Build-phase 4: `app.html` — unified switcher + shared view layer

**Files:**
- Create: `app.html`, `app.css`

Built as a copy. `index.html` is NOT touched in this phase.

- [ ] **Step 1: Publish all four so real data exists**

Run: `python -m leagues.publish` — writes `pl.json`, `laliga.json`, `bundesliga.json`, `ligue1.json`.

- [ ] **Step 2: Build the competition manifest + switcher shell**

`app.html` opens with a manifest and a switcher bound to it:
```html
<script>
const COMPETITIONS = [
  {key: "wc",         label: "World Cup",      data: "data/predictions.json", kind: "cup",    clubs: null},
  {key: "pl",         label: "Premier League", data: "data/leagues/pl.json",  kind: "league", clubs: "data/leagues/clubs.json"},
  {key: "laliga",     label: "La Liga",        data: "data/leagues/laliga.json",     kind: "league", clubs: "data/leagues/clubs_laliga.json"},
  {key: "bundesliga", label: "Bundesliga",     data: "data/leagues/bundesliga.json", kind: "league", clubs: "data/leagues/clubs_bundesliga.json"},
  {key: "ligue1",     label: "Ligue 1",        data: "data/leagues/ligue1.json",     kind: "league", clubs: "data/leagues/clubs_ligue1.json"},
];
// switcher writes location.hash; on load, hash picks the competition (default wc)
</script>
```
Selecting a competition fetches its data (+ clubs for leagues), then dispatches on `kind`.

- [ ] **Step 3: Port the shared components from `leagues.html`**

Move the fixture-card, projected-table, props-list and performance-panel renderers from `leagues.html` into `app.html` unchanged (they are already data-driven). The **league view** = record chip + fixtures + table + performance, exactly as `leagues.html` renders today.

- [ ] **Step 4: Add the WC view via an adapter**

The WC payload (`predictions.json`) has `matches`, `record`, `knockout`, `standings` — a different shape. Write a small adapter (in JS, at the top of the view) that maps it onto the same card/table components: WC shows the **bracket + group standings + pick record**, no relegation/title-odds table. Reuse the fixture card for WC matches. Do **not** modify `predictions.json`.

- [ ] **Step 5: Verify every competition renders from real data**

Serve and drive each tab:
```bash
python -m http.server 8765 &
```
For each of the five competitions, confirm in the DOM: cards render, no JS errors, the switcher highlights the active tab, and the data is that competition's (not a stale carry-over). The WC tab must match what the live site shows today (same picks, same record).

- [ ] **Step 6: Commit**

```bash
git add app.html app.css
git commit -m "feat(ui): unified competition switcher (WC + 4 leagues), built as app.html"
```

---

## Build-phase 5: WC performance panel (isolated, no regression)

**Files:**
- Modify: `predict.py` (surface the existing record for the panel; add nothing that changes a published field)

The WC engine has **no market-odds backtest** (no odds data), so its performance panel is the **honest pick record it already tracks** (`record` + `by_confidence`), not an RPS-vs-market panel. `predictions.json` already carries `record`, so the panel may need **no** engine change at all.

- [ ] **Step 1: Check whether `predictions.json` already has what the panel needs**

Run:
```bash
python -c "import json; d=json.load(open('data/predictions.json')); print('record' in d, list(d['record'].keys()))"
```
If `record` + `by_confidence` are present (they are, as of 2026-07-14), the panel is pure UI — skip to Step 3.

- [ ] **Step 2 (only if a field is missing): add it TDD-first**

Write a test asserting the new field appears AND that every pre-existing top-level key is unchanged, then add the minimal emission to `predict.py`. Run `python predict.py`, diff `predictions.json` against `git HEAD` and confirm only the new key was added.

- [ ] **Step 3: Render the WC performance panel in `app.html`**

In the WC view, show the record chip + a by-confidence breakdown (correct/total per confidence band), reusing the league performance panel's markup. State it honestly: this is the live pick record, not a backtest.

- [ ] **Step 4: Verify**

Confirm the WC tab shows the record (62-24 etc. as of 2026-07-14) and the by-confidence table, and that `git diff HEAD -- data/predictions.json predict.py` is empty if Step 2 was skipped.

- [ ] **Step 5: Commit**

```bash
git add app.html predict.py tests/  # predict.py + tests only if Step 2 ran
git commit -m "feat(ui): World Cup model-performance panel (honest pick record)"
```

---

## Build-phase 6: Swap `index.html` — AFTER 2026-07-19 ONLY

**Files:**
- Rename/replace: `index.html` ← `app.html`

- [ ] **Step 1: Confirm the date gate**

Do not proceed before 2026-07-20. The World Cup final is 2026-07-19; the live app must not change during it.

- [ ] **Step 2: Screenshot the current live WC app first**

Capture `index.html` (WC view) as it is now, so the post-swap WC tab can be diffed against it.

- [ ] **Step 3: Swap**

```bash
git mv index.html index_wc_legacy.html    # keep the old file in history + on disk as a fallback
git mv app.html index.html
# fix any asset paths (app.css stays; ensure index.html references it)
```
The old WC app remains reachable at `index_wc_legacy.html` until the unified WC view is confirmed equivalent.

- [ ] **Step 4: Verify the WC view is unchanged for a user**

Serve, open `index.html`, default to the WC tab, and confirm it matches the pre-swap screenshot: same picks, same record, same bracket. Then click through all five tabs.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: unified predictor is now the site entry point (post-WC-final swap)"
```

---

## Build-phase 7: Ops — refresh all four leagues

**Files:**
- Modify: `ops/leagues_weekly.py`, `ops/leagues_matchday.py`

- [ ] **Step 1: Confirm the jobs already call the generalised publish**

Both jobs call `leagues.publish.main()`, which now loops all four leagues (build-phase 2). If so, no change is needed beyond confirming the abort-per-league behaviour and re-running the dry run:

Run: `python -c "from leagues import publish; publish.main()"`
Expected: four files rewritten (or fewer, with a clear ABORT line per league that failed); `git status` shows only league files changed — **no WC files.**

- [ ] **Step 2: Update the registration note in `CLAUDE.md`**

The mid-August registration note already exists; confirm it refers to all four leagues, not just the PL. Do not register the tasks until mid-August (season starts 2026-08-21).

- [ ] **Step 3: Commit**

```bash
git add ops/ CLAUDE.md
git commit -m "chore(leagues): ops jobs refresh all four leagues"
```

---

## Build-phase 8: Full check and handoff

- [ ] **Step 1: Full test suite** — `python -m pytest tests/ -q` — expect all green.
- [ ] **Step 2: Confirm the WC engine is untouched** (until the phase-6 swap): `git diff main --stat -- predict.py data/predictions.json data-raw/results.json` empty before build-phase 6.
- [ ] **Step 3: Report the four-league gate table + a screenshot of each switcher tab, and confirm nothing is deployed until the user approves** (per the standing deploy-only-when-verified rule).

---

## Self-Review Notes

- **Spec coverage:** §2 scope → build-phases 1-7; per-league gates §3 → build-phase 3 (incl. the Bundesliga xg_weight check); unified UI §4 → build-phase 4; WC contract adapter §5 → build-phase 4 step 4 + build-phase 5; risks §7 → the date gate (phase 6), ClubElo precondition, and the `app.html`-behind-a-copy pattern.
- **Honesty rule enforced:** a league that fails either gate is held out of the switcher (build-phase 3 step 2/5), not shipped silently.
- **Live-app safety:** `index.html` is untouched until build-phase 6, which is date-gated to after the WC final and keeps the old file as `index_wc_legacy.html`.
- **Known deferrals, not gaps:** CL/cups; second-tier data for promoted-club player props. Both explicitly out of scope per the parent spec.
- **Provisional in the plan:** whether the WC adapter is cleaner in JS (in `app.html`) or as `leagues/wc_adapter.py` is decided at build-phase 4 step 4 by which keeps `predict.py` untouched — the plan mandates the outcome (no WC-engine rewrite), not the mechanism.
