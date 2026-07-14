# Multi-League Predictor — Phase 1: Data Layer + Model + Backtest

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline and the match model for 4 leagues (PL, La Liga, Bundesliga, Ligue 1), and prove via walk-forward backtest that it approaches bookmaker accuracy — the hard gate before any UI work.

**Architecture:** A `leagues/` Python package inside the existing `worldcup` repo. Data is pulled from free sources into `data-raw/leagues/<lg>/`, a Dixon-Coles model is fitted with `penaltyblog` on actual goals (for `rho`/home-advantage/time-weighted strengths), xG-derived strengths are computed separately and blended at the *strength* level (penaltyblog requires integer goals, so xG cannot be passed as the response), promoted teams are shrunk toward ClubElo priors, and the resulting 1X2 is ensembled with pi-ratings and isotonically calibrated. A walk-forward backtest scores it against de-vigged bookmaker closing odds.

**Tech Stack:** Python 3.14, `penaltyblog` 1.11, `soccerdata` 1.9, pandas 3, scipy, scikit-learn (isotonic), pytest.

**Spec:** `docs/superpowers/specs/2026-07-13-multileague-predictor-design.md`

---

## File Structure

```
worldcup/
  requirements.txt              NEW — pins league-engine deps
  leagues/
    __init__.py                 NEW
    config.py                   NEW — per-league codes, seasons, sources
    names.py                    NEW — cross-source team-name normalization
    fixtures.py                 NEW — fixturedownload feed loader
    history.py                  NEW — football-data.co.uk results + odds loader
    xg.py                       NEW — Understat/FBref xG loader (soccerdata)
    elo.py                      NEW — ClubElo prior loader
    dataset.py                  NEW — assembles the unified match table
    weights.py                  NEW — exponential time-decay
    model.py                    NEW — DC fit + xG blend + priors + ensemble + calibration
    backtest.py                 NEW — walk-forward validation vs de-vigged odds
  tests/leagues/
    test_names.py               NEW
    test_fixtures.py            NEW
    test_history.py             NEW
    test_weights.py             NEW
    test_model.py               NEW
    test_backtest.py            NEW
  data-raw/leagues/<lg>/        NEW — cached raw data (gitignored except manifests)
```

**Responsibilities:** each loader module owns exactly one source and returns a clean pandas DataFrame with canonical team names. `dataset.py` is the only module that joins them. `model.py` never touches the network.

---

## Task 1: Scaffold package, deps, and league config

**Files:**
- Create: `requirements.txt`
- Create: `leagues/__init__.py`
- Create: `leagues/config.py`
- Create: `tests/leagues/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
penaltyblog==1.11.0
soccerdata==1.9.0
pandas>=3.0
scipy>=1.18
numpy>=2.0
scikit-learn>=1.9
pytest>=8.0
```

- [ ] **Step 2: Create the package with league config**

`leagues/__init__.py` — empty file.

`leagues/config.py`:

```python
"""Per-league configuration. One entry per competition; the engine is generic."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class League:
    key: str              # our canonical key, used in paths/URLs
    name: str             # display name
    fd_code: str          # football-data.co.uk division code
    fixture_slug: str     # fixturedownload.com slug for 2026-27
    understat: str        # soccerdata/Understat league id
    fbref: str            # soccerdata/FBref league id
    n_teams: int
    relegation_spots: int
    europe_spots: int     # top-N qualifying for the Champions League
    # 5 completed seasons used to fit, as football-data.co.uk season codes
    history_seasons: tuple = ("2122", "2223", "2324", "2425", "2526")


LEAGUES = {
    "PL": League("PL", "Premier League", "E0", "epl-2026",
                 "ENG-Premier League", "ENG-Premier League", 20, 3, 4),
    "LALIGA": League("LALIGA", "La Liga", "SP1", "la-liga-2026",
                     "ESP-La Liga", "ESP-La Liga", 20, 3, 4),
    "BUNDESLIGA": League("BUNDESLIGA", "Bundesliga", "D1", "bundesliga-2026",
                         "GER-Bundesliga", "GER-Bundesliga", 18, 2, 4),
    "LIGUE1": League("LIGUE1", "Ligue 1", "F1", "ligue-1-2026",
                     "FRA-Ligue 1", "FRA-Ligue 1", 18, 2, 4),
}


def get(key: str) -> League:
    if key not in LEAGUES:
        raise KeyError(f"unknown league {key!r}; known: {sorted(LEAGUES)}")
    return LEAGUES[key]
```

- [ ] **Step 3: Verify config loads**

Run: `python -c "from leagues import config; print(config.get('PL').name, config.get('PL').n_teams)"`
Expected: `Premier League 20`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt leagues/__init__.py leagues/config.py tests/leagues/__init__.py
git commit -m "feat(leagues): scaffold package and league config"
```

---

## Task 2: Team-name normalization

Different sources spell clubs differently ("Deportivo Alavés" vs "Alaves", "Olympique de Marseille" vs "Marseille"). Everything downstream keys on a **canonical** name. Unmapped names must fail loudly, never silently drop a team.

**Files:**
- Create: `leagues/names.py`
- Test: `tests/leagues/test_names.py`

- [ ] **Step 1: Write the failing test**

`tests/leagues/test_names.py`:

```python
import pytest
from leagues.names import canonical, UnknownTeam


def test_canonical_passes_through_known_canonical_name():
    assert canonical("Arsenal", "PL") == "Arsenal"


def test_canonical_maps_source_aliases():
    assert canonical("Man United", "PL") == "Manchester United"
    assert canonical("Man Utd", "PL") == "Manchester United"


def test_canonical_strips_accents_and_suffixes():
    assert canonical("Deportivo Alavés", "LALIGA") == "Alaves"
    assert canonical("Olympique de Marseille", "LIGUE1") == "Marseille"


def test_unknown_team_raises_loudly():
    with pytest.raises(UnknownTeam):
        canonical("Nonexistent FC", "PL")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_names.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'leagues.names'`

- [ ] **Step 3: Implement names.py**

```python
"""Canonical team names. Every source (football-data.co.uk, fixturedownload,
Understat, FBref, ClubElo) spells clubs differently — normalize once, here."""
import unicodedata


class UnknownTeam(Exception):
    """Raised when a source name has no canonical mapping — never guess."""


# canonical -> set of aliases seen across sources
ALIASES = {
    "PL": {
        "Manchester United": {"Man United", "Man Utd", "Manchester Utd"},
        "Manchester City": {"Man City"},
        "Newcastle United": {"Newcastle", "Newcastle Utd"},
        "Tottenham": {"Tottenham Hotspur", "Spurs"},
        "Wolves": {"Wolverhampton Wanderers", "Wolverhampton"},
        "Nottingham Forest": {"Nott'm Forest", "Nottingham"},
        "Brighton": {"Brighton & Hove Albion", "Brighton and Hove Albion"},
        "West Ham": {"West Ham United"},
        "Leeds": {"Leeds United"},
        "Sunderland": set(),
        "Coventry": {"Coventry City"},
        "Arsenal": set(), "Chelsea": set(), "Liverpool": set(), "Everton": set(),
        "Aston Villa": set(), "Fulham": set(), "Brentford": set(),
        "Crystal Palace": set(), "Bournemouth": {"AFC Bournemouth"},
    },
    "LALIGA": {
        "Alaves": {"Deportivo Alavés", "Deportivo Alaves", "Alavés"},
        "Ath Bilbao": {"Athletic Club", "Athletic Bilbao"},
        "Ath Madrid": {"Atlético Madrid", "Atletico Madrid", "Atlético de Madrid"},
        "Barcelona": {"FC Barcelona"},
        "Real Madrid": {"Real Madrid CF"},
        "Sociedad": {"Real Sociedad"},
        "Betis": {"Real Betis"},
        "Celta": {"Celta Vigo", "RC Celta"},
        "Getafe": {"Getafe CF"},
        "Sevilla": {"Sevilla FC"},
        "Valencia": {"Valencia CF"},
        "Villarreal": {"Villarreal CF"},
        "Espanol": {"Espanyol", "RCD Espanyol"},
    },
    "BUNDESLIGA": {
        "Bayern Munich": {"FC Bayern München", "Bayern München", "Bayern Munchen"},
        "Dortmund": {"Borussia Dortmund", "BVB"},
        "Leverkusen": {"Bayer 04 Leverkusen", "Bayer Leverkusen"},
        "M'gladbach": {"Borussia Mönchengladbach", "Borussia Monchengladbach"},
        "Ein Frankfurt": {"Eintracht Frankfurt"},
        "Stuttgart": {"VfB Stuttgart"},
        "Wolfsburg": {"VfL Wolfsburg"},
        "RB Leipzig": {"RasenBallsport Leipzig"},
        "Union Berlin": {"1. FC Union Berlin"},
        "Werder Bremen": {"SV Werder Bremen"},
        "Hoffenheim": {"TSG 1899 Hoffenheim", "TSG Hoffenheim"},
        "Freiburg": {"SC Freiburg"},
        "Mainz": {"1. FSV Mainz 05", "Mainz 05"},
        "Augsburg": {"FC Augsburg"},
        "Heidenheim": {"1. FC Heidenheim"},
        "St Pauli": {"FC St. Pauli", "St. Pauli"},
        "Hamburg": {"Hamburger SV"},
        "Elversberg": {"SV Elversberg"},
    },
    "LIGUE1": {
        "Marseille": {"Olympique de Marseille", "Olympique Marseille"},
        "Paris SG": {"Paris Saint-Germain", "PSG", "Paris Saint Germain"},
        "Lyon": {"Olympique Lyonnais"},
        "Monaco": {"AS Monaco"},
        "Lille": {"LOSC Lille", "LOSC"},
        "Nice": {"OGC Nice"},
        "Rennes": {"Stade Rennais", "Stade Rennais FC"},
        "Lens": {"RC Lens"},
        "Strasbourg": {"RC Strasbourg Alsace", "RC Strasbourg"},
        "Nantes": {"FC Nantes"},
        "Toulouse": {"Toulouse FC"},
        "Brest": {"Stade Brestois", "Stade Brestois 29"},
        "Auxerre": {"AJ Auxerre"},
        "Angers": {"Angers SCO"},
        "Le Havre": {"Le Havre AC"},
        "Metz": {"FC Metz"},
        "Lorient": {"FC Lorient"},
        "Paris FC": {"Paris FC"},
    },
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _key(s: str) -> str:
    return _strip_accents(s).strip().lower()


# build reverse lookup once: normalized alias -> canonical
_LOOKUP: dict[str, dict[str, str]] = {}
for _lg, _mapping in ALIASES.items():
    table = {}
    for _canon, _aliases in _mapping.items():
        table[_key(_canon)] = _canon
        for _a in _aliases:
            table[_key(_a)] = _canon
    _LOOKUP[_lg] = table


def canonical(name: str, league: str) -> str:
    """Map any source spelling to our canonical club name."""
    table = _LOOKUP.get(league, {})
    hit = table.get(_key(name))
    if hit is None:
        raise UnknownTeam(f"{name!r} is not mapped for league {league!r}. "
                          f"Add it to leagues/names.py ALIASES.")
    return hit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_names.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add leagues/names.py tests/leagues/test_names.py
git commit -m "feat(leagues): canonical team-name normalization with loud failure"
```

---

## Task 3: Fixture loader (fixturedownload)

**Files:**
- Create: `leagues/fixtures.py`
- Test: `tests/leagues/test_fixtures.py`

The feed is verified live: `https://fixturedownload.com/feed/json/epl-2026` returns 380 rows with keys `MatchNumber, RoundNumber, DateUtc, Location, HomeTeam, AwayTeam, HomeTeamScore, AwayTeamScore`.

- [ ] **Step 1: Write the failing test** (parses a fixed payload — no network in tests)

`tests/leagues/test_fixtures.py`:

```python
from leagues.fixtures import parse_fixtures

RAW = [
    {"MatchNumber": 1, "RoundNumber": 1, "DateUtc": "2026-08-21 19:00:00Z",
     "Location": "Emirates Stadium", "HomeTeam": "Arsenal", "AwayTeam": "Coventry",
     "HomeTeamScore": None, "AwayTeamScore": None},
    {"MatchNumber": 2, "RoundNumber": 1, "DateUtc": "2026-08-22 14:00:00Z",
     "Location": "Anfield", "HomeTeam": "Liverpool", "AwayTeam": "Man City",
     "HomeTeamScore": 2, "AwayTeamScore": 1},
]


def test_parse_fixtures_normalizes_names_and_types():
    df = parse_fixtures(RAW, "PL")
    assert len(df) == 2
    assert list(df.columns) == ["match_id", "round", "date", "venue",
                                "home", "away", "home_goals", "away_goals", "played"]
    assert df.loc[0, "home"] == "Arsenal"
    assert df.loc[1, "away"] == "Manchester City"   # alias normalized


def test_parse_fixtures_marks_played_only_when_both_scores_present():
    df = parse_fixtures(RAW, "PL")
    assert bool(df.loc[0, "played"]) is False
    assert bool(df.loc[1, "played"]) is True
    assert df.loc[1, "home_goals"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'leagues.fixtures'`

- [ ] **Step 3: Implement fixtures.py**

```python
"""2026-27 fixtures (and live results) from fixturedownload.com JSON feeds."""
import json
import urllib.request

import pandas as pd

from leagues import config
from leagues.names import canonical

FEED = "https://fixturedownload.com/feed/json/{slug}"


def parse_fixtures(raw: list[dict], league: str) -> pd.DataFrame:
    """Pure parser — takes the decoded JSON list, returns a clean DataFrame."""
    rows = []
    for r in raw:
        hg, ag = r.get("HomeTeamScore"), r.get("AwayTeamScore")
        played = hg is not None and ag is not None
        rows.append({
            "match_id": r["MatchNumber"],
            "round": r["RoundNumber"],
            "date": pd.to_datetime(r["DateUtc"], utc=True),
            "venue": r.get("Location") or "",
            "home": canonical(r["HomeTeam"], league),
            "away": canonical(r["AwayTeam"], league),
            "home_goals": int(hg) if played else pd.NA,
            "away_goals": int(ag) if played else pd.NA,
            "played": played,
        })
    return pd.DataFrame(rows, columns=["match_id", "round", "date", "venue",
                                       "home", "away", "home_goals", "away_goals",
                                       "played"])


def fetch_fixtures(league: str) -> pd.DataFrame:
    """Download the season's fixtures+results for one league."""
    slug = config.get(league).fixture_slug
    with urllib.request.urlopen(FEED.format(slug=slug), timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return parse_fixtures(raw, league)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_fixtures.py -v`
Expected: 2 passed

- [ ] **Step 5: Smoke-test the live feed for all 4 leagues**

Run:
```bash
python -c "
from leagues import fixtures, config
for lg in config.LEAGUES:
    df = fixtures.fetch_fixtures(lg)
    print(lg, len(df), 'fixtures,', df['home'].nunique(), 'teams')
"
```
Expected: `PL 380 fixtures, 20 teams` / `LALIGA 380 ... 20` / `BUNDESLIGA 306 ... 18` / `LIGUE1 306 ... 18`.
If an `UnknownTeam` is raised, add the alias to `leagues/names.py` and re-run — this is the intended loud failure.

- [ ] **Step 6: Commit**

```bash
git add leagues/fixtures.py tests/leagues/test_fixtures.py leagues/names.py
git commit -m "feat(leagues): fixturedownload fixture+result loader"
```

---

## Task 4: Historical results + closing odds (football-data.co.uk)

This is the model's training backbone and the bookmaker benchmark.

**Files:**
- Create: `leagues/history.py`
- Test: `tests/leagues/test_history.py`

- [ ] **Step 1: Write the failing test**

`tests/leagues/test_history.py`:

```python
import io
import pandas as pd
from leagues.history import parse_history

CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgCH,AvgCD,AvgCA\n"
    "E0,16/08/2025,Liverpool,Man Utd,2,1,H,1.80,3.60,4.50\n"
    "E0,17/08/2025,Arsenal,Chelsea,0,0,D,2.10,3.40,3.60\n"
)


def test_parse_history_normalizes_and_types():
    df = parse_history(io.StringIO(CSV), "PL", season="2526")
    assert len(df) == 2
    assert df.loc[0, "home"] == "Liverpool"
    assert df.loc[0, "away"] == "Manchester United"
    assert df.loc[0, "home_goals"] == 2
    assert df.loc[0, "season"] == "2526"
    assert isinstance(df.loc[0, "date"], pd.Timestamp)


def test_parse_history_keeps_closing_odds():
    df = parse_history(io.StringIO(CSV), "PL", season="2526")
    assert df.loc[0, "odds_h"] == 1.80
    assert df.loc[0, "odds_d"] == 3.60
    assert df.loc[0, "odds_a"] == 4.50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_history.py -v`
Expected: FAIL — no module `leagues.history`

- [ ] **Step 3: Implement history.py**

```python
"""5 seasons of results + closing odds from football-data.co.uk."""
import io
import urllib.request

import pandas as pd

from leagues import config
from leagues.names import canonical

URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
# Average closing odds; fall back to Bet365 closing, then Bet365 pre-match.
ODDS_SETS = [("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA"),
             ("B365H", "B365D", "B365A")]


def parse_history(buf, league: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(buf, encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "home": [canonical(t, league) for t in df["HomeTeam"]],
        "away": [canonical(t, league) for t in df["AwayTeam"]],
        "home_goals": df["FTHG"].astype(int).values,
        "away_goals": df["FTAG"].astype(int).values,
    })
    for h, d, a in ODDS_SETS:
        if h in df.columns:
            out["odds_h"] = pd.to_numeric(df[h], errors="coerce").values
            out["odds_d"] = pd.to_numeric(df[d], errors="coerce").values
            out["odds_a"] = pd.to_numeric(df[a], errors="coerce").values
            break
    else:
        out["odds_h"] = out["odds_d"] = out["odds_a"] = pd.NA
    return out.dropna(subset=["date"]).reset_index(drop=True)


def fetch_history(league: str) -> pd.DataFrame:
    """All configured history seasons for a league, concatenated."""
    lg = config.get(league)
    frames = []
    for season in lg.history_seasons:
        url = URL.format(season=season, div=lg.fd_code)
        with urllib.request.urlopen(url, timeout=60) as resp:
            buf = io.StringIO(resp.read().decode("latin-1"))
        frames.append(parse_history(buf, league, season))
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_history.py -v`
Expected: 2 passed

- [ ] **Step 5: Smoke-test live download for all 4 leagues**

Run:
```bash
python -c "
from leagues import history, config
for lg in config.LEAGUES:
    df = history.fetch_history(lg)
    print(lg, len(df), 'matches', df['date'].min().date(), '->', df['date'].max().date(),
          '| odds coverage', round(df['odds_h'].notna().mean()*100), '%')
"
```
Expected: ~1,900 matches for PL/LALIGA (5 × 380), ~1,530 for BUNDESLIGA/LIGUE1 (5 × 306), odds coverage near 100%.
Add any `UnknownTeam` aliases to `names.py` and re-run until clean.

- [ ] **Step 6: Commit**

```bash
git add leagues/history.py tests/leagues/test_history.py leagues/names.py
git commit -m "feat(leagues): football-data.co.uk history + closing odds loader"
```

---

## Task 5: xG loader (Understat via soccerdata)

**Files:**
- Create: `leagues/xg.py`

Understat is the primary xG source (lighter and less likely to rate-limit than FBref). We need **per-match team xG** to build xG-based strengths.

- [ ] **Step 1: Implement xg.py**

```python
"""Per-match team xG from Understat (via soccerdata, disk-cached)."""
import pandas as pd
import soccerdata as sd

from leagues import config
from leagues.names import canonical

# Understat seasons are the starting year: 2021 => 2021/22
SEASONS = [2021, 2022, 2023, 2024, 2025]


def fetch_team_match_xg(league: str) -> pd.DataFrame:
    """Return one row per team-match: date, team, opponent, home flag, xg, xga."""
    lg = config.get(league)
    us = sd.Understat(leagues=lg.understat, seasons=SEASONS)
    raw = us.read_team_match_stats().reset_index()

    cols = {c.lower(): c for c in raw.columns}
    date_c = cols.get("date")
    team_c = cols.get("team")
    xg_c = cols.get("xg")
    xga_c = cols.get("xga")
    ha_c = cols.get("home_away") or cols.get("venue") or cols.get("h_a")
    missing = [n for n, c in [("date", date_c), ("team", team_c), ("xg", xg_c)] if c is None]
    if missing:
        raise RuntimeError(f"Understat schema changed; missing {missing}. Got: {list(raw.columns)}")

    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date_c]),
        "team": [canonical(t, league) for t in raw[team_c]],
        "xg": pd.to_numeric(raw[xg_c], errors="coerce"),
        "xga": pd.to_numeric(raw[xga_c], errors="coerce") if xga_c else pd.NA,
        "is_home": raw[ha_c].astype(str).str.lower().str[0].eq("h") if ha_c else pd.NA,
    })
    return out.dropna(subset=["xg"]).reset_index(drop=True)
```

- [ ] **Step 2: Smoke-test (network + cache; slow on first run)**

Run:
```bash
python -c "
from leagues import xg
df = xg.fetch_team_match_xg('PL')
print(len(df), 'team-matches'); print(df.head(3))
print('teams:', df['team'].nunique())
"
```
Expected: several thousand team-match rows, 20+ teams (more than 20 across 5 seasons due to promotion/relegation).
If Understat's schema differs, the `RuntimeError` tells you exactly which columns exist — adjust the mapping.
If an `UnknownTeam` fires, add the alias to `names.py`.

- [ ] **Step 3: Commit**

```bash
git add leagues/xg.py leagues/names.py
git commit -m "feat(leagues): Understat per-match team xG loader"
```

---

## Task 6: ClubElo prior loader

Used to give promoted / low-data teams a sane strength prior.

**Files:**
- Create: `leagues/elo.py`

- [ ] **Step 1: Implement elo.py**

```python
"""ClubElo ratings — the cross-league strength prior (free, no key, HTTP only)."""
import io
import urllib.request
from datetime import date

import pandas as pd

from leagues.names import UnknownTeam, canonical

API = "http://api.clubelo.com/{d}"   # NOTE: http only — clubelo does not serve https


def fetch_elo_snapshot(on: date | None = None) -> pd.DataFrame:
    """Every club's Elo on a date: columns Rank, Club, Country, Level, Elo, From, To."""
    d = (on or date.today()).isoformat()
    with urllib.request.urlopen(API.format(d=d), timeout=30) as resp:
        buf = io.StringIO(resp.read().decode("utf-8"))
    return pd.read_csv(buf)


def elo_for_league(league: str, teams: list[str], on: date | None = None) -> dict[str, float]:
    """Map our canonical team names -> ClubElo rating. Teams ClubElo doesn't know
    get the league's median (a safe neutral prior rather than a wrong number)."""
    snap = fetch_elo_snapshot(on)
    found: dict[str, float] = {}
    for _, row in snap.iterrows():
        try:
            name = canonical(str(row["Club"]), league)
        except UnknownTeam:
            continue
        if name in teams and name not in found:
            found[name] = float(row["Elo"])
    if found:
        median = pd.Series(list(found.values())).median()
    else:
        median = 1500.0
    return {t: found.get(t, float(median)) for t in teams}
```

- [ ] **Step 2: Smoke-test**

Run:
```bash
python -c "
from leagues import elo, fixtures
teams = sorted(set(fixtures.fetch_fixtures('PL')['home']))
r = elo.elo_for_league('PL', teams)
for t,v in sorted(r.items(), key=lambda kv:-kv[1])[:5]: print(t, round(v))
print('missing (got median):', sum(1 for v in r.values() if v == __import__('statistics').median(r.values())))
"
```
Expected: the strongest PL clubs (Liverpool/Arsenal/Man City) at the top with Elo ~1900-2000.

- [ ] **Step 3: Commit**

```bash
git add leagues/elo.py
git commit -m "feat(leagues): ClubElo prior loader"
```

---

## Task 7: Time-decay weights

**Files:**
- Create: `leagues/weights.py`
- Test: `tests/leagues/test_weights.py`

- [ ] **Step 1: Write the failing test**

`tests/leagues/test_weights.py`:

```python
import pandas as pd
from leagues.weights import decay_weights, HALF_LIFE_DAYS


def test_weight_is_one_at_reference_date():
    dates = pd.Series([pd.Timestamp("2026-07-01")])
    w = decay_weights(dates, ref=pd.Timestamp("2026-07-01"))
    assert abs(w.iloc[0] - 1.0) < 1e-9


def test_weight_halves_after_the_half_life():
    ref = pd.Timestamp("2026-07-01")
    old = ref - pd.Timedelta(days=HALF_LIFE_DAYS)
    w = decay_weights(pd.Series([old]), ref=ref)
    assert abs(w.iloc[0] - 0.5) < 1e-6


def test_older_matches_weigh_less():
    ref = pd.Timestamp("2026-07-01")
    dates = pd.Series([ref - pd.Timedelta(days=d) for d in (0, 200, 800)])
    w = decay_weights(dates, ref=ref)
    assert w.iloc[0] > w.iloc[1] > w.iloc[2] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_weights.py -v`
Expected: FAIL — no module `leagues.weights`

- [ ] **Step 3: Implement weights.py**

```python
"""Exponential time-decay: recent matches count more (Dixon-Coles xi).

CAUTION: the famous xi=0.0065 from the 1997 paper is per HALF-WEEK, not per day.
In days the sweet spot is ~0.0018-0.0033; we default to 0.003 (half-life ~231d)
and tune per league by walk-forward RPS in backtest.py.
"""
import numpy as np
import pandas as pd

XI_PER_DAY = 0.003
HALF_LIFE_DAYS = np.log(2) / XI_PER_DAY   # ~231 days


def decay_weights(dates: pd.Series, ref: pd.Timestamp, xi: float = XI_PER_DAY) -> pd.Series:
    """weight = exp(-xi * days_before_ref), clipped at 0 for future matches."""
    dates = pd.to_datetime(dates)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    if getattr(ref, "tzinfo", None) is not None:
        ref = ref.tz_localize(None)
    age_days = (ref - dates).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0)
    return np.exp(-xi * age_days)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_weights.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add leagues/weights.py tests/leagues/test_weights.py
git commit -m "feat(leagues): exponential time-decay weights"
```

---

## Task 8: Unified dataset assembly

Joins history + xG into the single table the model trains on.

**Files:**
- Create: `leagues/dataset.py`

- [ ] **Step 1: Implement dataset.py**

```python
"""Assemble the unified per-match training table: results + closing odds + xG."""
import pandas as pd

from leagues import history, xg


def build_matches(league: str) -> pd.DataFrame:
    """One row per match: date, home, away, goals, odds, and (where available)
    home_xg / away_xg. Matches with no xG keep NaN — the model falls back to goals."""
    hist = history.fetch_history(league)
    try:
        tx = xg.fetch_team_match_xg(league)
    except Exception as exc:  # xG is an enhancement, never a hard dependency
        print(f"WARNING: xG unavailable for {league} ({exc}); falling back to goals only")
        hist["home_xg"] = pd.NA
        hist["away_xg"] = pd.NA
        return hist

    tx = tx.copy()
    tx["day"] = pd.to_datetime(tx["date"]).dt.normalize()
    home_x = tx[tx["is_home"] == True][["day", "team", "xg"]].rename(  # noqa: E712
        columns={"team": "home", "xg": "home_xg"})
    away_x = tx[tx["is_home"] == False][["day", "team", "xg"]].rename(  # noqa: E712
        columns={"team": "away", "xg": "away_xg"})

    hist = hist.copy()
    hist["day"] = pd.to_datetime(hist["date"]).dt.normalize()
    # join on (day, team); allow +/-1 day slack for timezone-shifted listings
    out = hist.merge(home_x, on=["day", "home"], how="left")
    out = out.merge(away_x, on=["day", "away"], how="left")
    for shift in (1, -1):
        need = out["home_xg"].isna()
        if not need.any():
            break
        alt = hist.loc[need].copy()
        alt["day"] = alt["day"] + pd.Timedelta(days=shift)
        patch = (alt.merge(home_x, on=["day", "home"], how="left")
                    .merge(away_x, on=["day", "away"], how="left"))
        out.loc[need, "home_xg"] = patch["home_xg"].values
        out.loc[need, "away_xg"] = patch["away_xg"].values

    cov = out["home_xg"].notna().mean()
    print(f"{league}: {len(out)} matches, xG coverage {cov:.0%}")
    return out.drop(columns=["day"])
```

- [ ] **Step 2: Smoke-test all 4 leagues**

Run:
```bash
python -c "
from leagues import dataset, config
for lg in config.LEAGUES:
    df = dataset.build_matches(lg)
    print(lg, len(df), 'matches | xG cov', round(df['home_xg'].notna().mean()*100), '%')
"
```
Expected: xG coverage ≥ 90% for each league (Understat covers all 5 seasons of the big-5).
If coverage is low, the date-join slack needs widening — investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add leagues/dataset.py
git commit -m "feat(leagues): unified match dataset (results + odds + xG)"
```

---

## Task 9: The model — Dixon-Coles fit + xG-blended strengths + grid

**The key constraint:** `penaltyblog.models.DixonColesGoalModel` takes `Sequence[int]` goals — floats are rejected. So we fit it on **actual goals** (to get `rho`, home advantage, and time-weighted attack/defence), compute **xG-based strengths separately**, blend at the *strength* level, and build the scoreline grid ourselves with the Dixon-Coles tau correction.

**Files:**
- Create: `leagues/model.py`
- Test: `tests/leagues/test_model.py`

- [ ] **Step 1: Write the failing test**

`tests/leagues/test_model.py`:

```python
import numpy as np
import pandas as pd
import pytest

from leagues.model import LeagueModel, dc_tau, scoreline_grid


def test_dc_tau_lifts_draws_and_trims_1_0():
    # negative rho must raise 0-0 and 1-1, and reduce 1-0 / 0-1
    assert dc_tau(0, 0, 1.4, 1.2, rho=-0.1) > 1.0
    assert dc_tau(1, 1, 1.4, 1.2, rho=-0.1) > 1.0
    assert dc_tau(1, 0, 1.4, 1.2, rho=-0.1) < 1.0
    assert dc_tau(3, 2, 1.4, 1.2, rho=-0.1) == 1.0   # untouched


def test_scoreline_grid_is_a_normalized_distribution():
    g = scoreline_grid(1.5, 1.1, rho=-0.1, max_goals=10)
    assert abs(g.sum() - 1.0) < 1e-9
    assert (g >= 0).all()


def test_grid_gives_higher_home_win_prob_for_stronger_home_team():
    strong = scoreline_grid(2.2, 0.8, rho=-0.1)
    weak = scoreline_grid(0.8, 2.2, rho=-0.1)
    home_win_strong = np.tril(strong, -1).sum()
    home_win_weak = np.tril(weak, -1).sum()
    assert home_win_strong > 0.5 > home_win_weak


def _toy_matches(n=200):
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 1.8, "B": 1.4, "C": 1.0, "D": 0.7}
    rows = []
    start = pd.Timestamp("2025-08-01")
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        hg = rng.poisson(strength[h] * 1.15)
        ag = rng.poisson(strength[a] * 0.85)
        rows.append({"date": start + pd.Timedelta(days=i), "home": h, "away": a,
                     "home_goals": int(hg), "away_goals": int(ag),
                     "home_xg": float(hg), "away_xg": float(ag)})
    return pd.DataFrame(rows)


def test_model_fits_and_ranks_teams_correctly():
    m = LeagueModel().fit(_toy_matches(), ref=pd.Timestamp("2026-02-20"))
    probs = m.predict("A", "D")
    assert probs["p_home"] > probs["p_away"]          # strongest at home vs weakest
    assert abs(sum(probs[k] for k in ("p_home", "p_draw", "p_away")) - 1.0) < 1e-6
    assert 0 < probs["p_draw"] < 0.45


def test_predict_unknown_team_raises():
    m = LeagueModel().fit(_toy_matches(), ref=pd.Timestamp("2026-02-20"))
    with pytest.raises(KeyError):
        m.predict("A", "ZZ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_model.py -v`
Expected: FAIL — no module `leagues.model`

- [ ] **Step 3: Implement model.py**

```python
"""The match model.

penaltyblog's DixonColesGoalModel requires INTEGER goals, so xG cannot be the
response. Instead we:
  1. fit Dixon-Coles on actual goals  -> rho, home advantage, attack/defence
  2. compute xG-based attack/defence separately (ratio to league average)
  3. blend the two strength sets (default 75% xG / 25% goals)
  4. build the scoreline grid ourselves with the Dixon-Coles tau correction
This keeps penaltyblog's rigorous MLE while gaining xG's lower-variance signal.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import penaltyblog as pb
from scipy.stats import poisson

from leagues.weights import XI_PER_DAY, decay_weights

XG_WEIGHT = 0.75          # response blend: 0.75*xG + 0.25*actual goals
MAX_GOALS = 10
PRIOR_STRENGTH = 6.0      # pseudo-matches of prior for a brand-new team


def dc_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles low-score correction; negative rho lifts 0-0 and 1-1."""
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def scoreline_grid(lh: float, la: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Normalized (max_goals+1) x (max_goals+1) correct-score matrix [home, away]."""
    hp = poisson.pmf(np.arange(max_goals + 1), lh)
    ap = poisson.pmf(np.arange(max_goals + 1), la)
    grid = np.outer(hp, ap)
    for h in range(min(2, max_goals + 1)):
        for a in range(min(2, max_goals + 1)):
            grid[h, a] *= dc_tau(h, a, lh, la, rho)
    return grid / grid.sum()


def outcome_probs(grid: np.ndarray) -> tuple[float, float, float]:
    """(home, draw, away) from a scoreline grid."""
    home = float(np.tril(grid, -1).sum())
    draw = float(np.trace(grid))
    away = float(np.triu(grid, 1).sum())
    return home, draw, away


@dataclass
class LeagueModel:
    xi: float = XI_PER_DAY
    xg_weight: float = XG_WEIGHT
    rho: float = 0.0
    home_adv: float = 0.0
    attack: dict = field(default_factory=dict)     # team -> log attack strength
    defence: dict = field(default_factory=dict)    # team -> log defence strength
    league_mean_goals: float = 1.4

    # -- fitting -----------------------------------------------------------
    def fit(self, matches: pd.DataFrame, ref: pd.Timestamp | None = None,
            priors: dict[str, float] | None = None) -> "LeagueModel":
        """matches: date, home, away, home_goals, away_goals [, home_xg, away_xg].
        priors: optional team -> prior strength multiplier (from ClubElo), used to
        shrink teams with little history (promoted sides)."""
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        if df.empty:
            raise ValueError("no played matches to fit on")
        ref = ref or pd.to_datetime(df["date"]).max()
        w = decay_weights(df["date"], ref=ref, xi=self.xi).to_numpy()

        # 1. Dixon-Coles on ACTUAL goals (integers required by penaltyblog)
        clf = pb.models.DixonColesGoalModel(
            df["home_goals"].astype(int).to_numpy(),
            df["away_goals"].astype(int).to_numpy(),
            df["home"].to_numpy(), df["away"].to_numpy(),
            weights=w,
        )
        clf.fit()
        params = clf.get_params()
        self.rho = float(params.get("rho", 0.0))
        self.home_adv = float(params.get("home_advantage", 0.0))
        goal_att = {t: float(v) for k, v in params.items()
                    if k.startswith("attack_") for t in [k[len("attack_"):]]}
        goal_def = {t: float(v) for k, v in params.items()
                    if k.startswith("defence_") for t in [k[len("defence_"):]]}

        # 2. xG-based strengths (log ratio to league average), same time weights
        xg_att, xg_def = self._xg_strengths(df, w)

        # 3. blend at the strength level
        teams = sorted(set(goal_att) | set(goal_def))
        self.attack, self.defence = {}, {}
        for t in teams:
            ga, gd = goal_att.get(t, 0.0), goal_def.get(t, 0.0)
            xa, xd = xg_att.get(t), xg_def.get(t)
            if xa is None or xd is None:
                self.attack[t], self.defence[t] = ga, gd
            else:
                self.attack[t] = self.xg_weight * xa + (1 - self.xg_weight) * ga
                self.defence[t] = self.xg_weight * xd + (1 - self.xg_weight) * gd

        # 4. shrink low-data teams toward their prior (promoted sides)
        if priors:
            self._apply_priors(df, w, priors)

        self.league_mean_goals = float(
            (df["home_goals"].sum() + df["away_goals"].sum()) / (2 * len(df)))
        return self

    def _xg_strengths(self, df: pd.DataFrame, w: np.ndarray):
        """Weighted log(xG for / league avg) and log(xG against / league avg)."""
        if "home_xg" not in df or df["home_xg"].isna().all():
            return {}, {}
        d = df.dropna(subset=["home_xg", "away_xg"]).copy()
        if d.empty:
            return {}, {}
        wd = decay_weights(d["date"], ref=pd.to_datetime(df["date"]).max(), xi=self.xi).to_numpy()
        rows = pd.concat([
            pd.DataFrame({"team": d["home"], "xgf": d["home_xg"], "xga": d["away_xg"], "w": wd}),
            pd.DataFrame({"team": d["away"], "xgf": d["away_xg"], "xga": d["home_xg"], "w": wd}),
        ])
        avg_f = np.average(rows["xgf"], weights=rows["w"])
        att, dfn = {}, {}
        for team, g in rows.groupby("team"):
            f = np.average(g["xgf"], weights=g["w"])
            a = np.average(g["xga"], weights=g["w"])
            att[team] = float(np.log(max(f, 0.05) / avg_f))
            dfn[team] = float(np.log(max(a, 0.05) / avg_f))
        return att, dfn

    def _apply_priors(self, df: pd.DataFrame, w: np.ndarray, priors: dict[str, float]):
        """Shrink each team's strengths toward its prior with weight
        PRIOR_STRENGTH / (PRIOR_STRENGTH + effective_matches)."""
        eff = {}
        for team in self.attack:
            mask = ((df["home"] == team) | (df["away"] == team)).to_numpy()
            eff[team] = float(w[mask].sum())
        for team in list(self.attack):
            p = priors.get(team)
            if p is None:
                continue
            k = PRIOR_STRENGTH / (PRIOR_STRENGTH + eff.get(team, 0.0))
            self.attack[team] = (1 - k) * self.attack[team] + k * p
            self.defence[team] = (1 - k) * self.defence[team] - k * p

    # -- prediction --------------------------------------------------------
    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        for t in (home, away):
            if t not in self.attack:
                raise KeyError(f"team {t!r} not in fitted model")
        lh = np.exp(self.attack[home] + self.defence[away] + self.home_adv)
        la = np.exp(self.attack[away] + self.defence[home])
        return float(np.clip(lh, 0.05, 6.0)), float(np.clip(la, 0.05, 6.0))

    def predict(self, home: str, away: str) -> dict:
        lh, la = self.lambdas(home, away)
        grid = scoreline_grid(lh, la, self.rho)
        ph, pd_, pa = outcome_probs(grid)
        h, a = np.unravel_index(np.argmax(grid), grid.shape)
        return {"p_home": ph, "p_draw": pd_, "p_away": pa,
                "lambda_home": lh, "lambda_away": la,
                "score": f"{int(h)}-{int(a)}", "grid": grid}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_model.py -v`
Expected: 6 passed. If `get_params()` uses different key names than `attack_`/`defence_`/`home_advantage`/`rho`, print `clf.get_params()` once and adjust the three prefixes — do not guess.

- [ ] **Step 5: Commit**

```bash
git add leagues/model.py tests/leagues/test_model.py
git commit -m "feat(leagues): Dixon-Coles + xG-blended strengths + scoreline grid"
```

---

## Task 10: pi-ratings ensemble + isotonic calibration

**Files:**
- Modify: `leagues/model.py` (append)
- Test: `tests/leagues/test_model.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/leagues/test_model.py`)

```python
from leagues.model import Calibrator, blend_probs


def test_blend_probs_averages_and_normalizes():
    a = (0.5, 0.3, 0.2)
    b = (0.7, 0.2, 0.1)
    out = blend_probs(a, b, weight=0.5)
    assert abs(sum(out) - 1.0) < 1e-9
    assert abs(out[0] - 0.6) < 1e-9


def test_calibrator_is_identity_ish_on_perfect_probabilities():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([2, 1, 2], size=400)
    # outcomes drawn from p itself => already calibrated
    y = np.array([rng.choice(3, p=row) for row in p])
    cal = Calibrator().fit(p, y)
    out = cal.transform(p)
    assert out.shape == p.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    # calibration must not destroy discrimination
    assert np.corrcoef(out[:, 0], p[:, 0])[0, 1] > 0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/leagues/test_model.py -k "blend or calibrator" -v`
Expected: FAIL — cannot import `Calibrator` / `blend_probs`

- [ ] **Step 3: Append to `leagues/model.py`**

```python
from sklearn.isotonic import IsotonicRegression


def blend_probs(a: tuple[float, float, float], b: tuple[float, float, float],
                weight: float = 0.5) -> tuple[float, float, float]:
    """Convex blend of two 1X2 forecasts, renormalized."""
    out = np.array(a) * weight + np.array(b) * (1 - weight)
    out = np.clip(out, 1e-9, None)
    out = out / out.sum()
    return float(out[0]), float(out[1]), float(out[2])


class Calibrator:
    """One-vs-rest isotonic recalibration of 1X2 probabilities, renormalized.

    Fit on a HELD-OUT period only — never on the training matches.
    """

    def __init__(self):
        self.iso: list[IsotonicRegression] = []

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "Calibrator":
        """probs: (n,3) [home, draw, away]; outcomes: (n,) ints 0/1/2."""
        self.iso = []
        for k in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(probs[:, k], (outcomes == k).astype(float))
            self.iso.append(ir)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.iso:
            return probs
        out = np.column_stack([self.iso[k].predict(probs[:, k]) for k in range(3)])
        out = np.clip(out, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)


def pi_rating_probs(matches: pd.DataFrame, home: str, away: str) -> tuple[float, float, float]:
    """1X2 from penaltyblog's pi-ratings, walked over the match history."""
    pi = pb.ratings.PiRatingSystem()
    for _, r in matches.iterrows():
        pi.update_ratings(r["home"], r["away"], int(r["home_goals"]), int(r["away_goals"]))
    p = pi.calculate_match_probabilities(home, away)
    if isinstance(p, dict):
        vals = (float(p.get("home_win", p.get("home", 0.0))),
                float(p.get("draw", 0.0)),
                float(p.get("away_win", p.get("away", 0.0))))
    else:
        vals = tuple(float(x) for x in p)
    total = sum(vals) or 1.0
    return (vals[0] / total, vals[1] / total, vals[2] / total)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/leagues/test_model.py -v`
Expected: 8 passed. If `PiRatingSystem.calculate_match_probabilities` returns a different shape, print it once and fix `pi_rating_probs` accordingly.

- [ ] **Step 5: Commit**

```bash
git add leagues/model.py tests/leagues/test_model.py
git commit -m "feat(leagues): pi-ratings ensemble + isotonic calibration"
```

---

## Task 11: Walk-forward backtest vs de-vigged bookmaker odds

This is the credibility gate. Train on everything before a matchweek, predict it, roll forward.

**Files:**
- Create: `leagues/backtest.py`
- Test: `tests/leagues/test_backtest.py`

- [ ] **Step 1: Write the failing test**

`tests/leagues/test_backtest.py`:

```python
import numpy as np
from leagues.backtest import devig, rps, accuracy, outcome_index


def test_outcome_index_maps_results():
    assert outcome_index(2, 1) == 0   # home win
    assert outcome_index(1, 1) == 1   # draw
    assert outcome_index(0, 3) == 2   # away win


def test_devig_removes_overround_and_sums_to_one():
    p = devig(2.0, 3.5, 4.0)
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[0] > p[1] and p[0] > p[2]


def test_rps_is_zero_for_perfect_forecast():
    assert rps(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == 0.0


def test_rps_penalizes_distance_ordering():
    # predicting away when home wins is worse than predicting draw when home wins
    far = rps(np.array([[0.0, 0.0, 1.0]]), np.array([0]))
    near = rps(np.array([[0.0, 1.0, 0.0]]), np.array([0]))
    assert far > near


def test_accuracy_counts_argmax_hits():
    p = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    assert accuracy(p, np.array([0, 2])) == 1.0
    assert accuracy(p, np.array([1, 2])) == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/leagues/test_backtest.py -v`
Expected: FAIL — no module `leagues.backtest`

- [ ] **Step 3: Implement backtest.py**

```python
"""Walk-forward validation against de-vigged bookmaker closing odds.

The bookmaker closing line is the benchmark: getting close to it is success;
beating it consistently on accuracy is not realistic for a free-data model.
"""
import numpy as np
import pandas as pd

from leagues.model import Calibrator, LeagueModel


def outcome_index(hg: int, ag: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if hg > ag:
        return 0
    if hg == ag:
        return 1
    return 2


def devig(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
    """Proportional de-vig: raw implied probs normalized to sum to 1."""
    raw = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a])
    return tuple(raw / raw.sum())


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score for ordered 3-outcome forecasts (lower = better)."""
    n = len(outcomes)
    obs = np.zeros_like(probs)
    obs[np.arange(n), outcomes] = 1.0
    cp, co = np.cumsum(probs, axis=1), np.cumsum(obs, axis=1)
    return float(((cp - co) ** 2)[:, :2].sum() / (2 * n) * 2 / 1)  # mean over (k-1)=2 terms


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    n = len(outcomes)
    obs = np.zeros_like(probs)
    obs[np.arange(n), outcomes] = 1.0
    return float(((probs - obs) ** 2).sum() / n)


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == outcomes).mean())


def walk_forward(matches: pd.DataFrame, xi: float = 0.003, xg_weight: float = 0.75,
                 min_train: int = 380, step_days: int = 7,
                 priors: dict[str, float] | None = None) -> pd.DataFrame:
    """Roll through the last part of the data, refitting weekly.

    Returns one row per predicted match with model probs, market probs and outcome.
    """
    df = matches.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
    df = df.reset_index(drop=True)
    if len(df) <= min_train:
        raise ValueError(f"need > {min_train} matches to backtest, got {len(df)}")

    rows = []
    start_date = df.loc[min_train, "date"]
    cutoffs = pd.date_range(start_date, df["date"].max(), freq=f"{step_days}D")
    for cutoff in cutoffs:
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if test.empty or train.empty:
            continue
        try:
            model = LeagueModel(xi=xi, xg_weight=xg_weight).fit(train, ref=cutoff, priors=priors)
        except Exception as exc:
            print(f"  skip {cutoff.date()}: fit failed ({exc})")
            continue
        for _, m in test.iterrows():
            try:
                p = model.predict(m["home"], m["away"])
            except KeyError:
                continue   # a team with no history yet (promoted); skipped, not guessed
            row = {"date": m["date"], "home": m["home"], "away": m["away"],
                   "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
                   "outcome": outcome_index(int(m["home_goals"]), int(m["away_goals"]))}
            if pd.notna(m.get("odds_h")):
                mh, md, ma = devig(m["odds_h"], m["odds_d"], m["odds_a"])
                row |= {"m_home": mh, "m_draw": md, "m_away": ma}
            rows.append(row)
    return pd.DataFrame(rows)


def score(results: pd.DataFrame) -> dict:
    """Model vs market on the same matches."""
    p = results[["p_home", "p_draw", "p_away"]].to_numpy()
    y = results["outcome"].to_numpy()
    out = {"n": len(results), "accuracy": accuracy(p, y),
           "rps": rps(p, y), "brier": brier(p, y)}
    mk = results.dropna(subset=["m_home"]) if "m_home" in results else results.iloc[0:0]
    if len(mk):
        mp = mk[["m_home", "m_draw", "m_away"]].to_numpy()
        my = mk["outcome"].to_numpy()
        out |= {"market_n": len(mk), "market_accuracy": accuracy(mp, my),
                "market_rps": rps(mp, my), "market_brier": brier(mp, my)}
    return out
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/leagues/test_backtest.py -v`
Expected: 5 passed. If `test_rps_is_zero_for_perfect_forecast` fails, the RPS normalization is wrong — RPS for a 3-outcome forecast is `sum_{k=1}^{2} (cumP_k - cumO_k)^2 / 2`, averaged over matches. Fix `rps()` to exactly that.

- [ ] **Step 5: Commit**

```bash
git add leagues/backtest.py tests/leagues/test_backtest.py
git commit -m "feat(leagues): walk-forward backtest vs de-vigged closing odds"
```

---

## Task 12: THE GATE — run the backtest, tune, and report

No UI work begins until this passes.

**Files:**
- Create: `leagues/tune.py`

- [ ] **Step 1: Implement the tuning/report script**

`leagues/tune.py`:

```python
"""Grid-search xi and xg_weight per league by walk-forward RPS, then report
model vs de-vigged market. This is the credibility gate."""
import itertools
import json
import sys

from leagues import backtest, config, dataset

XIS = [0.0018, 0.0025, 0.003, 0.0035]
XGWS = [0.0, 0.5, 0.75, 0.9]


def tune(league: str) -> dict:
    matches = dataset.build_matches(league)
    best, best_rps = None, float("inf")
    for xi, xgw in itertools.product(XIS, XGWS):
        res = backtest.walk_forward(matches, xi=xi, xg_weight=xgw)
        if res.empty:
            continue
        s = backtest.score(res)
        print(f"  xi={xi:<7} xg_w={xgw:<5} n={s['n']:<5} "
              f"acc={s['accuracy']:.3f} rps={s['rps']:.4f}")
        if s["rps"] < best_rps:
            best, best_rps = {"xi": xi, "xg_weight": xgw, **s}, s["rps"]
    return best


def main():
    leagues = sys.argv[1:] or list(config.LEAGUES)
    report = {}
    for lg in leagues:
        print(f"\n=== {config.get(lg).name} ===")
        report[lg] = tune(lg)
        b = report[lg]
        print(f"  BEST xi={b['xi']} xg_weight={b['xg_weight']}")
        print(f"  MODEL : acc {b['accuracy']:.1%}  RPS {b['rps']:.4f}  Brier {b['brier']:.4f}")
        if "market_rps" in b:
            print(f"  MARKET: acc {b['market_accuracy']:.1%}  RPS {b['market_rps']:.4f}  "
                  f"Brier {b['market_brier']:.4f}")
            print(f"  GAP   : RPS {b['rps'] - b['market_rps']:+.4f} (negative = we beat market)")
    with open("data-raw/leagues/backtest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nWrote data-raw/leagues/backtest_report.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

Run:
```bash
mkdir -p data-raw/leagues
python -m leagues.tune PL
```
Expected (from the research, these are the pass criteria):
- **accuracy ≈ 50–55%**
- **RPS ≈ 0.19–0.21**
- **RPS gap vs market ≈ +0.005 to +0.02** (we are slightly worse than the closing line — that is success; a large negative gap means a bug, likely lookahead leakage)

**Sanity checks — if any of these trip, STOP and debug rather than proceeding:**
- accuracy > 60% or RPS < 0.17 → almost certainly **lookahead leakage** (training data includes the test match). Verify `train = df[df["date"] < cutoff]`.
- accuracy < 45% or RPS > 0.23 → the model is broken; check that `get_params()` prefixes were parsed correctly and that team names joined (xG coverage, `UnknownTeam`).

- [ ] **Step 3: Run all four leagues**

Run: `python -m leagues.tune`
Expected: a table per league and `data-raw/leagues/backtest_report.json` written.

- [ ] **Step 4: Commit the results**

```bash
git add leagues/tune.py data-raw/leagues/backtest_report.json
git commit -m "feat(leagues): tuning + backtest gate report (model vs market)"
```

- [ ] **Step 5: STOP — report to the user**

Present the per-league table: accuracy, RPS, Brier, and the gap vs the de-vigged market. State plainly whether each league passes the gate. **Do not start Phase 2 (player props, season sim, UI) until the user has seen these numbers and approved.** If a league fails, diagnose (leakage, name joins, xG coverage, prior handling) before building anything on top.

---

## Self-Review Notes

- **Spec coverage:** data sources (Tasks 3–6), name normalization (2), unified dataset (8), time-decay (7), Dixon-Coles + xG blend (9), ClubElo priors/shrinkage (9 `_apply_priors` + 6), pi-ratings ensemble + isotonic calibration (10), walk-forward backtest vs de-vigged odds (11), tuning + gate (12). **Deferred to Phase 2 by design:** player props (scorers + shots), season Monte Carlo, UI/competition switcher, pick tracking, ops — these are gated on Task 12.
- **Known API risks flagged in-plan:** `penaltyblog.get_params()` key prefixes (Task 9 Step 4) and `PiRatingSystem.calculate_match_probabilities` return shape (Task 10 Step 4) are verified at runtime with explicit "print it and fix, don't guess" instructions.
- **Integer-goals constraint** on `DixonColesGoalModel` is the central design driver and is handled explicitly in Task 9.
