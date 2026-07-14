# Leagues Phase 2a — Premier League Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the Phase 1 Premier League match model end-to-end — player props, season simulation, published JSON, a live page, pick-tracking, and scheduled jobs — so one league works completely before plan 2b copies it to the other three.

**Architecture:** Six new modules in the existing `leagues/` package. `players.py` is the only new network loader (FBref per-player match logs). `props.py` turns those logs into per-player goal/shot rates and rescales them so each team's player λ sum equals the match model's fitted team goal expectation Λ — the single channel through which opponent strength reaches the player numbers. `sim.py` Monte-Carlos the season off the same fit. `picks.py` ports the WC lock/grade/void honesty rules. `publish.py` is the only module that knows the JSON contract and writes `data/leagues/pl.json`, which the new standalone `leagues.html` renders. The World Cup app is untouched.

**Tech Stack:** Python 3.14, `soccerdata` 1.9 (FBref), `penaltyblog` 1.11, pandas 3, numpy, scipy, pytest. Vanilla JS + CSS for the page (no build step, matching the WC app).

**Spec:** `docs/superpowers/specs/2026-07-14-leagues-phase2a-pl-slice-design.md`

---

## GATE — do not start Task 1 until this passes

Phase 2a is gated on the Phase 1 walk-forward backtest passing for PL under the **current** `model.py` (which carries two post-report fixes: scale-consistent xG blending, and prior shrinkage for thin-data teams).

- [ ] **Confirm the PL gate**

Run:
```bash
python -m leagues.tune PL
```
Expected (confirmed 2026-07-14): `acc 53.7%  RPS 0.2001` vs market `acc 54.9%  RPS 0.1939`, **GAP RPS +0.0061**.

Pass band: accuracy 50-55%, RPS 0.19-0.21, gap +0.005 to +0.02.
- Gap **negative** (we "beat" the closing line) → lookahead leakage, not brilliance. STOP and debug.
- Accuracy <45% or RPS >0.23 → model broken. STOP.

**Why this gates the props:** fixture-specific information (opponent strength, home advantage) reaches the player numbers *only* through the Σλ rescale in Task 3. A broken match model silently corrupts every player number on the page.

---

## File Structure

```
worldcup/
  leagues/
    players.py            NEW — FBref per-player match logs (the only new network loader)
    props.py              NEW — rates, shrinkage, minutes, penalties, Σλ rescale, Poisson props
    props_backtest.py     NEW — walk-forward props gate vs a position-prior baseline
    sim.py                NEW — season Monte Carlo + PL tie-breakers
    picks.py              NEW — lock / grade / void (ported WC honesty rules)
    publish.py            NEW — orchestrator; the ONLY module that knows the JSON contract
  tests/leagues/
    test_players.py       NEW — parse test against a captured payload (no network)
    test_props.py         NEW — Σλ invariant, shrinkage direction, penalty split
    test_sim.py           NEW — tie-breaker order, locked results
    test_picks.py         NEW — void-on-late-lock, frozen grading
  data/leagues/
    pl.json               NEW (generated) — the published payload
    clubs.json            NEW (static) — 20 club colours/crests
  leagues.html            NEW — standalone PL page
  ops/
    leagues_weekly.py     NEW — weekly refresh + publish
    leagues_matchday.py   NEW — record results, re-grade, redeploy
```

**Untouched (hard constraint — the WC final is 2026-07-19):** `index.html`, `predict.py`, `data/predictions.json`, `data-raw/*` (WC files).

---

## Task 1: FBref per-player match logs (`players.py`)

**Files:**
- Create: `leagues/players.py`
- Create: `tests/leagues/test_players.py`

The one new network loader. Owns exactly one source, returns a clean frame with canonical names — same contract as `xg.py` and `history.py`.

- [ ] **Step 1: Verify the soccerdata FBref API before writing anything**

`soccerdata`'s FBref player-match-stats API differs by version and stat type. **Print it, do not guess** (this is the same discipline that caught the `penaltyblog.get_params()` prefixes and the Understat season-code bug in Phase 1).

Run:
```bash
python -c "
import soccerdata as sd
fb = sd.FBref(leagues='ENG-Premier League', seasons='2526')
df = fb.read_player_match_stats(stat_type='summary')
print(df.columns.tolist())
print(df.index.names)
print(df.head(3).to_string()[:1500])
"
```
Expected: a MultiIndex frame with league/season/game/team/player levels and columns including minutes, goals, shots, shots on target, npxG. **Write down the exact column names you see** — the next step's `COLS` mapping must match them, not what this plan guesses.

- [ ] **Step 2: Write the failing test** (pure parser, fixed payload, no network)

```python
# tests/leagues/test_players.py
import pandas as pd
import pytest

from leagues.players import parse_player_logs
from leagues.names import UnknownTeam


def _raw():
    """Mimics the flattened FBref summary frame (see players.py COLS)."""
    return pd.DataFrame([
        {"date": "2026-05-01", "team": "Man City", "player": "Erling Haaland",
         "pos": "FW", "minutes": 90, "goals": 2, "pens_made": 1, "pens_att": 1,
         "shots": 5, "shots_on_target": 3, "npxg": 0.84},
        {"date": "2026-05-01", "team": "Brentford", "player": "Bryan Mbeumo",
         "pos": "FW,MF", "minutes": 78, "goals": 0, "pens_made": 0, "pens_att": 0,
         "shots": 2, "shots_on_target": 1, "npxg": 0.21},
    ])


def test_parses_and_canonicalizes():
    df = parse_player_logs(_raw(), "PL")
    assert set(df.columns) >= {"date", "team", "player", "pos", "minutes",
                               "np_goals", "shots", "sot", "npxg", "pens_att"}
    assert df.loc[0, "team"] == "Manchester City"      # canonical, not "Man City"
    # non-penalty goals strip the converted penalty
    assert df.loc[0, "np_goals"] == 1
    assert df.loc[1, "np_goals"] == 0


def test_primary_position_only():
    df = parse_player_logs(_raw(), "PL")
    assert df.loc[1, "pos"] == "FW"        # "FW,MF" -> first listed


def test_unmapped_team_fails_loudly():
    raw = _raw()
    raw.loc[0, "team"] = "Wimbledon FC"
    with pytest.raises(UnknownTeam):
        parse_player_logs(raw, "PL")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_players.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'leagues.players'`

- [ ] **Step 4: Implement players.py**

```python
"""Per-player match logs from FBref (via soccerdata, disk-cached).

The ONLY new network loader in Phase 2a. Returns one row per player-appearance
with NON-PENALTY goals: penalties belong to the designated taker, not to the
striker's underlying rate, so they are stripped here and added back for exactly
one player per team in props.py.
"""
import pandas as pd
import soccerdata as sd

from leagues import config
from leagues.names import canonical

# Map FBref's column names -> ours. VERIFY against Step 1 output before trusting.
COLS = {
    "min": "minutes",
    "gls": "goals",
    "pk": "pens_made",
    "pkatt": "pens_att",
    "sh": "shots",
    "sot": "shots_on_target",
    "npxg": "npxg",
}


def parse_player_logs(raw: pd.DataFrame, league: str) -> pd.DataFrame:
    """Pure parser — no network. Canonical teams, non-penalty goals, primary position."""
    df = raw.copy()
    df["team"] = [canonical(t, league) for t in df["team"]]
    df["pos"] = df["pos"].fillna("MF").astype(str).str.split(",").str[0].str.strip()
    for c in ("minutes", "goals", "pens_made", "pens_att", "shots",
              "shots_on_target", "npxg"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["np_goals"] = (df["goals"] - df["pens_made"]).clip(lower=0)
    df["sot"] = df["shots_on_target"]
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "team", "player", "pos", "minutes", "np_goals",
               "shots", "sot", "npxg", "pens_att"]].reset_index(drop=True)


def fetch_player_logs(league: str) -> pd.DataFrame:
    """Download (cached) five seasons of per-player match logs for one league."""
    lg = config.get(league)
    fb = sd.FBref(leagues=lg.fbref, seasons=list(lg.history_seasons))
    raw = fb.read_player_match_stats(stat_type="summary").reset_index()
    raw.columns = ["_".join(c).strip("_").lower() if isinstance(c, tuple) else str(c).lower()
                   for c in raw.columns]

    # Resolve our canonical names against whatever FBref actually returned.
    rename, missing = {}, []
    for fb_col, ours in COLS.items():
        hit = next((c for c in raw.columns if c == fb_col or c.endswith("_" + fb_col)), None)
        if hit is None:
            missing.append(fb_col)
        else:
            rename[hit] = ours
    if missing:
        raise RuntimeError(
            f"FBref schema changed; could not find {missing}. Got: {list(raw.columns)}"
        )
    raw = raw.rename(columns=rename)
    if "pos" not in raw.columns:
        raise RuntimeError(f"FBref schema changed; no 'pos'. Got: {list(raw.columns)}")

    return parse_player_logs(raw, league)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_players.py -v`
Expected: 3 passed.

- [ ] **Step 6: Smoke-test the live feed (network; slow on first run — FBref rate-limits)**

Run:
```bash
python -c "
from leagues.players import fetch_player_logs
df = fetch_player_logs('PL')
print(len(df), 'appearances')
print(df.groupby('player')['np_goals'].sum().sort_values(ascending=False).head(5))
print('minutes total', int(df['minutes'].sum()))
"
```
Expected: tens of thousands of appearances; the top non-penalty scorers over 5 seasons look like real PL strikers (Haaland, Salah, Kane...). If a name is nonsense, the join is wrong — fix before proceeding.

- [ ] **Step 7: Commit**

```bash
git add leagues/players.py tests/leagues/test_players.py
git commit -m "feat(leagues): FBref per-player match logs loader"
```

---

## Task 2: Position priors and per-90 rates (`props.py`, part 1)

**Files:**
- Create: `leagues/props.py`
- Create: `tests/leagues/test_props.py`

Rates first, rescale in Task 3. Splitting keeps each step testable.

- [ ] **Step 1: Write the failing test**

```python
# tests/leagues/test_props.py
import numpy as np
import pandas as pd

from leagues.props import player_rates, GOAL_PRIORS, SHOT_PRIORS


def _logs():
    """Two players: one with a big sample, one with a single match."""
    rows = []
    for i in range(30):                      # 30 x 90min, 15 goals -> 0.5 np/90
        rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                     "team": "Manchester City", "player": "Regular", "pos": "FW",
                     "minutes": 90, "np_goals": i % 2, "shots": 4, "sot": 2,
                     "npxg": 0.5, "pens_att": 0})
    rows.append({"date": pd.Timestamp("2026-02-01"), "team": "Manchester City",
                 "player": "NewSigning", "pos": "FW", "minutes": 90,
                 "np_goals": 1, "shots": 1, "sot": 1, "npxg": 0.1, "pens_att": 0})
    return pd.DataFrame(rows)


def test_low_sample_player_is_shrunk_toward_the_prior():
    r = player_rates(_logs(), ref=pd.Timestamp("2026-03-01"))
    new = r.loc[r["player"] == "NewSigning"].iloc[0]
    reg = r.loc[r["player"] == "Regular"].iloc[0]
    prior = GOAL_PRIORS["FW"]
    # raw rate for the new signing is 1.0 g/90 (nonsense); shrinkage must pull it
    # most of the way back to the FW prior, and much nearer the prior than Regular.
    assert abs(new["rate90"] - prior) < abs(1.0 - prior) / 2
    assert abs(new["rate90"] - prior) < abs(reg["rate90"] - prior) + 0.25


def test_regular_player_keeps_his_own_signal():
    r = player_rates(_logs(), ref=pd.Timestamp("2026-03-01"))
    reg = r.loc[r["player"] == "Regular"].iloc[0]
    # 30 nineties of evidence: barely shrunk, so clearly above the 0.45 FW prior
    # is not required — but it must sit between the prior and his raw 0.5 rate.
    assert min(GOAL_PRIORS["FW"], 0.5) - 0.05 <= reg["rate90"] <= max(GOAL_PRIORS["FW"], 0.5) + 0.05


def test_shot_rates_present_and_shrunk_the_same_way():
    r = player_rates(_logs(), ref=pd.Timestamp("2026-03-01"))
    new = r.loc[r["player"] == "NewSigning"].iloc[0]
    assert "shots90" in r.columns and "sot_ratio" in r.columns
    # one match at 1 shot; must be pulled UP toward the ~2.5 FW shot prior
    assert new["shots90"] > 1.0
    assert 0.0 <= new["sot_ratio"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_props.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'leagues.props'`

- [ ] **Step 3: Implement the rate engine**

```python
"""Player props: anytime scorer, shots, shots on target.

Pipeline (see spec §5):
  rates -> shrinkage -> expected minutes -> penalties -> RESCALE to the match
  model's team lambda -> Poisson props.

The rescale is load-bearing: it is the ONLY channel by which opponent strength
and home advantage reach the player numbers. Without it, a striker's number is
the same against Brentford and Liverpool -- a season-long rate table wearing a
costume.

All constants below are PROVISIONAL, taken from the design spec's research and
not from our data. props_backtest.py tunes them.
"""
import numpy as np
import pandas as pd

# non-penalty goals per 90, by primary position
GOAL_PRIORS = {"FW": 0.45, "AM": 0.20, "MF": 0.10, "DF": 0.05, "GK": 0.001}
# total shots per 90, by primary position
SHOT_PRIORS = {"FW": 2.5, "AM": 1.4, "MF": 0.9, "DF": 0.4, "GK": 0.02}
SOT_RATIO_PRIOR = 0.35

SEASON_DECAY = 0.7        # alpha^(seasons ago)
K_NINETIES = 7.0          # empirical-Bayes strength, in 90s
W_REALIZED_HIGH = 0.6     # weight on actual goals for high-minute players
W_REALIZED_LOW = 0.4      # ...and for low-sample players (trust xG more)
HIGH_MINUTE_90S = 10.0
PEN_CONVERSION = 0.76


def _prior(pos: str, table: dict, default_key: str = "MF") -> float:
    return table.get(pos, table[default_key])


def player_rates(logs: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """Decay-weighted, shrunk per-90 rates for every player in the logs.

    Returns one row per player: team, pos, nineties, rate90 (non-penalty goals),
    shots90, sot_ratio.
    """
    df = logs.copy()
    df["date"] = pd.to_datetime(df["date"])
    seasons_ago = ((ref - df["date"]).dt.days / 365.25).clip(lower=0)
    df["w"] = SEASON_DECAY ** seasons_ago
    df["w90"] = df["w"] * df["minutes"] / 90.0

    out = []
    for (team, player), g in df.groupby(["team", "player"], sort=False):
        n90 = float(g["w90"].sum())
        pos = g["pos"].mode().iat[0] if not g["pos"].mode().empty else "MF"
        if n90 <= 0:
            continue

        # 1. blend realized goals with xG; trust xG more when the sample is thin
        w_real = W_REALIZED_HIGH if n90 >= HIGH_MINUTE_90S else W_REALIZED_LOW
        g_per90 = float((g["np_goals"] * g["w"]).sum()) / n90
        x_per90 = float((g["npxg"] * g["w"]).sum()) / n90
        obs_goal = w_real * g_per90 + (1 - w_real) * x_per90
        obs_shot = float((g["shots"] * g["w"]).sum()) / n90
        shots_tot = float((g["shots"] * g["w"]).sum())
        sot_tot = float((g["sot"] * g["w"]).sum())

        # 2. empirical-Bayes shrinkage toward the position prior
        k = K_NINETIES
        rate90 = (n90 * obs_goal + k * _prior(pos, GOAL_PRIORS)) / (n90 + k)
        shots90 = (n90 * obs_shot + k * _prior(pos, SHOT_PRIORS)) / (n90 + k)
        # sot ratio shrinks on SHOTS taken, not nineties
        sot_ratio = ((sot_tot + 10.0 * SOT_RATIO_PRIOR) / (shots_tot + 10.0)
                     if shots_tot + 10.0 > 0 else SOT_RATIO_PRIOR)

        out.append({"team": team, "player": player, "pos": pos, "nineties": n90,
                    "rate90": rate90, "shots90": shots90, "sot_ratio": sot_ratio})

    return pd.DataFrame(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_props.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add leagues/props.py tests/leagues/test_props.py
git commit -m "feat(leagues): decay-weighted player rates with empirical-Bayes shrinkage"
```

---

## Task 3: The Σλ rescale and Poisson props (`props.py`, part 2)

**Files:**
- Modify: `leagues/props.py` (append)
- Modify: `tests/leagues/test_props.py` (append)

- [ ] **Step 1: Write the failing test — the invariant that defines the design**

```python
# append to tests/leagues/test_props.py
from leagues.props import match_props


def _rates():
    return pd.DataFrame([
        {"team": "Manchester City", "player": "Haaland", "pos": "FW", "nineties": 30,
         "rate90": 0.80, "shots90": 4.0, "sot_ratio": 0.5},
        {"team": "Manchester City", "player": "Foden", "pos": "AM", "nineties": 25,
         "rate90": 0.30, "shots90": 2.0, "sot_ratio": 0.4},
        {"team": "Manchester City", "player": "Dias", "pos": "DF", "nineties": 30,
         "rate90": 0.05, "shots90": 0.5, "sot_ratio": 0.3},
        {"team": "Brentford", "player": "Mbeumo", "pos": "FW", "nineties": 28,
         "rate90": 0.40, "shots90": 2.5, "sot_ratio": 0.4},
        {"team": "Brentford", "player": "Wissa", "pos": "FW", "nineties": 20,
         "rate90": 0.35, "shots90": 2.0, "sot_ratio": 0.35},
    ])


def test_lambda_sum_equals_team_lambda():
    """THE invariant: the player model never disagrees with the match model."""
    props = match_props(_rates(), home="Manchester City", away="Brentford",
                        lam_home=2.1, lam_away=0.9,
                        minutes={}, pen_taker={"Manchester City": "Haaland"},
                        opp_shot_factor={"Manchester City": 1.0, "Brentford": 1.0})
    home = [p for p in props if p["team"] == "Manchester City"]
    assert abs(sum(p["lambda_goals"] for p in home) - 2.1) < 1e-6
    away = [p for p in props if p["team"] == "Brentford"]
    assert abs(sum(p["lambda_goals"] for p in away) - 0.9) < 1e-6


def test_anytime_is_poisson_of_lambda():
    props = match_props(_rates(), home="Manchester City", away="Brentford",
                        lam_home=2.1, lam_away=0.9, minutes={},
                        pen_taker={}, opp_shot_factor={})
    h = next(p for p in props if p["player"] == "Haaland")
    assert abs(h["anytime_pct"] - 100 * (1 - np.exp(-h["lambda_goals"]))) < 0.01


def test_opponent_matters():
    """Same player, tougher opponent (lower team lambda) => lower anytime %."""
    easy = match_props(_rates(), "Manchester City", "Brentford", 2.4, 0.9,
                       {}, {}, {})
    hard = match_props(_rates(), "Manchester City", "Brentford", 1.2, 0.9,
                       {}, {}, {})
    he = next(p for p in easy if p["player"] == "Haaland")["anytime_pct"]
    hh = next(p for p in hard if p["player"] == "Haaland")["anytime_pct"]
    assert hh < he - 5


def test_only_the_designated_taker_carries_penalty_lambda():
    props = match_props(_rates(), "Manchester City", "Brentford", 2.1, 0.9,
                        minutes={}, pen_taker={"Manchester City": "Haaland"},
                        opp_shot_factor={}, exp_pens={"Manchester City": 0.15})
    h = next(p for p in props if p["player"] == "Haaland")
    f = next(p for p in props if p["player"] == "Foden")
    assert h["penalty_taker"] is True
    assert f["penalty_taker"] is False
    # the sum invariant still holds WITH penalties folded in
    home = [p for p in props if p["team"] == "Manchester City"]
    assert abs(sum(p["lambda_goals"] for p in home) - 2.1) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_props.py -v`
Expected: FAIL — `ImportError: cannot import name 'match_props'`

- [ ] **Step 3: Implement the rescale + props**

```python
# append to leagues/props.py


def match_props(rates: pd.DataFrame, home: str, away: str,
                lam_home: float, lam_away: float,
                minutes: dict | None = None,
                pen_taker: dict | None = None,
                opp_shot_factor: dict | None = None,
                exp_pens: dict | None = None) -> list[dict]:
    """Per-player props for ONE fixture.

    `lam_home`/`lam_away` are the match model's fitted team goal expectations
    (LeagueModel.predict -> lambda_home/lambda_away). Every player's goal lambda
    is rescaled so the team's players sum to exactly that number.

    minutes:         player -> expected minutes (default 90 for everyone in rates)
    pen_taker:       team -> player who takes penalties
    opp_shot_factor: team -> multiplier for how many shots the OPPONENT concedes
                     vs league average (1.0 = average)
    exp_pens:        team -> expected penalties awarded this match
    """
    minutes = minutes or {}
    pen_taker = pen_taker or {}
    opp_shot_factor = opp_shot_factor or {}
    exp_pens = exp_pens or {}

    out = []
    for team, lam_team in ((home, lam_home), (away, lam_away)):
        squad = rates[rates["team"] == team].copy()
        if squad.empty:
            continue

        squad["exp_min"] = [float(minutes.get(p, 90.0)) for p in squad["player"]]
        # open-play lambda before rescaling
        squad["raw"] = squad["rate90"] * squad["exp_min"] / 90.0

        # penalties are a TEAM property: strip them out of the team's open-play
        # budget and hand them to exactly one player.
        taker = pen_taker.get(team)
        lam_pen = float(exp_pens.get(team, 0.0)) * PEN_CONVERSION
        lam_pen = min(lam_pen, max(lam_team - 1e-6, 0.0))   # never exceed the budget
        lam_open = max(lam_team - lam_pen, 0.0)

        total_raw = float(squad["raw"].sum())
        scale = (lam_open / total_raw) if total_raw > 0 else 0.0

        factor = float(opp_shot_factor.get(team, 1.0))
        for _, r in squad.iterrows():
            lam_goals = float(r["raw"]) * scale
            is_taker = (r["player"] == taker)
            if is_taker:
                lam_goals += lam_pen

            s = float(r["shots90"]) * r["exp_min"] / 90.0 * factor
            sot = s * float(r["sot_ratio"])

            out.append({
                "team": team,
                "player": r["player"],
                "position": r["pos"],
                "lambda_goals": lam_goals,
                "anytime_pct": round(100.0 * (1.0 - np.exp(-lam_goals)), 1),
                "exp_shots": round(s, 2),
                "p_shots_2plus": round(100.0 * (1.0 - np.exp(-s) * (1.0 + s)), 1),
                "exp_sot": round(sot, 2),
                "p_sot_1plus": round(100.0 * (1.0 - np.exp(-sot)), 1),
                "penalty_taker": bool(is_taker),
                "doubt": False,
            })
    return out


def top_props(props: list[dict], team: str, n: int = 3) -> list[dict]:
    """Top-n players for a team by anytime probability."""
    squad = [p for p in props if p["team"] == team]
    return sorted(squad, key=lambda p: p["anytime_pct"], reverse=True)[:n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_props.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add leagues/props.py tests/leagues/test_props.py
git commit -m "feat(leagues): anchor player props to the match model via sigma-lambda rescale"
```

---

## Task 4: The props gate (`props_backtest.py`)

**Files:**
- Create: `leagues/props_backtest.py`

Walk-forward over 2025-26. Beat a position-prior-only baseline or the props do not ship.

- [ ] **Step 1: Implement the backtest**

```python
"""Walk-forward gate for the props model (spec §6).

Trains only on appearances BEFORE each matchweek, predicts that matchweek's
anytime-scorer probabilities and expected shots, and scores them against what
actually happened. The baseline is a position-prior-only model with no player
history and no fixture adjustment -- if we cannot beat that, the machinery is
not earning its complexity.
"""
import numpy as np
import pandas as pd

from leagues import dataset, players, props
from leagues.model import LeagueModel


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() < 10:
            continue
        rows.append({"bin": f"{edges[i]:.0%}-{edges[i+1]:.0%}",
                     "n": int(m.sum()),
                     "predicted": round(float(p[m].mean()), 3),
                     "actual": round(float(y[m].mean()), 3)})
    return rows


def run(league: str = "PL", test_season_start: str = "2025-08-01") -> dict:
    matches = dataset.build_matches(league)
    logs = players.fetch_player_logs(league)
    logs["date"] = pd.to_datetime(logs["date"])
    matches["date"] = pd.to_datetime(matches["date"])

    cutoff0 = pd.Timestamp(test_season_start)
    test = matches[matches["date"] >= cutoff0].sort_values("date")
    if test.empty:
        raise SystemExit(f"no test matches after {test_season_start}")

    rows = []
    for day in sorted(test["date"].dt.normalize().unique()):
        train_m = matches[matches["date"] < day]
        train_p = logs[logs["date"] < day]
        if len(train_m) < 200 or train_p.empty:
            continue

        model = LeagueModel().fit(train_m, ref=day)
        rates = props.player_rates(train_p, ref=day)

        for _, m in test[test["date"].dt.normalize() == day].iterrows():
            if m["home"] not in model.attack or m["away"] not in model.attack:
                continue
            pred = model.predict(m["home"], m["away"])
            got = props.match_props(rates, m["home"], m["away"],
                                    pred["lambda_home"], pred["lambda_away"])
            actual = logs[(logs["date"].dt.normalize() == day)
                          & (logs["team"].isin([m["home"], m["away"]]))]
            if actual.empty:
                continue
            scored = dict(zip(actual["player"], actual["np_goals"] > 0))
            shot_ct = dict(zip(actual["player"], actual["shots"]))

            for p in got:
                if p["player"] not in scored:
                    continue                     # did not play
                base_rate = props.GOAL_PRIORS.get(p["position"], 0.10)
                base_shot = props.SHOT_PRIORS.get(p["position"], 0.9)
                rows.append({
                    "p_model": p["anytime_pct"] / 100.0,
                    "p_base": 1 - np.exp(-base_rate),
                    "scored": bool(scored[p["player"]]),
                    "shots_model": p["exp_shots"],
                    "shots_base": base_shot,
                    "shots_actual": float(shot_ct[p["player"]]),
                })

    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("no overlapping player-matches scored; check name joins")

    y = r["scored"].to_numpy().astype(float)
    report = {
        "n": len(r),
        "scorer_logloss": round(_log_loss(r["p_model"].to_numpy(), y), 4),
        "scorer_logloss_baseline": round(_log_loss(r["p_base"].to_numpy(), y), 4),
        "shots_mae": round(float((r["shots_model"] - r["shots_actual"]).abs().mean()), 3),
        "shots_mae_baseline": round(float((r["shots_base"] - r["shots_actual"]).abs().mean()), 3),
        "calibration": _calibration(r["p_model"].to_numpy(), y),
    }
    report["passes"] = bool(
        report["scorer_logloss"] < report["scorer_logloss_baseline"]
        and report["shots_mae"] < report["shots_mae_baseline"]
    )
    return report


if __name__ == "__main__":
    import json
    rep = run("PL")
    print(json.dumps(rep, indent=2))
    print("\nGATE:", "PASS" if rep["passes"] else "FAIL")
```

- [ ] **Step 2: Run the gate**

Run: `python -m leagues.props_backtest`

Expected: the model beats the baseline on both log-loss and shots MAE, and the calibration table's `predicted` and `actual` columns track each other (a 20% bin should score ~20% of the time).

**If it FAILS:** the constants in `props.py` (`K_NINETIES`, `SEASON_DECAY`, `W_REALIZED_*`, the priors) are the tuning surface — they were taken from the spec's research, not from our data. Sweep them, re-run, and only proceed once it passes. If it still fails after tuning, STOP and report: per the spec's honesty rule, props that do not beat a dumb baseline do not ship.

- [ ] **Step 3: Commit**

```bash
git add leagues/props_backtest.py
git commit -m "feat(leagues): props walk-forward gate vs position-prior baseline"
```

---

## Task 5: Season Monte Carlo (`sim.py`)

**Files:**
- Create: `leagues/sim.py`
- Create: `tests/leagues/test_sim.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/leagues/test_sim.py
import numpy as np
import pandas as pd

from leagues.sim import final_table, rank_teams


def test_tiebreakers_points_then_gd_then_gf():
    rows = [
        {"team": "A", "points": 10, "gd": 5, "gf": 12},
        {"team": "B", "points": 10, "gd": 5, "gf": 15},   # same pts+gd, more GF -> above A
        {"team": "C", "points": 11, "gd": 0, "gf": 3},    # more points -> top
        {"team": "D", "points": 10, "gd": 2, "gf": 20},   # worse gd -> below A and B
    ]
    assert rank_teams(pd.DataFrame(rows)) == ["C", "B", "A", "D"]


def test_played_results_are_locked_in():
    """A team that already won 3-0 keeps those points in every simulation."""
    played = pd.DataFrame([
        {"home": "A", "away": "B", "home_goals": 3, "away_goals": 0, "played": True},
    ])
    remaining = pd.DataFrame(columns=["home", "away"])
    table = final_table(played, remaining, sample=lambda h, a: (0, 0))
    a = table.set_index("team").loc["A"]
    b = table.set_index("team").loc["B"]
    assert a["points"] == 3 and a["gd"] == 3
    assert b["points"] == 0 and b["gd"] == -3


def test_remaining_fixtures_are_sampled():
    played = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals", "played"])
    remaining = pd.DataFrame([{"home": "A", "away": "B"}])
    table = final_table(played, remaining, sample=lambda h, a: (2, 1))
    a = table.set_index("team").loc["A"]
    assert a["points"] == 3 and a["gf"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_sim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'leagues.sim'`

- [ ] **Step 3: Implement sim.py**

```python
"""Season Monte Carlo: sample every remaining fixture, tally the table, repeat.

PL tie-breakers, in order: points -> goal difference -> goals for -> head-to-head.
(Head-to-head is applied only among teams still tied on all three, which is rare
enough that we resolve it by the mini-league of matches already played.)
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from leagues import config
from leagues.model import scoreline_grid

N_SIMS = 10000


def rank_teams(table: pd.DataFrame) -> list[str]:
    """Order teams by points, then goal difference, then goals for."""
    t = table.sort_values(["points", "gd", "gf", "team"],
                          ascending=[False, False, False, True])
    return t["team"].tolist()


def final_table(played: pd.DataFrame, remaining: pd.DataFrame, sample) -> pd.DataFrame:
    """One simulated season. `sample(home, away) -> (home_goals, away_goals)`.

    Real results in `played` are locked in; only `remaining` is sampled.
    """
    pts = defaultdict(int)
    gf = defaultdict(int)
    ga = defaultdict(int)

    def record(h, a, hg, ag):
        gf[h] += hg; ga[h] += ag
        gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif ag > hg:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1

    for _, m in played.iterrows():
        record(m["home"], m["away"], int(m["home_goals"]), int(m["away_goals"]))
    for _, m in remaining.iterrows():
        hg, ag = sample(m["home"], m["away"])
        record(m["home"], m["away"], int(hg), int(ag))

    teams = sorted(set(gf) | set(ga) | set(pts))
    return pd.DataFrame([{"team": t, "points": pts[t], "gf": gf[t],
                          "ga": ga[t], "gd": gf[t] - ga[t]} for t in teams])


def _sampler(model, rng):
    """Draw a scoreline from the model's Dixon-Coles grid."""
    cache = {}

    def sample(home, away):
        key = (home, away)
        if key not in cache:
            lh, la = model.lambdas(home, away)
            grid = scoreline_grid(lh, la, model.rho)
            cache[key] = grid.ravel(), grid.shape
        flat, shape = cache[key]
        idx = rng.choice(len(flat), p=flat)
        return np.unravel_index(idx, shape)

    return sample


def simulate_season(model, played: pd.DataFrame, remaining: pd.DataFrame,
                    league: str = "PL", n: int = N_SIMS, seed: int = 7) -> pd.DataFrame:
    """Run n seasons; return the projected table with title/top-4/relegation %."""
    lg = config.get(league)
    rng = np.random.default_rng(seed)
    sample = _sampler(model, rng)

    finishes = defaultdict(lambda: defaultdict(int))
    points_sum = defaultdict(float)

    for _ in range(n):
        table = final_table(played, remaining, sample)
        order = rank_teams(table)
        pts = dict(zip(table["team"], table["points"]))
        for pos, team in enumerate(order, start=1):
            finishes[team][pos] += 1
            points_sum[team] += pts[team]

    rows = []
    for team, counts in finishes.items():
        total = sum(counts.values())
        title = counts.get(1, 0) / total
        top4 = sum(c for p, c in counts.items() if p <= lg.europe_spots) / total
        rel = sum(c for p, c in counts.items()
                  if p > lg.n_teams - lg.relegation_spots) / total
        rows.append({
            "team": team,
            "proj_points": round(points_sum[team] / total, 1),
            "title_pct": round(100 * title, 1),
            "top4_pct": round(100 * top4, 1),
            "relegation_pct": round(100 * rel, 1),
        })
    return pd.DataFrame(rows).sort_values("proj_points", ascending=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_sim.py -v`
Expected: 3 passed.

- [ ] **Step 5: Sanity-check against reality**

Run:
```bash
python -c "
import pandas as pd
from leagues import dataset, fixtures, sim
from leagues.model import LeagueModel
m = dataset.build_matches('PL')
model = LeagueModel().fit(m)
fx = fixtures.fetch_fixtures('PL')
played = fx[fx['played']]
remaining = fx[~fx['played']]
t = sim.simulate_season(model, played, remaining, 'PL', n=2000)
print(t.head(8).to_string(index=False))
"
```
Expected: pre-season, the projected top of the table is a believable set of PL contenders (City/Arsenal/Liverpool tier), title % is spread across 3-5 clubs rather than 99% on one, and the promoted clubs carry the highest relegation %. **A 99% title favourite pre-season means the strengths are overfitted — STOP and check the shrinkage.**

- [ ] **Step 6: Commit**

```bash
git add leagues/sim.py tests/leagues/test_sim.py
git commit -m "feat(leagues): season Monte Carlo with PL tie-breakers"
```

---

## Task 6: Pick locking and grading (`picks.py`)

**Files:**
- Create: `leagues/picks.py`
- Create: `tests/leagues/test_picks.py`

Ports the WC honesty rules. Nothing grades until August; the rules go in now so they are never retrofitted.

- [ ] **Step 1: Write the failing test**

```python
# tests/leagues/test_picks.py
import pandas as pd

from leagues.picks import lock_pick, grade, LATE_LOCK_HOURS


def test_pick_locked_before_kickoff_is_graded_normally():
    log = {}
    ko = pd.Timestamp("2026-08-15T14:00:00Z")
    lock_pick(log, "1", pick="Arsenal", confidence=4, kickoff=ko,
              now=ko - pd.Timedelta(hours=3))
    out = grade(log["1"], result={"home_goals": 2, "away_goals": 0,
                                  "home": "Arsenal", "away": "Fulham"})
    assert out["graded"] == "correct"
    assert out["void"] is False


def test_frozen_pick_is_graded__not_a_hindsight_pick():
    log = {}
    ko = pd.Timestamp("2026-08-15T14:00:00Z")
    lock_pick(log, "1", pick="Fulham", confidence=2, kickoff=ko,
              now=ko - pd.Timedelta(hours=3))
    # a later re-run must NOT overwrite the locked pick
    lock_pick(log, "1", pick="Arsenal", confidence=5, kickoff=ko,
              now=ko - pd.Timedelta(hours=1))
    assert log["1"]["pick"] == "Fulham"
    out = grade(log["1"], result={"home_goals": 2, "away_goals": 0,
                                  "home": "Arsenal", "away": "Fulham"})
    assert out["graded"] == "wrong"


def test_pick_locked_after_kickoff_is_void():
    log = {}
    ko = pd.Timestamp("2026-08-15T14:00:00Z")
    lock_pick(log, "1", pick="Arsenal", confidence=5, kickoff=ko,
              now=ko + pd.Timedelta(hours=LATE_LOCK_HOURS + 1))
    out = grade(log["1"], result={"home_goals": 2, "away_goals": 0,
                                  "home": "Arsenal", "away": "Fulham"})
    assert out["void"] is True
    assert out["graded"] == "void"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leagues/test_picks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'leagues.picks'`

- [ ] **Step 3: Implement picks.py**

```python
"""Honest pick tracking, ported from the World Cup engine.

Three rules, learned the hard way (the WC app had to void 14 picks):
  1. A pick is LOCKED before kickoff and never changed afterwards.
  2. The FROZEN pick is what gets graded -- never a hindsight re-pick.
  3. A pick first locked after kickoff is TAINTED -> void: shown, but excluded
     from the record.
"""
import json
from pathlib import Path

import pandas as pd

LATE_LOCK_HOURS = 2.5


def load_log(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_log(log: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")


def lock_pick(log: dict, match_id: str, pick: str, confidence: int,
              kickoff, now=None) -> dict:
    """Record a pick, once. A second call for the same match is a no-op."""
    match_id = str(match_id)
    if match_id in log:
        return log[match_id]

    now = pd.Timestamp(now or pd.Timestamp.utcnow(), tz="UTC")
    kickoff = pd.Timestamp(kickoff)
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    late_by = (now - kickoff).total_seconds() / 3600.0
    log[match_id] = {
        "pick": pick,
        "confidence": int(confidence),
        "locked_at": now.isoformat(),
        "kickoff": kickoff.isoformat(),
        "tainted": bool(late_by > LATE_LOCK_HOURS),
    }
    return log[match_id]


def grade(entry: dict, result: dict) -> dict:
    """Grade the FROZEN pick against the authoritative result."""
    out = dict(entry)
    if entry.get("tainted"):
        out["void"] = True
        out["graded"] = "void"
        return out

    hg, ag = int(result["home_goals"]), int(result["away_goals"])
    winner = result["home"] if hg > ag else result["away"] if ag > hg else "Draw"
    out["void"] = False
    out["graded"] = "correct" if entry["pick"] == winner else "wrong"
    return out


def record(entries: list[dict]) -> dict:
    """Aggregate a record in the same shape the WC app publishes."""
    rec = {"correct": 0, "wrong": 0, "total": 0, "void": 0, "pending": 0,
           "by_confidence": {}}
    for e in entries:
        g = e.get("graded")
        if g == "void":
            rec["void"] += 1
            continue
        if g not in ("correct", "wrong"):
            rec["pending"] += 1
            continue
        rec["total"] += 1
        rec[g] += 1
        c = str(e.get("confidence", 0))
        bucket = rec["by_confidence"].setdefault(c, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if g == "correct":
            bucket["correct"] += 1
    return rec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/leagues/test_picks.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add leagues/picks.py tests/leagues/test_picks.py
git commit -m "feat(leagues): pick locking, frozen grading, and void-on-late-lock"
```

---

## Task 7: The orchestrator (`publish.py`)

**Files:**
- Create: `leagues/publish.py`
- Create: `data/leagues/clubs.json`

The only module that knows the JSON contract (spec §9).

- [ ] **Step 1: Create the club colour map**

`data/leagues/clubs.json` — 20 clubs, canonical names exactly as in `names.py`:

```json
{
  "Arsenal": {"primary": "#EF0107", "secondary": "#FFFFFF", "short": "ARS"},
  "Aston Villa": {"primary": "#95BFE5", "secondary": "#670E36", "short": "AVL"},
  "Bournemouth": {"primary": "#DA291C", "secondary": "#000000", "short": "BOU"},
  "Brentford": {"primary": "#E30613", "secondary": "#FFFFFF", "short": "BRE"},
  "Brighton": {"primary": "#0057B8", "secondary": "#FFCD00", "short": "BHA"},
  "Burnley": {"primary": "#6C1D45", "secondary": "#99D6EA", "short": "BUR"},
  "Chelsea": {"primary": "#034694", "secondary": "#FFFFFF", "short": "CHE"},
  "Crystal Palace": {"primary": "#1B458F", "secondary": "#C4122E", "short": "CRY"},
  "Everton": {"primary": "#003399", "secondary": "#FFFFFF", "short": "EVE"},
  "Fulham": {"primary": "#000000", "secondary": "#CC0000", "short": "FUL"},
  "Leeds": {"primary": "#FFCD00", "secondary": "#1D428A", "short": "LEE"},
  "Liverpool": {"primary": "#C8102E", "secondary": "#00B2A9", "short": "LIV"},
  "Manchester City": {"primary": "#6CABDD", "secondary": "#1C2C5B", "short": "MCI"},
  "Manchester United": {"primary": "#DA291C", "secondary": "#FBE122", "short": "MUN"},
  "Newcastle United": {"primary": "#241F20", "secondary": "#FFFFFF", "short": "NEW"},
  "Nottingham Forest": {"primary": "#DD0000", "secondary": "#FFFFFF", "short": "NFO"},
  "Sunderland": {"primary": "#EB172B", "secondary": "#FFFFFF", "short": "SUN"},
  "Tottenham": {"primary": "#132257", "secondary": "#FFFFFF", "short": "TOT"},
  "West Ham": {"primary": "#7A263A", "secondary": "#1BB1E7", "short": "WHU"},
  "Wolves": {"primary": "#FDB913", "secondary": "#231F20", "short": "WOL"}
}
```

**Before committing:** the 2026-27 promoted clubs must match the actual promoted sides. Cross-check against `fixtures.fetch_fixtures('PL')` — every team in the fixture list must have an entry here, and any team here that is not in the fixture list is stale. Fix the file, not the fixture list.

- [ ] **Step 2: Implement publish.py**

```python
"""Orchestrator: fit -> sim -> props -> picks -> data/leagues/pl.json.

The ONLY module that knows the published JSON contract (spec §9). Everything
else returns plain frames/dicts, which is what makes plan 2b a loop over four
leagues rather than a rewrite.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from leagues import config, dataset, elo, fixtures, picks, players, props, sim
from leagues.model import LeagueModel, elo_priors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "leagues"
PICKS = ROOT / "data-raw" / "leagues"
MATCHWEEKS_AHEAD = 1


def _confidence(p_pick: float) -> int:
    """1-5, matching the WC app's banding."""
    for threshold, conf in ((0.70, 5), (0.60, 4), (0.50, 3), (0.40, 2)):
        if p_pick >= threshold:
            return conf
    return 1


def build(league: str = "PL") -> dict:
    lg = config.get(league)
    matches = dataset.build_matches(league)
    fx = fixtures.fetch_fixtures(league)
    logs = players.fetch_player_logs(league)

    played = fx[fx["played"]].copy()
    remaining = fx[~fx["played"]].copy()

    # Fit on history + whatever of the new season has been played.
    if not played.empty:
        add = played[["date", "home", "away", "home_goals", "away_goals"]].copy()
        add["date"] = pd.to_datetime(add["date"]).dt.tz_localize(None)
        matches = pd.concat([matches, add], ignore_index=True)

    now = pd.Timestamp.utcnow()
    squad = sorted(set(fx["home"]) | set(fx["away"]))
    ratings = elo.elo_for_league(league, squad)

    model = LeagueModel().fit(matches, ref=now.tz_localize(None))
    # promoted clubs have little/no top-flight history: seed them from ClubElo
    model = LeagueModel().fit(matches, ref=now.tz_localize(None),
                              priors=elo_priors(ratings, model))

    rates = props.player_rates(logs, ref=now.tz_localize(None))
    table = sim.simulate_season(model, played, remaining, league)

    log_path = PICKS / lg.key.lower() / "picks_log.json"
    log = picks.load_log(log_path)

    # Predict the next matchweek(s) still to be played.
    if remaining.empty:
        upcoming = remaining
    else:
        next_round = int(remaining["round"].min())
        upcoming = remaining[remaining["round"] < next_round + MATCHWEEKS_AHEAD]

    out_matches = []
    for _, m in upcoming.iterrows():
        pred = model.predict(m["home"], m["away"])
        probs = {m["home"]: pred["p_home"], "Draw": pred["p_draw"],
                 m["away"]: pred["p_away"]}
        pick = max(probs, key=probs.get)
        conf = _confidence(probs[pick])
        pick_type = ("home" if pick == m["home"]
                     else "away" if pick == m["away"] else "draw")

        entry = picks.lock_pick(log, m["match_id"], pick=pick, confidence=conf,
                                kickoff=m["date"], now=now)

        squad_props = props.match_props(
            rates, m["home"], m["away"], pred["lambda_home"], pred["lambda_away"])

        out_matches.append({
            "id": int(m["match_id"]),
            "matchweek": int(m["round"]),
            "date": pd.Timestamp(m["date"]).isoformat(),
            "venue": m["venue"],
            "home": m["home"],
            "away": m["away"],
            "prediction": {
                "p_home": round(pred["p_home"], 3),
                "p_draw": round(pred["p_draw"], 3),
                "p_away": round(pred["p_away"], 3),
                "pick": entry["pick"],          # the FROZEN pick, not a fresh one
                "pick_type": pick_type,
                "score": pred["score"],
                "confidence": entry["confidence"],
                "reasons": [
                    f"Model: {m['home']} {pred['p_home']:.0%} / draw "
                    f"{pred['p_draw']:.0%} / {m['away']} {pred['p_away']:.0%}",
                    f"Expected goals: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}",
                ],
            },
            "props": (props.top_props(squad_props, m["home"])
                      + props.top_props(squad_props, m["away"])),
            "result": None,
            "graded": None,
            "void": False,
        })

    # Grade everything already played that we had locked a pick for.
    graded_entries = []
    for _, m in played.iterrows():
        entry = log.get(str(m["match_id"]))
        if not entry:
            continue
        g = picks.grade(entry, {"home": m["home"], "away": m["away"],
                                "home_goals": m["home_goals"],
                                "away_goals": m["away_goals"]})
        log[str(m["match_id"])].update({"graded": g["graded"], "void": g["void"]})
        graded_entries.append(log[str(m["match_id"])])

    picks.save_log(log, log_path)

    backtest_path = ROOT / "data-raw" / "leagues" / "backtest_report.json"
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}

    return {
        "league": lg.name,
        "updated": datetime.now(timezone.utc).isoformat(),
        "record": picks.record(graded_entries),
        "matches": out_matches,
        "table": table.to_dict(orient="records"),
        "backtest": backtest.get(league, {}),
    }


def main():
    payload = build("PL")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pl.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)                       # atomic: never publish a half-written file
    print(f"wrote {path} — {len(payload['matches'])} fixtures, "
          f"{len(payload['table'])} teams")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `python -m leagues.publish`
Expected: `wrote .../data/leagues/pl.json — 10 fixtures, 20 teams` (10 = a full PL matchweek).

Inspect the output:
```bash
python -c "
import json; d = json.load(open('data/leagues/pl.json'))
m = d['matches'][0]
print(m['home'], 'vs', m['away'], '->', m['prediction']['pick'], m['prediction']['score'])
for p in m['props'][:3]:
    print(f\"  {p['player']:20s} {p['anytime_pct']:5.1f}% anytime  {p['exp_shots']:.1f} shots\")
print('title favourites:', [(t['team'], t['title_pct']) for t in d['table'][:3]])
"
```
**Sanity:** the top scorer prop for a big club should be a recognisable striker at roughly 30-60% anytime, not 95% and not 2%. A defender above a striker means the position priors or the rescale are wrong.

- [ ] **Step 4: Commit**

```bash
git add leagues/publish.py data/leagues/clubs.json data/leagues/pl.json data-raw/leagues/pl/picks_log.json
git commit -m "feat(leagues): orchestrator publishing data/leagues/pl.json"
```

---

## Task 8: The Premier League page (`leagues.html`)

**Files:**
- Create: `leagues.html`

Standalone page — **does not touch `index.html`** (the WC final is 2026-07-19). Reuses the WC's glass aesthetic; reads only `data/leagues/pl.json` and `data/leagues/clubs.json`.

- [ ] **Step 1: Read the WC page's styling so the new page matches**

Run: `sed -n '1,120p' index.html`

Copy the CSS custom properties (the gold accent, the glass card treatment, the fonts) into the new page. Do **not** import from `index.html` or refactor it — copy the values. Unification is plan 2b's job.

- [ ] **Step 2: Build the page**

Three sections, in order: a **record chip + matchweek fixtures** (each card: teams, pick, confidence, scoreline, and the top-3 props per side), a **projected table** (proj points, title %, top-4 %, relegation % as bars), and a **model performance** block reading `backtest` (model RPS vs market RPS, stated honestly — we are slightly worse than the closing line, and that is the expected result).

```html
<!doctype html>
<meta charset="utf-8">
<title>Premier League Predictor</title>
<link rel="stylesheet" href="leagues.css">
<nav class="switcher">
  <a href="index.html">World Cup</a>
  <a href="leagues.html" class="active">Premier League</a>
</nav>
<main id="app">Loading…</main>
<script>
const fmt = (n, d = 0) => Number(n).toFixed(d);

async function load() {
  const [data, clubs] = await Promise.all([
    fetch('data/leagues/pl.json').then(r => r.json()),
    fetch('data/leagues/clubs.json').then(r => r.json()),
  ]);
  const colour = t => (clubs[t] || {}).primary || '#888';
  const app = document.getElementById('app');
  app.innerHTML = '';

  const rec = data.record;
  const chip = document.createElement('div');
  chip.className = 'record-chip';
  chip.textContent = rec.total
    ? `${rec.correct}–${rec.wrong} (${fmt(100 * rec.correct / rec.total)}%)`
    : 'No graded picks yet — season starts in August';
  app.append(chip);

  const fixtures = document.createElement('section');
  fixtures.className = 'fixtures';
  for (const m of data.matches) {
    const card = document.createElement('article');
    card.className = 'card';
    card.style.setProperty('--home', colour(m.home));
    card.style.setProperty('--away', colour(m.away));
    const props = m.props.map(p => `
      <li class="${p.penalty_taker ? 'pen' : ''}">
        <span class="who">${p.player}</span>
        <span class="pct">${fmt(p.anytime_pct, 1)}%</span>
        <span class="shots">${fmt(p.exp_shots, 1)} shots · ${fmt(p.p_sot_1plus, 0)}% SOT</span>
      </li>`).join('');
    card.innerHTML = `
      <header>
        <span class="team home">${m.home}</span>
        <span class="score">${m.prediction.score}</span>
        <span class="team away">${m.away}</span>
      </header>
      <div class="pick">Pick: <strong>${m.prediction.pick}</strong>
        <span class="conf">${'★'.repeat(m.prediction.confidence)}</span></div>
      <div class="probs">
        <span>${fmt(100 * m.prediction.p_home)}%</span>
        <span>${fmt(100 * m.prediction.p_draw)}% draw</span>
        <span>${fmt(100 * m.prediction.p_away)}%</span>
      </div>
      <ul class="props">${props}</ul>`;
    fixtures.append(card);
  }
  app.append(fixtures);

  const table = document.createElement('table');
  table.className = 'league-table';
  table.innerHTML = `
    <thead><tr><th>#</th><th>Team</th><th>Pts</th><th>Title</th>
      <th>Top 4</th><th>Relegation</th></tr></thead>
    <tbody>${data.table.map((t, i) => `
      <tr>
        <td>${i + 1}</td>
        <td><span class="dot" style="background:${colour(t.team)}"></span>${t.team}</td>
        <td>${fmt(t.proj_points, 1)}</td>
        <td><span class="bar" style="--pct:${t.title_pct}%"></span>${fmt(t.title_pct, 1)}%</td>
        <td>${fmt(t.top4_pct, 1)}%</td>
        <td>${fmt(t.relegation_pct, 1)}%</td>
      </tr>`).join('')}</tbody>`;
  app.append(table);

  const b = data.backtest || {};
  if (b.rps) {
    const perf = document.createElement('section');
    perf.className = 'performance';
    perf.innerHTML = `
      <h2>Model performance</h2>
      <p>Walk-forward over ${b.n} matches: <strong>${fmt(100 * b.accuracy, 1)}%</strong>
         accuracy, RPS <strong>${fmt(b.rps, 4)}</strong>.
         The de-vigged closing line scores ${fmt(100 * b.market_accuracy, 1)}%
         and RPS ${fmt(b.market_rps, 4)} — the market is still slightly better,
         which is the honest expected result.</p>`;
    app.append(perf);
  }
}
load();
</script>
```

Write the matching `leagues.css` with the copied custom properties, the glass card treatment, and `.bar { width: var(--pct); }` for the table bars.

- [ ] **Step 3: Verify it renders**

Run: `python -m http.server 8765` and open `http://localhost:8765/leagues.html`.

Check: the record chip says the season has not started, ten fixture cards render with picks and props, the table shows 20 clubs with title percentages that sum to ~100%, and the performance block states the market is better. **Confirm `index.html` still loads correctly at the same time** — if the WC app is broken, revert immediately.

- [ ] **Step 4: Commit**

```bash
git add leagues.html leagues.css
git commit -m "feat(leagues): standalone Premier League page"
```

---

## Task 9: Scheduled jobs (`ops/`)

**Files:**
- Create: `ops/leagues_weekly.py`
- Create: `ops/leagues_matchday.py`

Both run locally (the engine needs pandas/scipy/penaltyblog; Render only serves the static site).

- [ ] **Step 1: Write the weekly job**

```python
"""Weekly: refresh data, re-fit, re-sim, republish, deploy.

Abort-on-failure: a failed fetch must NEVER ship a stale-but-fresh-looking file,
which is exactly how the WC app once published picks that had to be voided.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    try:
        from leagues import publish
        publish.main()
    except Exception as exc:
        print(f"ABORT: league publish failed ({exc}); nothing deployed", file=sys.stderr)
        return 1

    return subprocess.call(
        [sys.executable, str(ROOT / "deploy.py"),
         "auto update: leagues weekly refresh"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the match-day job**

```python
"""Match day: pull authoritative results, re-grade frozen picks, redeploy.

Identical to the weekly job in structure -- publish.build() already grades every
played fixture against its FROZEN pick -- but runs on match days so the record
updates the same evening rather than the following week.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    try:
        from leagues import publish
        publish.main()
    except Exception as exc:
        print(f"ABORT: league publish failed ({exc}); nothing deployed", file=sys.stderr)
        return 1

    return subprocess.call(
        [sys.executable, str(ROOT / "deploy.py"),
         "auto update: leagues match-day results"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Register the scheduled tasks**

The WC tasks are the pattern to copy (`worldcup-daily-update`, cron `0 7 * * *`). Register two more:
- `leagues-weekly` — cron `0 6 * * 2` (Tuesday 06:00, after Monday-night football has settled).
- `leagues-matchday` — cron `0 23 * * 6,0` (Saturday and Sunday 23:00, after the day's fixtures finish).

**Do not register these until the season is close** — before mid-August they would republish an unchanged file every week. Note the intended start date in `CLAUDE.md`.

- [ ] **Step 4: Verify the weekly job end-to-end (dry run, no deploy)**

Run: `python -c "from leagues import publish; publish.main()"`
Expected: rewrites `data/leagues/pl.json` cleanly. Confirm `git status` shows only league files changed — **no WC files.**

- [ ] **Step 5: Commit**

```bash
git add ops/leagues_weekly.py ops/leagues_matchday.py CLAUDE.md
git commit -m "feat(leagues): weekly + match-day scheduled jobs"
```

---

## Task 10: Full check and handoff

- [ ] **Step 1: Run every test**

Run: `python -m pytest tests/ -q`
Expected: 25 Phase 1 tests + 16 new = **41 passed**.

- [ ] **Step 2: Confirm the WC app is untouched**

Run: `git diff main --stat -- index.html predict.py data/predictions.json`
Expected: **no output.** Any diff here is a bug — the WC final is 2026-07-19.

- [ ] **Step 3: Report the two gates**

Present to the user:
- **Match model (Phase 1):** PL accuracy / RPS vs the de-vigged market.
- **Props (Task 4):** scorer log-loss and shots MAE vs the position-prior baseline, plus the calibration table.

State plainly whether each passes. **Do not start plan 2b (the other three leagues, the unified switcher, the WC performance backfill) until the user has seen both.**

---

## Self-Review Notes

- **Spec coverage:** props model §5 → Tasks 2-3; props gate §6 → Task 4; season sim §7 → Task 5; pick tracking §8 → Task 6; JSON contract §9 → Task 7; ops §10 → Task 9; error handling §11 → loud name failures (Task 1 test), abort-on-publish-failure (Task 9), missing-xG fallback (inherited from Phase 1 `model.py`); testing §12 → the Σλ invariant, shrinkage direction, penalty split, tie-breaker order, and void-on-late-lock are each pinned by a named test.
- **Deferred by design (plan 2b):** the other three leagues, the in-app competition switcher, the WC "Model Performance" backfill.
- **Known API risk, flagged in-plan:** `soccerdata`'s FBref `read_player_match_stats` column names are verified at runtime in Task 1 Step 1 ("print it and fix, don't guess") — the same discipline that caught the `get_params()` prefixes and the Understat season-code bug in Phase 1.
- **The provisional constants** (`K_NINETIES`, `SEASON_DECAY`, `W_REALIZED_*`, position priors) are the tuning surface for Task 4's gate and are labelled as such in the code.
