"""Soccer player props on the REBUILT model (leagues/forecast.py, docs/REBUILD_PLAN.md).

leagues/publish still decides who is in each squad (rosters, transfers, news); this
module replaces the old rate-table numbers (props.match_props) with the per-match
forecast: who starts, who comes on, and what each does in either case.

  * trained ONCE per run on all five leagues (the last TRAIN_SEASONS seasons plus
    this season's played matches, data-raw/leagues/understat_matches/, kept current
    by scripts/sync_understat_matches.py);
  * each board fixture is found among Understat's upcoming matches by the two clubs
    and the date;
  * news: a player ruled OUT does not play; DOUBTFUL halves his chance of featuring;
    a confirmed XI fixes the starters, and the named bench comes on at the rate the
    role model gives;
  * published at BARS, which the gate measured landing at the rate stated
    (2022-25, all five leagues: scorer 46.6% said / 47.7% landed, 2+ shots 69/70,
    on target 68/68). No after-the-fact recalibration is applied.
Picks carry engine=ENGINE.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from leagues import forecast as fc
from leagues import names

ENGINE = "forecast-v1"
BARS = {"goal": 0.40, "shots": 0.62, "sot": 0.62}
PER_MATCH = 3
HORIZON_DAYS = 10
DOUBT_FACTOR = 0.5
SUB_MINUTES = 20.0
_STATE: dict = {}


def _norm(s) -> str:
    from nfl import odds as odds_mod
    return odds_mod.norm_name(str(s or ""))


def _season_year(now) -> int:
    return now.year if now.month >= 7 else now.year - 1


def _upcoming(now) -> pd.DataFrame:
    try:
        rows = json.loads((fc.MATCHES_DIR / "upcoming.json").read_text(encoding="utf-8"))["matches"]
    except Exception:
        return pd.DataFrame()
    up = pd.DataFrame(rows)
    if up.empty:
        return up
    up["date"] = pd.to_datetime(up["date"])
    naive = now.replace(tzinfo=None)
    up = up[(up["date"] >= naive - pd.Timedelta(hours=6))
            & (up["date"] <= naive + pd.Timedelta(days=HORIZON_DAYS))]
    sides = []
    for r in up.itertuples():
        for team, opp, home in ((r.home, r.away, 1), (r.away, r.home, 0)):
            sides.append({"league": r.league, "season": int(r.season), "match_id": int(r.match_id),
                          "date": r.date.normalize(), "kickoff": r.date, "team": team,
                          "opponent": opp, "is_home": home})
    return pd.DataFrame(sides)


def prepare(now=None) -> dict:
    """Load, build and fit once per process."""
    if _STATE.get("model") is not None:
        return _STATE
    now = now or datetime.now(timezone.utc)
    year = _season_year(now)
    # This season's played matches and the upcoming list: incremental, never fatal
    # (a failed league keeps the rows it had).
    try:
        from scripts import sync_understat_matches
        sync_understat_matches.main()
    except Exception as exc:                               # noqa: BLE001
        print(f"  props [{ENGINE}]: Understat sync failed ({type(exc).__name__}: {exc}); "
              f"using the rows on hand")
    d = fc.load(range(year - fc.TRAIN_SEASONS, year + 1))
    up = _upcoming(now)
    built = fc.build(d, up[["league", "season", "match_id", "date", "team", "opponent", "is_home"]]
                     if len(up) else None)
    train = built[~built["is_next"] & (built["season"] >= year - fc.TRAIN_SEASONS)]
    model = fc.PropForecast().fit(train)
    _STATE.update(model=model, built=built, upcoming=up)
    print(f"  props [{ENGINE}]: trained on {len(train)} candidate rows; "
          f"{up['match_id'].nunique() if len(up) else 0} upcoming matches", flush=True)
    return _STATE


def _find(league: str, home: str, away: str, kickoff) -> int | None:
    up = _STATE.get("upcoming")
    if up is None or up.empty:
        return None
    up = up[(up["league"] == league) & (up["is_home"] == 1)]
    k = pd.Timestamp(kickoff)
    k = k.tz_convert(None) if k.tzinfo else k
    for r in up.itertuples():
        if (names.canonical(r.team, league) == home and names.canonical(r.opponent, league) == away
                and abs((r.kickoff - k).total_seconds()) <= 2 * 86400):
            return int(r.match_id)
    return None


def match_forecast(league, home, away, kickoff, unavailable=(), doubtful=(),
                   starters=(), bench=()) -> list | None:
    """Per-player props for one fixture, or None when the fixture is not found (or
    the model could not be prepared this run)."""
    st = _STATE if _STATE.get("model") is not None else None
    if st is None:
        return None
    mid = _find(league, home, away, kickoff)
    if mid is None:
        return None
    built = st["built"]
    rows = built[built["is_next"] & (built["match_id"] == mid)].copy()
    if rows.empty:
        return []
    model = st["model"]
    roles = model.roles(rows)
    out_n = {_norm(n) for n in unavailable or ()}
    doubt_n = {_norm(n) for n in doubtful or ()}
    start_n = {_norm(n) for n in starters or ()}
    bench_n = {_norm(n) for n in bench or ()}
    teams = [names.canonical(t, league) for t in rows["team"]]
    xi_known = {t for t in set(teams)
                if any(names.canonical(r.team, league) == t and _norm(r.player) in start_n
                       for r in rows.itertuples())}
    for i, (r, team) in enumerate(zip(rows.itertuples(), teams)):
        n = _norm(r.player)
        if n in out_n:
            roles[i] = [1.0, 0.0, 0.0]
            continue
        if team in xi_known:
            if n in start_n:
                roles[i] = [0.0, 0.0, 1.0]
            else:
                on = roles[i, 1] / max(roles[i, 0] + roles[i, 1], 1e-6) if n in bench_n else 0.0
                roles[i] = [1 - on, on, 0.0]
        elif n in doubt_n:
            roles[i, 1:] *= DOUBT_FACTOR
            roles[i, 0] = 1 - roles[i, 1:].sum()
    pred = model.predict(rows, roles)
    res = []
    for i, (r, team) in enumerate(zip(rows.itertuples(), teams)):
        p_app = float(roles[i, 1] + roles[i, 2])
        mins = float(roles[i, 2] * (r.min_start_e10 if pd.notna(r.min_start_e10) else 80.0)
                     + roles[i, 1] * SUB_MINUTES)
        res.append({"team": team, "player": r.player, "player_id": int(r.player_id),
                    "anytime_pct": round(100 * float(pred["goal"][i]), 1),
                    "p_shots_2plus": round(100 * float(pred["shots"][i]), 1),
                    "p_sot_1plus": round(100 * float(pred["sot"][i]), 1),
                    "appearance_pct": round(100 * p_app, 1),
                    "start_pct": round(100 * float(roles[i, 2]), 1),
                    "expected_minutes": round(mins, 1),
                    "penalty_taker": bool((r.pen_share or 0) >= 0.5),
                    "engine": ENGINE})
    return res


def apply(squad_props: list, forecast: list) -> list:
    """The old squad list with the rebuilt numbers; players the forecast has no read
    on (no match for this club in its window -- a new signing, a long absence) are
    left out rather than shown with the old model's number."""
    by = {(p["team"], _norm(p["player"])): p for p in forecast}
    out = []
    for p in squad_props:
        f = by.get((p["team"], _norm(p["player"])))
        if f is None:
            continue
        q = dict(p)
        for k in ("anytime_pct", "p_shots_2plus", "p_sot_1plus", "appearance_pct",
                  "expected_minutes", "penalty_taker", "start_pct", "engine"):
            q[k] = f[k]
        q["exp_shots"] = q["exp_sot"] = None      # the old model's counts, not this one's
        out.append(q)
    return out
