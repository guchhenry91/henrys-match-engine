"""The test for Champions League player props: does league form predict Europe?

The soccer props model (leagues/forecast.py) learns from Understat's big-five league
matches. A Champions League prop would price a player from that same league form,
against an opponent measured in ITS league -- often a different one. Nothing has
checked that the result holds in European matches, where rotation, opponents and
game states differ. This does, before anything is published.

Walk-forward, the soccer props gate's method (scripts/soccer_props_gate.py):
  - each Champions League season 2021-22 .. 2025-26 is priced by a model trained on
    the TRAIN_SEASONS league seasons before it;
  - each match's features are built from league matches BEFORE its date only, one
    match per club per batch (a club's other European games must not read as games
    every player missed);
  - outcomes from API-Football's per-player match stats
    (scripts/fetch_ucl_player_history.py), joined within one club in one fixture
    (players.resolve_squad_name). A player who did not take the field is VOID.
Scored as the gate scores the league: calibration over every candidate, and the
PUBLISHED subset (market bar, likely starter, top PER_MATCH a match) as said vs
landed -- before lineups and with the confirmed XI.

PASS (per market, with the XI, the way picks lock): at least MIN_PUBLISHED published
picks, landed no more than SLACK below said, and the said rate inside the 95% Wilson
interval of landed. Writes data-raw/ucl/props_backtest_report.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import dist
from leagues import forecast as fc
from leagues.players import resolve_squad_name
from scripts.soccer_props_gate import BARS, MIN_START, PER_MATCH, published
from ucl.props import batches, features
from ucl.props_names import to_understat

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data-raw" / "ucl" / "player_history.json"
REPORT = ROOT / "data-raw" / "ucl" / "props_backtest_report.json"
LEAGUE_REPORT = ROOT / "data-raw" / "leagues" / "props_forecast_report.json"
MIN_PUBLISHED = 60
SLACK = 0.03


def outcomes(rows: pd.DataFrame, fx: dict) -> pd.DataFrame:
    """Attach what each candidate did, joined within his club in that fixture."""
    by_mid = {f["mid"]: f for f in fx.values()}
    played, started, cols = [], [], {m: [] for m in fc.MARKETS.values()}
    joined = 0
    for r in rows.itertuples():
        f = by_mid[r.match_id]
        api_team = f["home"] if r.team == f["home_us"] else f["away"]
        squad = [p for p in f["players"] if p["team"] == api_team]
        hit = resolve_squad_name(str(r.player), [p["player"] for p in squad])
        p = next((p for p in squad if p["player"] == hit), None) if hit else None
        mins = (p or {}).get("minutes") or 0
        joined += p is not None
        played.append(mins > 0)
        started.append(bool(p and mins > 0 and not p.get("substitute")))
        cols["scored"].append(float((p or {}).get("goals", 0) > 0))
        cols["shots2"].append(float((p or {}).get("shots", 0) >= 2))
        cols["sot1"].append(float((p or {}).get("sot", 0) >= 1))
    out = rows.copy()
    out["played"], out["started_real"] = played, started
    for c, v in cols.items():
        out[c] = v
    out["role"] = np.where(out["started_real"], 2, np.where(out["played"], 1, 0))
    out.attrs["joined"] = joined
    return out


def wilson(k: float, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def verdict(pub: dict) -> dict:
    n = pub.get("n", 0)
    if n < MIN_PUBLISHED:
        return {"pass": False, "why": f"only {n} published picks (need {MIN_PUBLISHED})"}
    lo, hi = wilson(pub["landed"] * n, n)
    ok = pub["landed"] >= pub["said"] - SLACK and lo <= pub["said"] <= hi
    return {"pass": bool(ok), "landed_95": [round(lo, 4), round(hi, 4)],
            "why": None if ok else f"said {pub['said']:.1%}, landed {pub['landed']:.1%}"}


ESPN = ROOT / "data-raw" / "ucl" / "espn_history.json"


def espn_fixtures(raw: dict) -> dict:
    """The same fixtures with ESPN's player lines (shots INCLUDING blocked, as bet365's
    Opta-settled markets count them) in API-Football's shape (ucl/espn.py)."""
    from ucl.espn import same_club
    esp = json.loads(ESPN.read_text(encoding="utf-8"))["fixtures"]
    out = {}
    for fid, f in raw.items():
        e = esp.get(fid)
        if not e:
            continue
        players = []
        for p in e["players"]:
            club = f["home"] if same_club(p["team"], f["home"]) else (
                f["away"] if same_club(p["team"], f["away"]) else None)
            if club:
                players.append({"team": club, "player": p["player"],
                                "minutes": 90 if p["played"] else 0,
                                "substitute": not p["started"], "goals": p["goals"],
                                "shots": p["shots"], "sot": p["sot"]})
        out[fid] = {**f, "players": players}
    return out


def main() -> int:
    t0 = time.time()
    raw = json.loads(HISTORY.read_text(encoding="utf-8"))["fixtures"]
    source = "espn" if "--espn" in sys.argv else "api-football"
    if source == "espn":
        raw = espn_fixtures(raw)
    fixtures = {}
    for i, (fid, f) in enumerate(sorted(raw.items())):
        h, a = to_understat(f["home"]), to_understat(f["away"])
        if h and a and f.get("players"):
            fixtures[fid] = {**f, "home_us": h, "away_us": a, "mid": -(i + 1)}
    seasons = sorted({f["season"] for f in fixtures.values()})
    league = fc.load(range(min(seasons) - fc.TRAIN_SEASONS, max(seasons) + 1))
    frame = fc.build(league)
    print(f"{len(fixtures)} UCL fixtures, seasons {seasons}; league frame {len(frame)} rows "
          f"in {time.time() - t0:.0f}s", flush=True)
    acc = {m: {"pre": [], "xi": [], "y": [], "rows": []} for m in fc.MARKETS}
    joined = candidates = 0
    for season in seasons:
        train = frame[(frame["season"] < season) & (frame["season"] >= season - fc.TRAIN_SEASONS)]
        model = fc.PropForecast().fit(train)
        fx = {k: v for k, v in fixtures.items() if v["season"] == season}
        parts = []
        for b in batches(list(fx.values())):
            rows = features(league, b, season)
            if len(rows):
                parts.append(outcomes(rows, fx))
                joined += parts[-1].attrs["joined"]
        test = pd.concat(parts, ignore_index=True)
        candidates += len(test)
        pre = model.predict(test)
        r = model.roles(test)
        bench = r[:, 1] / np.clip(r[:, 0] + r[:, 1], 1e-6, None)
        xi_roles = np.zeros_like(r)
        st = (test["role"] == 2).to_numpy()
        xi_roles[st, 2] = 1.0
        xi_roles[~st, 1] = bench[~st]
        xi_roles[~st, 0] = 1 - bench[~st]
        xi = model.predict(test, xi_roles)
        played = test["played"].to_numpy(bool)
        for m, col in fc.MARKETS.items():
            acc[m]["pre"].append(pre[m][played]); acc[m]["xi"].append(xi[m][played])
            acc[m]["y"].append(test[col].to_numpy(float)[played])
            acc[m]["rows"].append(pd.DataFrame({"match_id": test["match_id"].to_numpy()[played],
                                                "p_start_pre": pre["p_start"][played],
                                                "p_start_xi": xi["p_start"][played]}))
        print(f"  {season}: {len(fx)} fixtures, {len(test)} candidates, "
              f"{int(played.sum())} played, {time.time() - t0:.0f}s", flush=True)
    try:
        lg = json.loads(LEAGUE_REPORT.read_text(encoding="utf-8"))["markets"]
    except Exception:
        lg = {}
    report = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
              "outcomes_from": source, "seasons": seasons, "fixtures": len(fixtures), "candidates": candidates,
              "joined_to_api": joined, "bars": BARS, "per_match": PER_MATCH,
              "min_start": MIN_START, "rule": {"min_published": MIN_PUBLISHED, "slack": SLACK},
              "markets": {}}
    for m in fc.MARKETS:
        y = np.concatenate(acc[m]["y"]); rows = pd.concat(acc[m]["rows"])
        out = {"n": int(len(y)), "base_rate": round(float(y.mean()), 4)}
        for k in ("pre", "xi"):
            p = np.concatenate(acc[m][k])
            rows = rows.assign(p_start=rows[f"p_start_{k}"])
            ll = float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
            pub = published(rows, p, y, BARS[m])
            out[k] = {"calibration": dist.calibration(p, y), "logloss": round(ll, 4),
                      "said_all": round(float(p.mean()), 4), "published": pub,
                      "verdict": verdict(pub)}
        out["league_gate"] = {k: (lg.get(m, {}).get(k) or {}).get("published") for k in ("pre", "xi")}
        out["pass"] = out["xi"]["verdict"]["pass"]
        report["markets"][m] = out
        print(f"{m}: base {out['base_rate']} | all said {out['xi']['said_all']} | "
              f"XI ECE {out['xi']['calibration']['ece']} pub {out['xi']['published']} "
              f"-> {'PASS' if out['pass'] else 'FAIL'} {out['xi']['verdict'].get('why') or ''}", flush=True)
    path = REPORT.with_name("props_backtest_report_espn.json") if source == "espn" else REPORT
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
