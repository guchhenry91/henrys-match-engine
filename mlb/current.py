"""The MLB season being played, from MLB's own free StatsAPI (statsapi.mlb.com).

Retrosheet publishes a season only after it ends, so the live season -- 2026
regular season and postseason -- comes from StatsAPI instead: the schedule (with
probable pitchers) and one box score per finished game, kept in
data-raw/mlb/current/ so each game is fetched once.

Rows are written in the SAME shape as the Retrosheet history (mlb/data.py) so the
features and the model need no second path:
  * players keyed by their RETROSHEET id, via the Chadwick register
    (data-raw/mlb/id_register.csv.gz); a player with no mapping keeps "mlbam:<id>"
    and simply has no history before this season;
  * teams by their Retrosheet code (NYA, LAN, ...), via TEAM_CODES.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data-raw" / "mlb" / "current"
REGISTER = ROOT / "data-raw" / "mlb" / "id_register.csv.gz"
API = "https://statsapi.mlb.com/api/v1"
SEASON = 2026
FINAL = "Final"
FINISHED = ("Final", "Completed Early", "Game Over")
PAUSE = 0.15
MAX_BOX_FETCHES = 3000

# StatsAPI team id -> Retrosheet team code.
TEAM_CODES = {108: "ANA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHN", 113: "CIN",
              114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KCA", 119: "LAN",
              120: "WAS", 121: "NYN", 133: "ATH", 134: "PIT", 135: "SDN", 136: "SEA",
              137: "SFN", 138: "SLN", 139: "TBA", 140: "TEX", 141: "TOR", 142: "MIN",
              143: "PHI", 144: "ATL", 145: "CHA", 146: "MIA", 147: "NYA", 158: "MIL"}
GAME_TYPES = ("R", "F", "D", "L", "W")      # regular season + every postseason round


def _get(path: str, opener=urllib.request.urlopen):
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "henrys-match-engine/1.0"})
    with opener(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


_REG = None


def retro_id(mlbam) -> str:
    """MLBAM id -> Retrosheet id, or "mlbam:<id>" when the register lacks him."""
    global _REG
    if _REG is None:
        try:
            reg = pd.read_csv(REGISTER)
            _REG = {int(m): r for m, r in zip(reg["key_mlbam"], reg["key_retro"])}
        except Exception:
            _REG = {}
    return _REG.get(int(mlbam), f"mlbam:{int(mlbam)}")


def schedule(start: str, end: str, opener=urllib.request.urlopen) -> pd.DataFrame:
    """Every game between two dates: ids, times, teams, scores, probable pitchers."""
    raw = _get(f"schedule?sportId=1&startDate={start}&endDate={end}"
               f"&gameType={','.join(GAME_TYPES)}&hydrate=probablePitcher", opener)
    rows = []
    for day in raw.get("dates", []):
        for g in day["games"]:
            h, a = g["teams"]["home"], g["teams"]["away"]
            hc, ac = TEAM_CODES.get(h["team"]["id"]), TEAM_CODES.get(a["team"]["id"])
            if not hc or not ac:
                continue
            rows.append({
                "game_pk": int(g["gamePk"]), "game_type": g["gameType"],
                "game_date": g.get("officialDate") or day["date"],
                "start": g["gameDate"], "status": g["status"]["detailedState"],
                "home_team": hc, "away_team": ac,
                "home_name": h["team"]["name"], "away_name": a["team"]["name"],
                "home_score": h.get("score"), "away_score": a.get("score"),
                "home_pitcher": (h.get("probablePitcher") or {}).get("id"),
                "away_pitcher": (a.get("probablePitcher") or {}).get("id"),
                "home_pitcher_name": (h.get("probablePitcher") or {}).get("fullName"),
                "away_pitcher_name": (a.get("probablePitcher") or {}).get("fullName"),
            })
    return pd.DataFrame(rows)


def availability(games: pd.DataFrame, opener=urllib.request.urlopen) -> dict:
    """Who can actually play: each team's ACTIVE roster and, once posted, each
    game's starting lineup -- both from MLB's own feed, free.

    {"active": {team: {player ids}}, "lineups": {(game_pk, team): {player ids}}}
    with ids in the same Retrosheet form as the box-score rows. A team whose roster
    call fails is simply absent, so it is not filtered (the board is no worse than
    before); a lineup not yet posted is absent too.
    """
    out = {"active": {}, "lineups": {}}
    if games is None or games.empty:
        return out
    code_to_id = {code: tid for tid, code in TEAM_CODES.items()}
    for team in sorted(set(games["home_team"]) | set(games["away_team"])):
        tid = code_to_id.get(team)
        if tid is None:
            continue
        try:
            roster = _get(f"teams/{tid}/roster?rosterType=active", opener).get("roster") or []
            time.sleep(PAUSE)
        except Exception as exc:
            print(f"  roster {team}: not fetched ({exc}); not filtered")
            continue
        ids = {retro_id(p["person"]["id"]) for p in roster if (p.get("person") or {}).get("id")}
        if ids:
            out["active"][team] = ids
    pks = ",".join(str(int(pk)) for pk in games["game_pk"])
    try:
        raw = _get(f"schedule?sportId=1&gamePks={pks}&hydrate=lineups", opener)
    except Exception as exc:
        print(f"  lineups: not fetched ({exc}); not filtered")
        return out
    by_pk = {int(r.game_pk): r for r in games.itertuples()}
    for day in raw.get("dates", []):
        for g in day.get("games", []):
            row = by_pk.get(int(g["gamePk"]))
            if row is None:
                continue
            lineups = g.get("lineups") or {}
            for side in ("home", "away"):
                players = lineups.get(f"{side}Players") or []
                if players:
                    out["lineups"][(int(row.game_pk), getattr(row, f"{side}_team"))] = {
                        retro_id(p["id"]) for p in players if p.get("id")}
    return out


NOT_PLAYED = ("Cancelled", "Postponed")


def game_status(game_pks, opener=urllib.request.urlopen) -> dict:
    """{game_pk: detailedState} for specific games, whatever their date."""
    pks = sorted({int(k) for k in game_pks})
    if not pks:
        return {}
    raw = _get(f"schedule?sportId=1&gamePks={','.join(str(k) for k in pks)}", opener)
    return {int(g["gamePk"]): g["status"]["detailedState"]
            for day in raw.get("dates", []) for g in day.get("games", [])}


def box_rows(g: dict, opener=urllib.request.urlopen) -> dict:
    """{"batting", "pitching", "team"} rows for one finished game, history-shaped."""
    box = _get(f"game/{g['game_pk']}/boxscore", opener)
    out = {"batting": [], "pitching": [], "team": []}
    gid, date = f"MLB{g['game_pk']}", g["game_date"]
    for side, opp_side in (("home", "away"), ("away", "home")):
        t = box["teams"][side]
        team, opp = g[f"{side}_team"], g[f"{opp_side}_team"]
        home = 1.0 if side == "home" else 0.0
        players = t.get("players") or {}
        for pid in t.get("batters") or []:
            p = players.get(f"ID{pid}") or {}
            order = str(p.get("battingOrder") or "")
            if not order.endswith("00"):
                continue                      # starters only: 100, 200, ... 900
            s = (p.get("stats") or {}).get("batting") or {}
            h, r, rbi = s.get("hits", 0), s.get("runs", 0), s.get("rbi", 0)
            out["batting"].append({
                "PLAYER_ID": retro_id(pid), "MLBAM": int(pid), "NAME": (p.get("person") or {}).get("fullName"),
                "TEAM": team, "GAME_ID": gid, "game_date": date, "season": SEASON,
                "opponent": opp, "is_home": home, "PA": s.get("plateAppearances", 0),
                "H": h, "R": r, "RBI": rbi, "HR": s.get("homeRuns", 0), "HRR": h + r + rbi,
                "lineup": int(order[0])})
        pitchers = t.get("pitchers") or []
        if pitchers:
            p = players.get(f"ID{pitchers[0]}") or {}
            s = (p.get("stats") or {}).get("pitching") or {}
            out["pitching"].append({
                "PLAYER_ID": retro_id(pitchers[0]), "MLBAM": int(pitchers[0]),
                "NAME": (p.get("person") or {}).get("fullName"), "TEAM": team,
                "GAME_ID": gid, "game_date": date, "season": SEASON, "opponent": opp,
                "is_home": home, "BFP": s.get("battersFaced", 0), "K": s.get("strikeOuts", 0),
                "IPOUTS": s.get("outs", 0)})
        tb = (t.get("teamStats") or {}).get("batting") or {}
        out["team"].append({"PLAYER_ID": team, "TEAM": team, "GAME_ID": gid,
                            "game_date": date, "season": SEASON, "opponent": opp,
                            "is_home": home, "PA": tb.get("plateAppearances", 0),
                            "RUNS": tb.get("runs", 0)})
    return out


def _path(kind: str) -> Path:
    return DIR / f"{kind}_{SEASON}.csv.gz"


def load(kind: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(_path(kind), low_memory=False)
    except Exception:
        return pd.DataFrame()
    # "mixed": files written before 2026-09-27 hold both "2026-09-26" and
    # "2026-09-26 00:00:00", which a single inferred format rejects.
    frame["game_date"] = pd.to_datetime(frame["game_date"], format="mixed")
    return frame


def sync(sched: pd.DataFrame, opener=urllib.request.urlopen, sleeper=time.sleep) -> dict:
    """Fetch box scores for finished games not already kept."""
    DIR.mkdir(parents=True, exist_ok=True)
    have = load("team")
    known = set(have["GAME_ID"]) if not have.empty else set()
    todo = sched[(sched["status"].isin(FINISHED))
                 & ~sched["game_pk"].map(lambda k: f"MLB{k}").isin(known)]
    new = {"batting": [], "pitching": [], "team": []}
    fetched = failed = 0
    for g in todo.sort_values("start").to_dict("records")[:MAX_BOX_FETCHES]:
        try:
            rows = box_rows(g, opener)
            for k in new:
                new[k].extend(rows[k])
            fetched += 1
        except Exception as exc:
            failed += 1
            print(f"  box score {g['game_pk']} unavailable ({type(exc).__name__})")
        sleeper(PAUSE)
    if fetched:
        for k, rows in new.items():
            old = load(k)
            frame = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if not old.empty \
                else pd.DataFrame(rows)
            # One date format on disk: loaded rows are Timestamps, new rows strings.
            frame["game_date"] = pd.to_datetime(frame["game_date"], format="mixed").dt.strftime("%Y-%m-%d")
            frame.to_csv(_path(k), index=False, compression="gzip")
    return {"fetched": fetched, "failed": failed, "pending": max(len(todo) - fetched - failed, 0)}
