"""The season being played, from the NBA's own CDN (cdn.nba.com).

WHY NOT stats.nba.com. It timed out on every request from home on 2026-09-27
and is widely known to refuse cloud IPs, which is where the scheduled board runs.
The CDN serves the same league's data as static JSON: the full season schedule in
one file, and one box score per game. Both answered in seconds.

WHAT IT GIVES THE BOARD
  * schedule()       -- every game this season: id, tip-off (UTC), teams, status.
  * player_games()   -- this season's regular-season box scores, in the SAME
                        columns as the stats.nba.com game log (nba.data), so the
                        engine's features and grading need no second code path.
                        Player ids match: the CDN's personId IS stats.nba's
                        PLAYER_ID (checked on 2025-26 box scores).
  * current_teams()  -- each player's most recent team, from the newest box score
                        he appears in, PRESEASON INCLUDED. Box scores say where a
                        man last played; after a summer of trades that is the only
                        free evidence of where he is now.

A finished game's box score never changes, so each is fetched once and kept in
data-raw/nba/current/ (committed): about seven requests on a normal night.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from nba import config

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data-raw" / "nba" / "current"
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (henrys-match-engine)",
           "Accept": "application/json", "Referer": "https://www.nba.com/"}
PAUSE_SECONDS = 0.3
MAX_BOX_FETCHES = 80           # per run; a normal night is ~7, a missed week ~50

REGULAR, PRESEASON = "002", "001"
FINAL = 3                      # gameStatus: 1 scheduled, 2 live, 3 final
COLUMNS = ["SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
           "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "FG3M"]


def _get(url: str, opener=urllib.request.urlopen, timeout=60):
    with opener(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def schedule(opener=urllib.request.urlopen) -> pd.DataFrame:
    """Every game of the season, one row each: from the NBA's CDN, or -- when the
    CDN refuses -- from The Odds API copy kept by scripts/sync_nba_odds.py.

    THE CDN ANSWERS "ACCESS DENIED" FROM OUTSIDE ITS ALLOWED NETWORKS. Checked
    2026-10-04: a UK browser, the UK home line and Cloudflare (Europe and North
    America) were all refused, though GitHub's runners had been served until
    2026-10-02. The Odds API lists the same games for free (its /events) and the
    finals for 2 credits (/scores), so the board keeps a schedule either way."""
    try:
        return cdn_schedule(opener)
    except Exception as exc:
        fallback = odds_schedule()
        if fallback.empty:
            raise
        print(f"NBA CDN schedule unavailable ({type(exc).__name__}); using The Odds API "
              f"schedule ({len(fallback)} games)")
        return fallback


def cdn_schedule(opener=urllib.request.urlopen) -> pd.DataFrame:
    """Every game of the season on the CDN, one row each."""
    raw = _get(SCHEDULE_URL, opener)
    rows = []
    for day in raw["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            rows.append({
                "game_id": g["gameId"], "stage": g["gameId"][:3],
                "tipoff": g["gameDateTimeUTC"],
                # The US date the league files the game under -- the same date
                # stats.nba.com's GAME_DATE uses.
                "game_date": str(g.get("gameDateEst", ""))[:10],
                "home_team": g["homeTeam"]["teamTricode"],
                "away_team": g["awayTeam"]["teamTricode"],
                "home_score": g["homeTeam"].get("score"),
                "away_score": g["awayTeam"].get("score"),
                "status": int(g.get("gameStatus") or 0),
                "neutral": bool(g.get("isNeutral")),
            })
    return pd.DataFrame(rows)


def _minutes(text) -> float:
    """'PT33M04.90S' -> 33.08."""
    m = re.match(r"PT(\d+)M([\d.]+)S", str(text or ""))
    return round(int(m.group(1)) + float(m.group(2)) / 60.0, 2) if m else 0.0


def box_rows(game_id: str, game_date: str, opener=urllib.request.urlopen) -> list:
    """One gamelog-shaped row per player who PLAYED, both teams."""
    game = _get(BOX_URL.format(gid=game_id), opener)["game"]
    home, away = game["homeTeam"], game["awayTeam"]
    rows = []
    for team, opp, is_home in ((home, away, True), (away, home, False)):
        won = (team.get("score") or 0) > (opp.get("score") or 0)
        matchup = (f"{team['teamTricode']} vs. {opp['teamTricode']}" if is_home
                   else f"{team['teamTricode']} @ {opp['teamTricode']}")
        for p in team.get("players") or []:
            s = p.get("statistics") or {}
            minutes = _minutes(s.get("minutes"))
            if str(p.get("played")) != "1" or minutes <= 0:
                continue                 # the game log lists only players who played
            rows.append({
                "SEASON_ID": f"2{config.CURRENT_SEASON - 1}",
                "PLAYER_ID": int(p["personId"]), "PLAYER_NAME": p.get("name"),
                "TEAM_ID": int(team["teamId"]), "TEAM_ABBREVIATION": team["teamTricode"],
                "GAME_ID": game_id, "GAME_DATE": game_date, "MATCHUP": matchup,
                "WL": "W" if won else "L", "MIN": minutes,
                "PTS": s.get("points"), "REB": s.get("reboundsTotal"),
                "AST": s.get("assists"), "FG3M": s.get("threePointersMade"),
            })
    return rows


def _path(stage: str) -> Path:
    name = "player" if stage == REGULAR else "preseason"
    return DIR / f"{name}_{config.CURRENT_SEASON}.csv"


def _load(stage: str) -> pd.DataFrame:
    try:
        return pd.read_csv(_path(stage), dtype={"GAME_ID": str}, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def sync(sched: pd.DataFrame, opener=urllib.request.urlopen,
         sleeper=time.sleep, fetch=None, today: str | None = None) -> dict:
    """Fetch box scores for finished games not already kept. Returns counts."""
    DIR.mkdir(parents=True, exist_ok=True)
    today = today or _eastern_date(pd.Timestamp.now(tz="UTC").isoformat())
    added, failed, budget = {}, 0, MAX_BOX_FETCHES
    for stage in (PRESEASON, REGULAR):
        have = _load(stage)
        known = set(have["GAME_ID"].astype(str)) if not have.empty else set()
        todo = sched[(sched["stage"] == stage) & (sched["status"] == FINAL)
                     & ~sched["game_id"].isin(known)
                     # Odds API numbering ("0029...") has no CDN box score to ask for.
                     & ~sched["game_id"].astype(str).str.startswith("0029")].sort_values("tipoff")
        new = []
        for _, g in todo.iterrows():
            if budget <= 0:
                break
            budget -= 1
            try:
                new.extend(box_rows(g["game_id"], g["game_date"], opener))
            except Exception as exc:
                failed += 1
                print(f"  box score {g['game_id']} unavailable ({type(exc).__name__}); "
                      f"will retry next run")
            sleeper(PAUSE_SECONDS)
        if new:
            out = pd.concat([have, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True)
            tmp = _path(stage).with_suffix(".csv.tmp")
            out.to_csv(tmp, index=False)
            tmp.replace(_path(stage))
        added[stage] = len({r["GAME_ID"] for r in new})
        # THE CDN REFUSES CLOUDFLARE (and the Odds API schedule's games have no CDN
        # box score at all), so finished games it could not supply come from
        # stats.nba.com's season game log instead -- one request per stage.
        have = _load(stage)
        known = set(have["GAME_ID"].astype(str)) if not have.empty else set()
        missing = sched[(sched["stage"] == stage) & (sched["status"] == FINAL)
                        & ~sched["game_id"].astype(str).isin(known)]
        # A test's stand-in opener means no network: only a stand-in fetch may run.
        live = fetch is not None or opener is urllib.request.urlopen
        # The Odds API schedule lists NO preseason games, so nothing is ever
        # "missing" there -- yet preseason box scores are where 2026-27 trades and
        # signings first show (current_teams). Until opening night, ask anyway.
        preseason_window = stage == PRESEASON and today < config.REGULAR_SEASON_START
        if (not missing.empty or preseason_window) and live:
            try:
                added[stage] = added.get(stage, 0) + sync_from_stats(sched, stage, fetch)
            except Exception as exc:
                print(f"  stats.nba.com {STATS_TYPE[stage]} log unavailable "
                      f"({type(exc).__name__}: {exc}); {len(missing)} game(s) still missing")
    return {"regular_games_added": added.get(REGULAR, 0),
            "preseason_games_added": added.get(PRESEASON, 0), "failed": failed}


STATS_TYPE = {REGULAR: "Regular Season", PRESEASON: "Pre Season"}


def sync_from_stats(sched: pd.DataFrame, stage: str, fetch=None) -> int:
    """Add this season's player rows from stats.nba.com's leaguegamelog (the same
    endpoint, headers and columns as the history, nba.data). Returns games added.

    GAME IDS ARE THE SCHEDULE'S. Picks freeze and grade on the schedule's game id,
    which is a stand-in ("0029...") while the schedule comes from The Odds API, so
    each row is re-keyed by its date and the two teams (US Eastern dates on both
    sides; the tricodes match nba.book_lines.TEAM_CODES). A row the schedule does
    not list keeps the NBA's own id -- it still tells current_teams where a player
    is, and simply cannot grade anything."""
    from nba import data
    fetch = fetch or (lambda st: data._fetch(config.CURRENT_SEASON, "P", attempts=2,
                                             season_type=st, timeout=45))
    block = fetch(STATS_TYPE[stage])["resultSets"][0]
    rows = pd.DataFrame(block["rowSet"], columns=block["headers"])
    if rows.empty:
        return 0
    rows = rows[[c for c in COLUMNS if c in rows.columns]].copy()
    rows["MIN"] = pd.to_numeric(rows["MIN"], errors="coerce").fillna(0)
    rows = rows[rows["MIN"] > 0]
    rows["GAME_DATE"] = rows["GAME_DATE"].astype(str).str[:10]
    rows["GAME_ID"] = rows["GAME_ID"].astype(str).str.zfill(10)
    by_pair = {(str(g.game_date)[:10], frozenset((g.home_team, g.away_team))): str(g.game_id)
               for g in sched[sched["stage"] == stage].itertuples()}
    opp = rows["MATCHUP"].astype(str).str.split(r"\s+(?:@|vs\.)\s+", regex=True).str[-1]
    rows["GAME_ID"] = [by_pair.get((d, frozenset((t, o))), gid) for d, t, o, gid in
                       zip(rows["GAME_DATE"], rows["TEAM_ABBREVIATION"], opp, rows["GAME_ID"])]
    have = _load(stage)
    before = set(have["GAME_ID"].astype(str)) if not have.empty else set()
    out = pd.concat([have, rows], ignore_index=True)
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    out = out.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], keep="first")
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(stage).with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(_path(stage))
    return len(set(out["GAME_ID"]) - before)


def player_games() -> pd.DataFrame:
    """This season's regular-season rows, gamelog-shaped (may be empty)."""
    return _load(REGULAR)


def current_teams() -> dict:
    """PLAYER_ID -> team tricode, from the newest box score each player is in.

    Regular season beats preseason for the same date order, simply because it is
    later. A player with no box score this season is absent -- the caller falls
    back to last season and says so, rather than assuming he stayed."""
    frames = [f for f in (_load(PRESEASON), _load(REGULAR)) if not f.empty]
    if not frames:
        return {}
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.sort_values(["GAME_DATE", "GAME_ID"])
    latest = rows.groupby("PLAYER_ID").tail(1)
    return {int(r.PLAYER_ID): r.TEAM_ABBREVIATION for r in latest.itertuples()}


# ---- The Odds API schedule (fallback) ------------------------------------------------
ODDS_SCHEDULE = ROOT / "data-raw" / "nba" / "odds_schedule.json"
SCHED_COLUMNS = ["game_id", "stage", "tipoff", "game_date", "home_team", "away_team",
                 "home_score", "away_score", "status", "neutral"]


def _load_odds_store() -> dict:
    try:
        return json.loads(ODDS_SCHEDULE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ids": {}, "games": {}}


def odds_schedule() -> pd.DataFrame:
    """The schedule as last saved from The Odds API (empty if never saved)."""
    games = _load_odds_store().get("games") or {}
    return pd.DataFrame(list(games.values()), columns=SCHED_COLUMNS)


def _eastern_date(iso: str) -> str:
    from zoneinfo import ZoneInfo
    t = pd.Timestamp(iso)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_convert(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def update_odds_schedule(events: list, scores: list, team_codes: dict, now) -> dict:
    """Fold The Odds API's /events and /scores into the saved schedule.

    GAME IDS ARE PERMANENT. Each Odds API event gets the next number in a series no
    real NBA id uses ("0029" + 6 digits; real ids are "002" + the season, e.g.
    "00226..."), recorded once and never reassigned -- picks freeze and grade on the
    game id, so an id that changed between runs would orphan them. Numeric, because
    nba/board.py derives a placeholder id from the last seven digits."""
    store = _load_odds_store()
    ids, games = store.setdefault("ids", {}), store.setdefault("games", {})
    now_t = pd.Timestamp(now)
    now_t = now_t.tz_localize("UTC") if now_t.tzinfo is None else now_t.tz_convert("UTC")
    for ev in list(events or []) + list(scores or []):
        home, away = team_codes.get(ev.get("home_team")), team_codes.get(ev.get("away_team"))
        if not home or not away or not ev.get("id") or not ev.get("commence_time"):
            continue
        if ev["id"] not in ids:
            ids[ev["id"]] = f"0029{len(ids) + 1:06d}"
        gid = ids[ev["id"]]
        tip = pd.Timestamp(ev["commence_time"])
        tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
        row = games.get(gid) or {}
        row.update({
            "game_id": gid, "tipoff": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "game_date": _eastern_date(ev["commence_time"]),
            "stage": REGULAR if _eastern_date(ev["commence_time"]) >= config.REGULAR_SEASON_START else PRESEASON,
            "home_team": home, "away_team": away, "neutral": False,
        })
        row.setdefault("home_score", None)
        row.setdefault("away_score", None)
        if ev.get("completed"):
            by_name = {x.get("name"): x.get("score") for x in ev.get("scores") or []}
            hs, as_ = by_name.get(ev["home_team"]), by_name.get(ev["away_team"])
            if hs is not None and as_ is not None:
                row.update(home_score=int(hs), away_score=int(as_), status=FINAL)
        if row.get("status") != FINAL:
            row["status"] = 1 if tip > now_t else 2
        games[gid] = row
    store["updated"] = now_t.isoformat()
    store["_note"] = ("NBA schedule from The Odds API, used while the NBA's CDN refuses. "
                      "Written by scripts/sync_nba_odds.py -- never by hand.")
    ODDS_SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ODDS_SCHEDULE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(ODDS_SCHEDULE)
    return store
