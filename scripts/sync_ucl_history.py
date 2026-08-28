"""Cache sixteen seasons of Champions League results from API-Football.

ONE REQUEST PER SEASON -- sixteen calls for 3,407 finished matches, and only for
seasons not already cached. The current season is always refetched because it is
the one that changes; the past never does.

Written to data-raw/ucl/history.json and committed, so the backtest and the board
run from a file rather than the API. That is not only about quota: a model whose
inputs can silently change between runs cannot be checked against its own earlier
numbers.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from leagues.api_football import Client
from ucl import config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "ucl" / "history.json"
FIXTURES_OUT = ROOT / "data-raw" / "ucl" / "fixtures.json"
FINISHED = {"FT", "AET", "PEN"}


def load() -> dict:
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        return {"seasons": {}}


def parse(rows) -> list:
    """Finished matches only, in the shape the model wants.

    NINETY MINUTES, NOT THE FINAL SCORE AFTER EXTRA TIME. A knockout tie settled
    in extra time or on penalties is a DRAW as a football match, and recording the
    winner's scoreline would teach the model that these clubs score more than they
    do. API-Football keeps the 90-minute score separately for exactly this reason.
    """
    out = []
    for row in rows:
        fixture = row.get("fixture") or {}
        status = (fixture.get("status") or {}).get("short")
        if status not in FINISHED:
            continue
        teams, goals = row.get("teams") or {}, row.get("score") or {}
        full = goals.get("fulltime") or {}
        home_goals, away_goals = full.get("home"), full.get("away")
        if home_goals is None or away_goals is None:
            # Fall back to the aggregate goals field, but never to extra time.
            g = row.get("goals") or {}
            home_goals, away_goals = g.get("home"), g.get("away")
        if home_goals is None or away_goals is None:
            continue
        home = ((teams.get("home") or {}).get("name"))
        away = ((teams.get("away") or {}).get("name"))
        date = fixture.get("date")
        if not (home and away and date):
            continue
        out.append({"date": date[:10], "home": home, "away": away,
                    "home_goals": int(home_goals), "away_goals": int(away_goals),
                    "round": (row.get("league") or {}).get("round"),
                    "status": status})
    return out


def upcoming(rows) -> list:
    """The drawn league-phase fixtures, played or not.

    Kept separately from history because they are a different thing: history is
    what happened, this is what was drawn. Qualifying rounds are excluded here --
    the board is about the league phase the 36 clubs were drawn into.
    """
    out = []
    for row in rows:
        fixture = row.get("fixture") or {}
        league = row.get("league") or {}
        rnd = str(league.get("round") or "")
        if "League Stage" not in rnd and "League Phase" not in rnd:
            continue
        teams = row.get("teams") or {}
        home = ((teams.get("home") or {}).get("name"))
        away = ((teams.get("away") or {}).get("name"))
        status = (fixture.get("status") or {}).get("short")
        if not (home and away):
            continue
        out.append({"date": (fixture.get("date") or "")[:10], "matchday": rnd,
                    "home": home, "away": away,
                    "played": status in FINISHED})
    return sorted(out, key=lambda f: (f["date"], f["home"]))


def main():
    cache = load()
    seasons = cache.get("seasons") or {}
    try:
        client = Client(limit=25)
    except RuntimeError as exc:
        print(f"{exc}; keeping the cached history")
        return 0

    fetched = 0
    for season in config.SEASONS:
        key = str(season)
        # The past does not change; only refetch the season still being played.
        if key in seasons and season != config.CURRENT_SEASON:
            continue
        try:
            rows = client.get("fixtures", league=config.API_LEAGUE_ID, season=season)
        except Exception as exc:
            print(f"  {season}: FAILED {exc}")
            continue
        seasons[key] = parse(rows)
        if season == config.CURRENT_SEASON:
            drawn = upcoming(rows)
            FIXTURES_OUT.parent.mkdir(parents=True, exist_ok=True)
            FIXTURES_OUT.write_text(
                json.dumps({"_note": "Drawn league-phase fixtures for the "
                                     "current season.",
                            "season": season, "fixtures": drawn}, indent=1),
                encoding="utf-8")
            print(f"  {season}: {len(drawn)} league-phase fixtures drawn")
        fetched += 1
        print(f"  {season}: {len(seasons[key])} finished matches")

    if not seasons:
        print("no history retrieved and nothing cached; writing nothing")
        return 0

    total = sum(len(v) for v in seasons.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_note": ("Champions League results from API-Football, one request per "
                  "season. Ninety-minute scores only -- a tie settled in extra "
                  "time is a draw as a football match. Generated by "
                  "scripts/sync_ucl_history.py; do not hand-edit."),
        "updated": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(f"{total} matches across {len(seasons)} seasons "
          f"({fetched} fetched this run); {client.used} request(s) used")
    return 0


if __name__ == "__main__":
    sys.exit(main())
