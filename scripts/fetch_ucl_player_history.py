"""Player stats for past Champions League matches between two big-five clubs.

The test data for Champions League player props (ucl/props_backtest.py): did players'
league form predict their goals and shots in Europe? One API-Football request per
fixture (/fixtures/players), only fixtures not already saved, at most RUN_LIMIT a
run, under the client's 1,500-a-day account floor. Written to
data-raw/ucl/player_history.json -- facts about played matches, never a pick.
"""
import json
import os
import sys
from pathlib import Path

from leagues.api_football import Client
from ucl.props_names import to_understat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "ucl" / "player_history.json"
SEASONS = range(2021, 2026)
RUN_LIMIT = 350
SKIP_ROUNDS = ("Qualif", "Preliminary", "Play-off")


def wanted(client) -> list:
    """Finished non-qualifying fixtures between two big-five clubs. The cached history
    holds no fixture ids before 2026, so each season's list is fetched once (5 calls)."""
    from scripts.sync_ucl_history import parse as parse_fixtures
    from ucl import config
    out = []
    for s in SEASONS:
        for r in parse_fixtures(client.get("fixtures", league=config.API_LEAGUE_ID, season=s)):
            if r.get("id") is None or any(k in str(r.get("round")) for k in SKIP_ROUNDS):
                continue
            if to_understat(r["home"]) and to_understat(r["away"]):
                out.append({**r, "season": s})
    return out


def parse(rows) -> list:
    players = []
    for side in rows or []:
        team = (side.get("team") or {}).get("name")
        for p in side.get("players") or []:
            st = (p.get("statistics") or [{}])[0]
            games, shots, goals = st.get("games") or {}, st.get("shots") or {}, st.get("goals") or {}
            players.append({"team": team, "player_id": (p.get("player") or {}).get("id"),
                            "player": (p.get("player") or {}).get("name"),
                            "minutes": games.get("minutes") or 0,
                            "substitute": bool(games.get("substitute")),
                            "position": games.get("position"),
                            "goals": goals.get("total") or 0,
                            "shots": shots.get("total") or 0,
                            "sot": shots.get("on") or 0})
    return players


def main(client=None) -> int:
    try:
        store = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        store = {"fixtures": {}}
    have = store.setdefault("fixtures", {})
    if client is None:
        if not os.environ.get("API_FOOTBALL_KEY"):
            print("API_FOOTBALL_KEY is not set")
            return 1
        client = Client(limit=RUN_LIMIT + len(SEASONS))
    todo = [f for f in wanted(client) if str(f["id"]) not in have]
    print(f"{len(todo)} fixture(s) to fetch, {len(have)} already saved")
    if not todo:
        return 0
    got = 0
    for f in todo[:RUN_LIMIT]:
        try:
            rows = client.get("fixtures/players", fixture=f["id"])
        except Exception as exc:                          # noqa: BLE001
            print(f"  stopped at fixture {f['id']}: {exc}")
            break
        players = parse(rows)
        if not players:
            continue                                      # not filed yet: try next run
        have[str(f["id"])] = {"season": f["season"], "date": f["date"], "round": f.get("round"),
                              "home": f["home"], "away": f["away"],
                              "home_goals": f.get("home_goals"), "away_goals": f.get("away_goals"),
                              "players": players}
        got += 1
    OUT.write_text(json.dumps(store, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {got} fixture(s); {len(have)} in the file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
