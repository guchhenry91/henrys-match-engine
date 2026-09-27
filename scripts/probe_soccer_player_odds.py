"""Print the exact shape of bet365's PLAYER markets in API-Football's odds.

One-off diagnostic before wiring soccer prop prices: which bet ids carry anytime
scorer, shots and shots on target, and how each value names the player and the
line. Manual only (odds-diagnostic.yml). About six API-Football requests.
"""
import json

from leagues.api_football import Client

WORDS = ("scorer", "shot", "player")


def main():
    client = Client(limit=8)
    fixtures = client.get("fixtures", league=39, season=2026, next=3)
    for fx in fixtures[:3]:
        fid = fx["fixture"]["id"]
        print(f"\n=== {fx['teams']['home']['name']} v {fx['teams']['away']['name']} "
              f"({fx['fixture']['date']}, fixture {fid}) ===")
        rows = client.get("odds", fixture=fid, bookmaker=8)
        for r in rows:
            for bm in r.get("bookmakers", []):
                for bet in bm.get("bets", []):
                    if not any(w in bet["name"].lower() for w in WORDS):
                        continue
                    vals = bet.get("values") or []
                    print(f"  bet id={bet['id']} name={bet['name']!r} values={len(vals)}")
                    for v in vals[:6]:
                        print(f"     {json.dumps(v, ensure_ascii=False)}")
    print(client.report())


if __name__ == "__main__":
    main()
