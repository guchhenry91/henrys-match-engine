"""One-off diagnostic: what does our API-Football plan actually give us on /odds?

Not wired into any scheduled job. Run via workflow_dispatch
(.github/workflows/odds-diagnostic.yml) since the key only lives in the GitHub
Actions secret. Prints the full pre-match bet-type list (so we can see whether
player-prop markets -- Anytime Goalscorer, player shots/SOT -- exist at all,
not just Correct Score) and the bookmaker list, then pulls one live PL fixture's
odds as a concrete sample of what's actually returned.
"""
import json

from leagues.api_football import Client


def main():
    client = Client(limit=10)

    bets = client.get("odds/bets")
    print(f"=== odds/bets: {len(bets)} pre-match bet types ===")
    for b in bets:
        print(f"  id={b['id']:>3}  {b['name']}")

    books = client.get("odds/bookmakers")
    print(f"\n=== odds/bookmakers: {len(books)} bookmakers ===")
    names = [b["name"] for b in books if b.get("name")]
    print(" ", ", ".join(names))
    print(f"  bet365 present: {any('bet365' in n.lower() for n in names)}")

    fixtures = client.get("fixtures", league=39, season=2026, next=1)
    if fixtures:
        fid = fixtures[0]["fixture"]["id"]
        home = fixtures[0]["teams"]["home"]["name"]
        away = fixtures[0]["teams"]["away"]["name"]
        print(f"\n=== odds for next PL fixture: {home} v {away} (fixture {fid}) ===")
        odds = client.get("odds", fixture=fid)
        if not odds:
            print("  no odds returned for this fixture (may be too far out)")
        else:
            for bookmaker in odds[0].get("bookmakers", []):
                print(f"  -- {bookmaker['name']} --")
                for bet in bookmaker.get("bets", []):
                    print(f"     {bet['name']}: "
                          f"{[(v['value'], v['odd']) for v in bet['values'][:5]]}")
    else:
        print("\nno upcoming PL fixture found to sample odds for")

    print(f"\n{client.used} API-Football requests used")


if __name__ == "__main__":
    main()
