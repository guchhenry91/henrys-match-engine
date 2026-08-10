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

    def sample_odds(fixtures, label):
        if not fixtures:
            print(f"\nno {label} fixture found to sample odds for")
            return
        fid = fixtures[0]["fixture"]["id"]
        home = fixtures[0]["teams"]["home"]["name"]
        away = fixtures[0]["teams"]["away"]["name"]
        date = fixtures[0]["fixture"]["date"]
        status = fixtures[0]["fixture"]["status"]["short"]
        print(f"\n=== odds for {label}: {home} v {away} "
              f"({date}, status={status}, fixture {fid}) ===")
        odds = client.get("odds", fixture=fid)
        print(f"  raw response length: {len(odds)}")
        if not odds:
            return
        for bookmaker in odds[0].get("bookmakers", []):
            print(f"  -- {bookmaker['name']} --")
            for bet in bookmaker.get("bets", []):
                print(f"     {bet['name']}: "
                      f"{[(v['value'], v['odd']) for v in bet['values'][:5]]}")

    # The dashboard's package list includes both Pre-match and In-play Odds, so
    # empty results below need real cause-finding, not just "not covered" --
    # try several angles: our tracked league far out, a marquee league/fixture
    # recently finished (heavy bookmaker coverage), and today's live matches.
    sample_odds(client.get("fixtures", league=39, season=2026, next=1), "next PL")
    sample_odds(client.get("fixtures", league=39, season=2025, last=1), "past PL (2025-26 season)")
    # Champions League final -- about as heavily covered by bookmakers as any
    # fixture gets, so if odds exist ANYWHERE in the archive, this is where.
    sample_odds(client.get("fixtures", league=2, season=2025, last=1), "past Champions League")
    # Today's live matches, if any -- tests /odds specifically for in-play
    # coverage rather than pre-match.
    live = client.get("fixtures", live="all")
    print(f"\n{len(live)} live fixtures right now")
    if live:
        sample_odds(live, "a LIVE fixture right now")

    # Last resort: what does /odds return with NO fixture filter at all --
    # does the endpoint have anything queryable today, for any match?
    any_odds = client.get("odds", league=39, season=2026)
    print(f"\n/odds with no fixture filter (league=39, season=2026): "
          f"{len(any_odds)} results")

    print(f"\n{client.used} API-Football requests used")


if __name__ == "__main__":
    main()
