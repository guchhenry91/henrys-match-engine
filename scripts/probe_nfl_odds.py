"""One-off reconnaissance of API-NFL's odds coverage. Prints, changes nothing.

Written before the odds integration rather than during it, because the two things
that decide whether "beat the books" is even possible here cannot be assumed:

  1. Is bet365 among the bookmakers this account can see?
  2. Are PLAYER PROP markets offered, or only game lines?

If it is only game lines, the team-winner board can be priced against a real book
and the four player markets cannot -- and it would be dishonest to ship an "edge"
column on props that is secretly computed against something else. Better to find
that out from the API than to discover it after building on the assumption.

Deliberately cheap: a handful of requests, and it reports what it spent.
"""
import json
import sys

from nfl import api, data, publish


def show(title, rows, limit=25):
    print(f"\n=== {title} ({len(rows)} rows) ===")
    for row in rows[:limit]:
        print("   ", json.dumps(row)[:220])


def main():
    if not api.available():
        print("API_NFL_KEY not set or allowance spent; nothing probed")
        return 0
    client = api.Client(budget=12)

    try:
        books = client.get("odds/bookmakers")
        show("bookmakers", books, limit=60)
        bet365 = [b for b in books
                  if "365" in str(b.get("name", "")).lower()]
        print(f"\n  bet365 present: {bool(bet365)} -> {bet365}")
    except Exception as exc:
        print(f"bookmakers lookup failed: {exc}")
        bet365 = []

    try:
        bets = client.get("odds/bets")
        show("bet types offered", bets, limit=80)
        prop_like = [b for b in bets if any(
            k in str(b.get("name", "")).lower()
            for k in ("player", "passing", "rushing", "receiving", "touchdown",
                      "anytime"))]
        print(f"\n  PLAYER-PROP bet types: {len(prop_like)}")
        for b in prop_like[:30]:
            print("     ", b)
    except Exception as exc:
        print(f"bet types lookup failed: {exc}")

    # The odds endpoint rejects league/season, so ask it the two ways it accepts:
    # by DATE, and per GAME across the earliest few unplayed fixtures. If nothing
    # anywhere is priced, the answer is "books have not posted yet", not "this API
    # has no odds" -- and those demand different responses from me.
    import datetime as dt
    today = dt.date.today()
    for offset in (0, 1, 2, 7, 14):
        day = (today + dt.timedelta(days=offset)).isoformat()
        try:
            rows = client.get("odds", date=day)
        except Exception as exc:
            print(f"  odds date={day}: FAILED {exc}")
            continue
        print(f"  odds date={day}: {len(rows)} row(s)")
        if rows:
            books = [b.get("name") for r in rows for b in (r.get("bookmakers") or [])]
            print(f"    bookmakers: {sorted(set(books))}")
            sample = rows[0]
            for b in (sample.get("bookmakers") or [])[:3]:
                names = [x.get("name") for x in (b.get("bets") or [])]
                print(f"    {b.get('name')}: {len(names)} markets -> {names[:10]}")
                for bet in (b.get("bets") or [])[:3]:
                    vals = [(v.get("value"), v.get("odd"))
                            for v in (bet.get("values") or [])[:4]]
                    print(f"       {bet.get('name')}: {vals}")
            break

    print(f"\n{client.report()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
