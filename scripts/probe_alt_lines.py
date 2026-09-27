"""Probe: do DraftKings / FanDuel quote ALTERNATE NFL yardage lines, and in what shape?

One event, three alternate markets, two named books (one region-equivalent):
at most 3 credits under oddsapi.Client's caps. Manual only (odds-probe.yml).
"""
from datetime import datetime, timezone

from nfl import book_lines as bl
from oddsapi import client as oc

ALT = ["player_rush_yds_alternate", "player_reception_yds_alternate",
       "player_pass_yds_alternate"]


def main():
    c = oc.Client()
    events = c.get(f"sports/{bl.SPORT_KEY}/events", sport="nfl", purpose="probe: events") or []
    now = datetime.now(timezone.utc)
    ahead = sorted((e for e in events if bl._utc(e["commence_time"]) > now),
                   key=lambda e: e["commence_time"])
    if not ahead:
        print("no upcoming NFL event")
        return
    ev = ahead[0]
    print(f"{ev['away_team']} @ {ev['home_team']} {ev['commence_time']}")
    payload = c.get(f"sports/{bl.SPORT_KEY}/events/{ev['id']}/odds", sport="nfl",
                    purpose="probe: alternate lines", est=oc.cost(len(ALT), bookmakers=2),
                    bookmakers="draftkings,fanduel", markets=",".join(ALT),
                    oddsFormat="decimal") or {}
    for book in payload.get("bookmakers", []):
        for m in book.get("markets", []):
            outs = m.get("outcomes", [])
            players = sorted({o.get("description") for o in outs})
            print(f"  {book['key']:<11} {m['key']:<32} {len(outs)} outcomes, {len(players)} players, "
                  f"sides {sorted({o.get('name') for o in outs})}")
            first = players[0] if players else None
            ladder = sorted((o["point"], o["price"], o["name"]) for o in outs
                            if o.get("description") == first)
            print(f"     e.g. {first}: {ladder[:10]}")
    print(c.report())


if __name__ == "__main__":
    main()
