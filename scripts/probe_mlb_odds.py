"""Probe: which MLB markets do Pinnacle / DraftKings / FanDuel price, in what shape?

One event, the six markets the board needs, three named books (one region-
equivalent): at most 6 credits. Manual only (odds-probe.yml).
"""
from datetime import datetime, timezone

from oddsapi import client as oc
from oddsapi import props as shared

MARKETS = ["batter_hits", "batter_hits_runs_rbis", "batter_home_runs", "batter_rbis",
           "pitcher_strikeouts", "team_totals"]


def main():
    c = oc.Client()
    events = c.get("sports/baseball_mlb/events", sport="mlb", purpose="probe: events") or []
    now = datetime.now(timezone.utc)
    ahead = sorted((e for e in events if shared.utc(e["commence_time"]) > now),
                   key=lambda e: e["commence_time"])
    print(f"{len(ahead)} upcoming MLB events")
    if not ahead:
        return
    ev = ahead[0]
    print(f"{ev['away_team']} @ {ev['home_team']} {ev['commence_time']}")
    payload = c.get(f"sports/baseball_mlb/events/{ev['id']}/odds", sport="mlb",
                    purpose="probe: mlb markets", est=oc.cost(len(MARKETS), bookmakers=3),
                    bookmakers="pinnacle,draftkings,fanduel", markets=",".join(MARKETS),
                    oddsFormat="decimal") or {}
    for book in payload.get("bookmakers", []):
        for m in book.get("markets", []):
            outs = m.get("outcomes", [])
            print(f"  {book['key']:<11} {m['key']:<24} {len(outs)} outcomes; "
                  f"e.g. {outs[:3]}")
    print(c.report())


if __name__ == "__main__":
    main()
