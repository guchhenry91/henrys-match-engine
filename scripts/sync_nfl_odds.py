"""Fetch book prices for the upcoming slate. bet365 first, as asked.

ONE REQUEST PER UNPLAYED GAME, so a 16-game week costs 16 calls plus one to list
fixtures -- trivial against 7,500. It writes prices only; the edge is computed at
publish time so the number on the card always comes from the model and the price
that were current together.

AS OF 2026-08-27 THIS RETURNS NOTHING. The API lists Bet365 among its bookmakers
and defines 185 player-prop bet types, but no NFL fixture has prices posted: week
1 is 14 days out and both the bet365 query and the all-books query come back
empty, and the endpoint rejects league, season and date filters so there is no way
to ask it for whatever it does hold. That is "the books have not posted yet", not
"this API has no odds", and the two need different responses -- so this script
runs, reports honestly that it found nothing, and changes no file rather than
writing an empty one over a good one.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl import api, config, data, odds, publish

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "nfl" / "odds.json"
BET365_ID = 4

# API-Sports posts pre-match odds between 1 and 7 days before kickoff and keeps a
# 7-day history. Read from the vendor's own documentation, not inferred.
ODDS_WINDOW_DAYS = 7.0


def team_codes(client) -> dict:
    """API team id -> team code (SEA, NE, ...).

    The games endpoint returns team NAMES, not codes -- proved by the previous
    attempt, which reported all 32 of our codes as "codes the API does not use".
    /teams carries both, so one extra request buys a reliable id-based join
    instead of matching "Seattle Seahawks" against "SEA" and hoping.
    """
    try:
        teams = client.get("teams", league=1, season=config.CURRENT_SEASON)
    except Exception as exc:
        print(f"WARNING: could not list teams ({exc})")
        return {}
    out = {}
    for team in teams:
        info = team.get("team") if isinstance(team.get("team"), dict) else team
        tid, code = info.get("id"), info.get("code") or info.get("abbreviation")
        if tid and code:
            out[int(tid)] = str(code).upper()
    return out


def upcoming_ids(client, upcoming) -> dict:
    """(home, away) -> API game id, joined through team IDS."""
    codes = team_codes(client)
    if not codes:
        return {}
    try:
        fixtures = client.get("games", league=1, season=config.CURRENT_SEASON)
    except Exception as exc:
        print(f"WARNING: could not list fixtures ({exc})")
        return {}

    wanted = {(r["home_team"], r["away_team"]) for _, r in upcoming.iterrows()}
    found = {}
    for row in fixtures:
        game = row.get("game") if isinstance(row.get("game"), dict) else row
        teams = row.get("teams") or {}
        try:
            hc = codes.get(int((teams.get("home") or {}).get("id")))
            ac = codes.get(int((teams.get("away") or {}).get("id")))
        except (TypeError, ValueError):
            continue
        if hc and ac and (hc, ac) in wanted:
            found[(hc, ac)] = game.get("id")

    missing = wanted - set(found)
    if missing:
        print(f"  {len(missing)} fixture(s) unmatched, e.g. {sorted(missing)[:3]}")
        unknown = {c for pair in missing for c in pair} - set(codes.values())
        if unknown:
            # A code we use that the API has never heard of is a MAPPING problem,
            # not a coverage one, and the two need different fixes.
            print(f"  codes the API does not use: {sorted(unknown)}")
    return found


def main():
    if not api.available():
        if api.breaker_tripped():
            print("API-NFL allowance already spent today; keeping existing odds")
        else:
            print("API_NFL_KEY is not set; skipping the odds sync")
        return 0

    schedule = pd.concat([data.games(), data.games(seasons=(config.CURRENT_SEASON,))],
                         ignore_index=True)
    upcoming = publish.upcoming_games(schedule)
    if upcoming.empty:
        print("no upcoming slate; nothing to price")
        return 0

    # ONLY FIXTURES THE API CAN ACTUALLY HAVE PRICED. API-Sports posts pre-match
    # odds "between 1 and 7 days before the game", so asking about a fixture ten
    # days out is a request guaranteed to come back empty.
    #
    # This is not just wasted quota, though it is that too -- 14 of 16 week-1
    # fixtures were outside the window on 2026-09-04, about 28 doomed calls twice
    # a day. The real damage was to the CONCLUSION: every fixture returning none
    # printed "no prices available yet (16 fixture(s) returned none)", which reads
    # as "this API has no odds" when it actually meant "we asked too early". Four
    # separate runs were read that way and nearly triggered a provider switch.
    #
    # Asking only inside the window means an empty answer is INFORMATIVE: it means
    # the books really are not quoting a game they should be quoting.
    within = upcoming.copy()
    kickoff = pd.to_datetime(within["kickoff"], utc=True, errors="coerce")
    days = (kickoff - pd.Timestamp.now("UTC")).dt.total_seconds() / 86400.0
    outside = int((days > ODDS_WINDOW_DAYS).sum())
    within = within[days <= ODDS_WINDOW_DAYS]
    if outside:
        print(f"  {outside} fixture(s) beyond the {ODDS_WINDOW_DAYS}-day odds "
              f"window -- not asked about")
    if within.empty:
        nearest = days.min()
        print(f"nothing inside the {ODDS_WINDOW_DAYS}-day odds window "
              f"(nearest kickoff is {nearest:.1f} days away); nothing to price")
        return 0
    upcoming = within

    client = api.Client(budget=40)
    ids = upcoming_ids(client, upcoming)
    print(f"matched {len(ids)} of {len(upcoming)} fixtures to API game ids")

    priced, unpriced = {}, 0
    for (home, away), gid in ids.items():
        try:
            rows = client.get("odds", game=gid, bookmaker=BET365_ID) \
                or client.get("odds", game=gid)
        except api.QuotaExhausted as exc:
            print(f"  stopped early: {exc}")
            break
        except Exception as exc:
            print(f"  {home} v {away}: {exc}")
            continue
        if not rows:
            unpriced += 1
            continue
        book = odds.pick_bookmaker(rows[0].get("bookmakers") or [])
        line = odds.moneyline(book, home, away)
        if not line:
            # Recognised nothing usable. Counted, never coerced -- a silent zero
            # would read as "the book thinks this is impossible" and manufacture
            # an enormous false edge.
            unpriced += 1
            continue
        priced[f"{home}|{away}"] = line

    if not priced:
        print(f"no prices available yet ({unpriced} fixture(s) returned none). "
              f"Leaving the existing file untouched.")
        print(client.report())
        return 0

    payload = {
        "_note": ("Book prices for the upcoming slate, de-vigged. Generated by "
                  "scripts/sync_nfl_odds.py -- do not hand-edit. Probabilities "
                  "here are FAIR (vig removed); raw_* are as quoted."),
        "updated": datetime.now(timezone.utc).isoformat(),
        "preferred_books": list(odds.PREFERRED_BOOKS),
        "games": priced,
        "unpriced": unpriced,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    books = sorted({v["book"] for v in priced.values() if v.get("book")})
    print(f"priced {len(priced)} fixture(s) from {books}; {unpriced} unpriced")
    print(client.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
