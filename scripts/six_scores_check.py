"""Validate the 6 Scores selection against our own fixture list.

THE FAILURE THIS EXISTS FOR. bet365 is unreachable -- their site 403s an
automated fetch and is blocked by browsing policy -- so the six have to be read
off aggregator sites that republish them. Those sites go DORMANT between
matchweeks and keep serving the previous round: on 2026-08-18 everytip and
aceodds were both still showing 24 May, the last round of the previous season.

An agent reading such a page finds six real Premier League fixtures, correctly
formatted, and entirely wrong. Nothing about them looks stale. So the selection
is checked against our own fixture data rather than trusted:

  * exactly six, no duplicates
  * every fixture EXISTS in data/leagues/pl.json
  * every one is UNPLAYED and in the FUTURE
  * all six fall within the next MAX_DAYS -- a page serving last season's round
    fails here even if those clubs meet again later

Run after writing six_scores.json. Non-zero exit means do not publish.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data-raw" / "leagues" / "six_scores.json"
PL = ROOT / "data" / "leagues" / "pl.json"
MAX_DAYS = 9
EXPECTED = 6


def check(now=None) -> list[str]:
    now = pd.Timestamp(now or pd.Timestamp.now("UTC"))
    problems = []
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    fixtures = sel.get("fixtures") or []
    if not fixtures:
        return ["no selection set"]
    if len(fixtures) != EXPECTED:
        problems.append(f"expected {EXPECTED} fixtures, got {len(fixtures)}")
    if len(set(fixtures)) != len(fixtures):
        problems.append("duplicate fixtures in the selection")
    pl = json.loads(PL.read_text(encoding="utf-8"))
    by_key = {f"{m['home']}|{m['away']}": m for m in pl.get("matches", [])}
    for key in fixtures:
        m = by_key.get(key)
        if not m:
            problems.append(f"{key!r} is not an upcoming Premier League fixture")
            continue
        ko = pd.Timestamp(m["date"])
        if m.get("result") is not None:
            problems.append(f"{key} has already been played")
        elif ko <= now:
            problems.append(f"{key} already kicked off ({ko:%Y-%m-%d %H:%M}Z)")
        elif (ko - now).days > MAX_DAYS:
            problems.append(f"{key} is {(ko - now).days} days away (> {MAX_DAYS}) "
                            f"-- this looks like a stale page")
    return problems


def main():
    problems = check()
    if not problems:
        sel = json.loads(SELECTION.read_text(encoding="utf-8"))
        print(f"6 Scores selection OK -- matchweek {sel.get('matchweek')}, "
              f"{len(sel['fixtures'])} fixtures, verified {sel.get('_verified_on')}")
        return 0
    if problems == ["no selection set"]:
        print("note: no selection set -- the board publishes empty and says so")
        return 0
    print("6 SCORES SELECTION REJECTED:")
    for p in problems:
        print(f"  - {p}")
    print("\nDo NOT publish. Leave `fixtures` empty and report that the week's "
          "six could not be confirmed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
