"""Print unplayed matches kicking off within the next N hours (US Eastern).

Used by the pre-match scheduled task to decide whether a last-minute
team-news refresh is warranted. Usage: python next_matches.py [hours=4]
Prints a JSON list (empty list = nothing imminent).
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:                       # no tzdata on Windows -> fixed EDT
    ET = timezone(timedelta(hours=-4))  # WC 2026 (Jun-Jul) is daylight time

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(ROOT, "data-raw", name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    sched = load("schedule.json")
    try:
        results = load("results.json") or {}
    except Exception:
        results = {}
    now = datetime.now(ET)
    out = []
    for m in sched["matches"]:
        if str(m["id"]) in results:
            continue
        t = m.get("time_et") or "12:00"
        try:
            ko = datetime.strptime(f"{m['date']} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        except ValueError:
            continue
        delta = (ko - now).total_seconds() / 3600.0
        if 0 <= delta <= hours:
            out.append({
                "id": m["id"], "home": m["home"], "away": m["away"],
                "group": m["group"], "date": m["date"], "time_et": t,
                "hours_to_kickoff": round(delta, 1),
            })
    out.sort(key=lambda x: x["hours_to_kickoff"])
    print(json.dumps(out))


if __name__ == "__main__":
    main()
