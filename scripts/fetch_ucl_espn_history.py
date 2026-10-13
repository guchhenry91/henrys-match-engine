"""ESPN player shots (blocked included) for the past matches in player_history.json --
the data the 2+ shots test needs (ucl/espn.py). Free; ~one request per match day and
one per match; only matches not already saved."""
import json
import sys
from pathlib import Path

from ucl import espn
from ucl.props_names import to_understat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data-raw" / "ucl" / "player_history.json"
OUT = ROOT / "data-raw" / "ucl" / "espn_history.json"


def main() -> int:
    src = json.loads(SRC.read_text(encoding="utf-8"))["fixtures"]
    try:
        store = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        store = {"fixtures": {}}
    have = store.setdefault("fixtures", {})
    days, got, missed = {}, 0, []
    for fid, f in sorted(src.items(), key=lambda kv: kv[1]["date"]):
        if fid in have:
            continue
        day = str(f["date"])[:10]
        try:
            if day not in days:
                days[day] = espn.events_on(day)
            eid = espn.find_event(days[day], (f["home"], to_understat(f["home"])),
                                  (f["away"], to_understat(f["away"])))
            if not eid:
                missed.append(f"{day} {f['home']} v {f['away']}")
                continue
            have[fid] = {"event": eid, "date": day, "home": f["home"], "away": f["away"],
                         "players": espn.players(eid)}
            got += 1
        except Exception as exc:                          # noqa: BLE001
            print(f"  {day} {f['home']} v {f['away']}: {exc}")
    OUT.write_text(json.dumps(store, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {got}; {len(have)} in the file; unmatched {len(missed)}: {missed[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
