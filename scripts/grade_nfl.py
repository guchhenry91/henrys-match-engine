"""Grade finished NFL games between full refreshes. No model, no API quota.

WHY. NFL picks were graded only inside nfl.yml, scheduled at 09:00 and 16:00 UTC
and in practice starting hours late (the 09:00 slot on 2026-09-13 began at
13:45). Sunday's early games finish around 20:00 UTC, and nflverse had every
score and player line by 04:47 the next morning -- yet the record still read
"Thursday and Friday only" at 10:22 on Monday, because no run had happened since
the games ended.

Grading needs no model: the pick was frozen before kickoff and the result is a
fact. So this rides the fast `lock` job in leagues.yml, which fires many times a
day, and does exactly what nfl.publish does to the record -- freeze_and_grade and
settled -- against the board already published. Predictions are untouched.

CHEAP WHEN IDLE. It downloads nothing unless a frozen pick's game ended long
enough ago to be graded and is still ungraded; most runs exit in milliseconds.
And it rewrites the board only when the record actually changed, so the job
deploys only when a reader would see something new.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "data" / "nfl" / "board.json"

# A game is worth checking once it has had time to finish (NFL games run about
# three and a quarter hours, overtime included).
GAME_LENGTH_HOURS = 3.5


def due(log: dict, now) -> list:
    """Frozen, ungraded picks whose game should be over by now."""
    out = []
    for part in ("games", "props"):
        for key, entry in (log.get(part) or {}).items():
            if key.startswith("_") or not isinstance(entry, dict):
                continue
            if entry.get("graded") is not None or entry.get("void"):
                continue
            kickoff = pd.Timestamp(entry.get("kickoff") or "NaT")
            if pd.isna(kickoff):
                continue
            if kickoff.tzinfo is None:
                kickoff = kickoff.tz_localize("UTC")
            if (now - kickoff).total_seconds() / 3600.0 >= GAME_LENGTH_HOURS:
                out.append(key)
    return out


def main(now=None) -> int:
    from nfl import picks

    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now("UTC")
    if not BOARD.exists():
        print("no NFL board published; nothing to grade")
        return 0
    log = picks.core.load_log(picks.PICKS_LOG)
    waiting = due(log, now)
    payload = json.loads(BOARD.read_text(encoding="utf-8"))
    before = (payload.get("record"), payload.get("settled"))
    if waiting:
        payload["record"] = picks.freeze_and_grade(payload, now=now)
    else:
        # NO NETWORK, but still refresh the record from the log. A pick frozen by
        # this job's lock step never reached the board otherwise: DEN @ KC was
        # locked at 23:52 on 2026-09-14 and the board still read 15 team-winner
        # picks, 0 pending, until a full refresh -- it looked as if the game had
        # no pick at all.
        print("NFL: no finished game awaiting a grade")
        payload["record"] = picks.record(log)
    payload["settled"] = picks.settled(picks.core.load_log(picks.PICKS_LOG))
    rec = payload["record"]
    print(f"NFL: {len(waiting)} finished pick(s) checked -> team winner "
          f"{rec['team_winner']['correct']}-{rec['team_winner']['wrong']}, props "
          f"{rec['props']['correct']}-{rec['props']['wrong']}, "
          f"{rec['props']['pending']} props still pending")
    if (payload["record"], payload["settled"]) == before:
        print("NFL: record unchanged (feed has not filed those games yet)")
        return 0
    tmp = BOARD.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(BOARD)
    print(f"wrote {BOARD}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:        # never let NFL grading fail the soccer lock job
        print(f"WARNING: NFL grading skipped ({type(exc).__name__}: {exc})")
        sys.exit(0)
