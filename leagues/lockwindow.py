"""How far before kickoff a pick may freeze, ADAPTED to how often the locker runs.

THE PROBLEM THIS SOLVES IS SELECTION BIAS, not inconvenience.

A pick enters the record only if a locker run happens inside LOCK_WINDOW_HOURS of
kickoff. With a fixed 2-hour window and a scheduler that fires reliably, that is
fine. With a scheduler firing three or four times a day it is not: on 2026-08-28
the runs were at 15:23 and 21:18, and every fixture kicking off between 18:30 and
19:30 -- Bayern Munich v Stuttgart, Lille v Paris SG, Alaves v Villarreal, Crystal
Palace v Manchester City -- fell in the gap. All four were shown on the board with
a pick, all four were played, and all four are absent from the record entirely.
Not graded, not void, just gone.

That is not a hole in coverage, it is a BIASED SAMPLE. The record ends up
describing the fixtures that happened to kick off near a workflow run, which is a
property of GitHub's scheduler and nothing to do with football. A 5-4 record that
quietly omits four played games is a worse number than a 5-4 record that says so.

THE FIX. Each locker run records a heartbeat. The next run widens its window to
cover the time that actually elapsed since the last one, so a fixture cannot fall
between two runs. If runs are frequent the window stays at its 2-hour floor and
nothing changes. If the scheduler drops runs for six hours, the next run freezes
what would otherwise have been missed.

WHAT THIS TRADES. A pick frozen six hours out has seen less team news than one
frozen at two. It is still frozen strictly before kickoff, so it is still honest --
`LATE_LOCK_HOURS` stays at 0.0 and nothing here relaxes it. The cost is measured
and small (the config note's own example is a confirmed XI moving Arsenal from
77.4% to 77.1%), and the alternative is losing the fixture from the record
altogether. Freezing early is a worse pick; freezing never is a worse RECORD.

THE CAP EXISTS because the widening must not become unbounded. After a multi-day
outage, `MAX_WINDOW_HOURS` stops the first run back from freezing an entire
matchweek at once on days-stale numbers -- at some point the honest answer is that
those fixtures have no pick, and `unrecorded` on the board says so.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from leagues.config import LOCK_WINDOW_HOURS

ROOT = Path(__file__).resolve().parent.parent
HEARTBEAT = ROOT / "data-raw" / "leagues" / "locker_heartbeat.json"

# Never freeze more than this far out, however long the locker has been down.
MAX_WINDOW_HOURS = 12.0


def _now(now=None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, str):
        now = datetime.fromisoformat(now)
    now = getattr(now, "to_pydatetime", lambda: now)()
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def last_run(path=None):
    """When a locker last ran, or None if it never has / the file is unreadable."""
    path = Path(path or HEARTBEAT)
    try:
        stamp = json.loads(path.read_text(encoding="utf-8")).get("last_run")
        return datetime.fromisoformat(stamp) if stamp else None
    except Exception:
        return None                       # a missing heartbeat is just a floor


def window(now=None, path=None) -> float:
    """Hours before kickoff within which a pick may freeze, right now.

    Never below LOCK_WINDOW_HOURS (the floor the product is designed around) and
    never above MAX_WINDOW_HOURS (so an outage cannot freeze a whole matchweek).
    """
    previous = last_run(path)
    if previous is None:
        return LOCK_WINDOW_HOURS
    elapsed = (_now(now) - previous).total_seconds() / 3600.0
    if elapsed <= 0:
        return LOCK_WINDOW_HOURS          # clock skew: fall back to the floor
    return min(MAX_WINDOW_HOURS, max(LOCK_WINDOW_HOURS, elapsed))


def beat(now=None, path=None) -> None:
    """Record that a locker has just run.

    Written by every path that locks -- the fast locker AND publish -- because
    what matters is the gap between runs that could have frozen something, not
    the gap between runs of any one script.
    """
    path = Path(path or HEARTBEAT)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_run": _now(now).isoformat(),
        "_note": ("Written by every locking run. The NEXT run widens its lock "
                  "window to cover the gap since this timestamp, so a fixture "
                  "cannot fall between two runs and vanish from the record. "
                  "Generated -- do not hand-edit."),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
