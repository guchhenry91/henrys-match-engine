"""The lock window must survive how the scheduler actually behaves.

Measured over 109 scheduled runs on 2026-08-26: a run starts a median 8 minutes
after its cron slot (90th percentile 14), the job needs about 10 more to reach
the lock, and 29% of scheduled runs fail outright. A one-hour window assumed
"four chances" and in practice had two or three, sometimes one -- and on
2026-08-26 GitHub fired nothing at all for 100 minutes before a 19:00 kickoff.

The cost was not theoretical: four La Liga fixtures locked 1, 12, 20 and 29
minutes AFTER kickoff, which taints a pick and voids it. Eight player picks and
thirty-seven parlays went with them.
"""
from leagues import publish, picks

CRON_MINUTES = (0, 15, 30, 45)
MEDIAN_DELAY = 8.1        # minutes, measured
P90_DELAY = 14.0
JOB_LATENCY = 10.0        # minutes from run start to the lock step


def _lock_offsets(window_hours, delay):
    """Minutes-before-kickoff at which each usable run would actually lock."""
    offsets = []
    for slot_before in range(0, int(window_hours * 60) + 1, 15):
        actual = slot_before - delay - JOB_LATENCY
        if actual > 0:
            offsets.append(actual)
    return offsets


def test_the_shipped_window_survives_typical_delay():
    chances = _lock_offsets(publish.LOCK_WINDOW_HOURS, MEDIAN_DELAY)
    assert len(chances) >= 5, (
        f"only {len(chances)} lock chances at median delay; one bad run and a "
        f"fixture voids")


def test_the_shipped_window_survives_the_90th_percentile_delay():
    chances = _lock_offsets(publish.LOCK_WINDOW_HOURS, P90_DELAY)
    assert len(chances) >= 4, f"only {len(chances)} chances at p90 delay"


def test_a_one_hour_window_would_not_have_been_enough():
    """The regression this replaced. Kept so nobody narrows it back by feel."""
    assert len(_lock_offsets(1.0, MEDIAN_DELAY)) <= 3


def test_late_locking_still_taints():
    """Widening the window is the fix; tolerating a late lock is NOT. A pick made
    after kickoff cannot be shown to be a pre-match pick, and the record's whole
    value is that every entry demonstrably was."""
    assert picks.LATE_LOCK_HOURS == 0.0
