"""record_history: the zero-row bug and its one-off repair.

CI ran publish.main() on empty stub boards inside the test suite, between the
real publish and the commit, with PICKS_DIR still pointing at the real
data-raw/leagues -- so every automated refresh committed a 0-0 row. The test is
now isolated (test_publish_multi.py); this covers the rebuild of the lost rows.
"""
import pathlib
import re

from scripts import backfill_record_history as bf

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _zero(day):
    z = {"correct": 0, "wrong": 0, "total": 0}
    return {"date": day, "best": dict(z), "players": dict(z),
            "stated_pct": None, "actual_pct": None}


BEST = [{"date": "2026-09-05T16:30:00+00:00", "graded": "correct", "p_pick": 0.7},
        {"date": "2026-09-06T14:00:00+00:00", "graded": "wrong", "p_pick": 0.66},
        {"date": "2026-09-06T14:00:00+00:00", "graded": "void", "p_pick": 0.68}]
PLAYERS = [{"date": "2026-09-05T16:30:00+00:00", "graded": "correct", "p_pick": 0.6}]


def test_a_zero_row_is_rebuilt_from_picks_played_before_that_day():
    rows, fixed = bf.rebuild([_zero("2026-09-07")], BEST, PLAYERS)
    row = rows[0]
    assert fixed == 1 and row["reconstructed"] is True
    assert row["best"] == {"correct": 1, "wrong": 1, "total": 2}      # void excluded
    assert row["players"] == {"correct": 1, "wrong": 0, "total": 1}
    assert row["stated_pct"] == 68.0 and row["actual_pct"] == 50.0


def test_a_same_day_result_is_not_counted_yet():
    """A publish on the 6th could not have seen games played that evening."""
    rows, _ = bf.rebuild([_zero("2026-09-06")], BEST, PLAYERS)
    assert rows[0]["best"]["total"] == 1


def test_a_real_row_is_never_touched():
    real = {"date": "2026-09-07", "best": {"correct": 9, "wrong": 1, "total": 10},
            "players": {"correct": 3, "wrong": 1, "total": 4},
            "stated_pct": 70.0, "actual_pct": 90.0}
    rows, fixed = bf.rebuild([real], BEST, PLAYERS)
    assert fixed == 0 and rows[0] is real


def test_a_genuinely_empty_day_stays_zero_and_unmarked():
    rows, fixed = bf.rebuild([_zero("2026-08-01")], BEST, PLAYERS)
    assert fixed == 0 and "reconstructed" not in rows[0]


def test_the_grades_tab_shows_the_calibration_watch():
    body = re.search(r"function calibWatch\(\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert "WATCH_MIN" in body and "2*se" in body
    assert "latestSettled()+calibWatch()" in HTML
