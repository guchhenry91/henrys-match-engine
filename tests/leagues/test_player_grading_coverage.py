"""The shot feed must not be allowed to grade fixtures it has never seen.

On 2026-08-23 the live board published player picks at 0 correct / 18 wrong.
Not one of those losses was real. Understat had published nothing for the
2026-27 season -- its newest row was 2026-05-24, the previous May -- while 26
fixtures had been played. The frame was nowhere near empty (26,401 rows for the
Premier League alone), so the "is the feed available" guard passed, every
per-player lookup missed, and every miss was graded a loss.

grade_prop reads a missing row as WRONG deliberately: the player either did not
play or played without shooting, and the feed cannot separate those, so we take
the harsher reading. That is sound where the feed HAS the match. Where it does
not, wrong is not a harsh reading of the evidence -- it is an answer invented in
the absence of any, written into an append-only record.
"""
import pandas as pd

from leagues import picks
from leagues.publish import _covered_sides


def _frame(rows):
    return pd.DataFrame(rows, columns=["team", "day", "player",
                                       "goals", "shots", "sot"])


def test_a_frame_full_of_last_season_covers_nothing_this_season():
    """The exact 2026-08-23 shape: plenty of rows, none for the fixture."""
    import datetime as dt
    last_season = _frame([
        ("Arsenal", dt.date(2026, 5, 24), "Saka", 1, 4, 2),
        ("Chelsea", dt.date(2026, 5, 24), "Palmer", 0, 3, 1),
    ])
    covered = _covered_sides(last_season)
    assert ("Arsenal", dt.date(2026, 8, 21)) not in covered
    assert ("Arsenal", dt.date(2026, 5, 24)) in covered


def test_empty_and_none_cover_nothing():
    assert _covered_sides(None) == set()
    assert _covered_sides(_frame([])) == set()


def test_coverage_is_per_side_not_per_league():
    """One club's data landing must not license grading another club's picks."""
    import datetime as dt
    day = dt.date(2026, 8, 22)
    covered = _covered_sides(_frame([("Tottenham", day, "Richarlison", 0, 2, 1)]))
    assert ("Tottenham", day) in covered
    assert ("Brentford", day) not in covered


def test_grade_prop_still_reads_a_missing_row_as_wrong():
    """The harsh reading is correct WHERE THE FEED HAS THE MATCH, and stays."""
    entry = {"market": "goal", "player": "Thiago", "team": "Brentford"}
    assert picks.grade_prop(entry, None)["graded"] == "wrong"
    assert picks.grade_prop(entry, {"goals": 1})["graded"] == "correct"


def test_grade_prop_still_voids_a_tainted_pick():
    entry = {"market": "goal", "player": "X", "team": "Y", "tainted": True}
    assert picks.grade_prop(entry, None)["graded"] == "void"
