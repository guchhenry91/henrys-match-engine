"""NFL grading between full refreshes (scripts/grade_nfl.py).

Sunday's games finished by 04:00 UTC and nflverse had every score by 04:47, but
the NFL record was still ungraded at 10:22 because grading only happened inside
the twice-daily NFL refresh. This rides the frequent lock job instead.
"""
import json

import pandas as pd

from nfl import picks
from scripts import grade_nfl

NOW = pd.Timestamp("2026-09-14T10:00:00Z")


def _entry(kickoff, graded=None, void=False):
    return {"kickoff": kickoff, "graded": graded, "void": void}


def test_a_finished_ungraded_game_is_due():
    log = {"games": {"G1": _entry("2026-09-13T17:00:00+00:00")}, "props": {}}
    assert grade_nfl.due(log, NOW) == ["G1"]


def test_a_game_still_in_play_is_not_due():
    log = {"games": {"G1": _entry("2026-09-14T08:00:00+00:00")}, "props": {}}
    assert grade_nfl.due(log, NOW) == []


def test_graded_and_void_picks_are_never_rechecked():
    log = {"games": {}, "props": {
        "P1": _entry("2026-09-13T17:00:00+00:00", graded="correct"),
        "P2": _entry("2026-09-13T17:00:00+00:00", void=True),
        "_meta": {"x": 1}}}
    assert grade_nfl.due(log, NOW) == []


def test_nothing_due_means_no_download_and_no_write(tmp_path, monkeypatch):
    """The common case must cost nothing: no network, no board rewrite."""
    board = tmp_path / "board.json"
    board.write_text(json.dumps({"record": {"x": 1}}), encoding="utf-8")
    monkeypatch.setattr(grade_nfl, "BOARD", board)
    monkeypatch.setattr(picks, "PICKS_LOG", tmp_path / "log.json")
    monkeypatch.setattr(picks, "freeze_and_grade",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    assert grade_nfl.main(now=NOW) == 0
    assert json.loads(board.read_text(encoding="utf-8")) == {"record": {"x": 1}}


def test_a_changed_record_rewrites_the_board(tmp_path, monkeypatch):
    board = tmp_path / "board.json"
    board.write_text(json.dumps({"record": {"old": True}, "settled": []}), encoding="utf-8")
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps({"games": {"G1": _entry("2026-09-13T17:00:00+00:00")},
                                    "props": {}}), encoding="utf-8")
    monkeypatch.setattr(grade_nfl, "BOARD", board)
    monkeypatch.setattr(picks, "PICKS_LOG", log_path)
    new = {"team_winner": {"correct": 1, "wrong": 0},
           "props": {"correct": 0, "wrong": 0, "pending": 0}}
    monkeypatch.setattr(picks, "freeze_and_grade", lambda payload, now=None: new)
    monkeypatch.setattr(picks, "settled", lambda log: [{"graded": "correct"}])
    grade_nfl.main(now=NOW)
    written = json.loads(board.read_text(encoding="utf-8"))
    assert written["record"] == new and written["settled"] == [{"graded": "correct"}]
