"""The self-check must catch the failures that actually happened."""
from datetime import datetime, timezone

from scripts import health_check as hc


def test_it_runs_on_the_published_data():
    rep = hc.run(check_workflows=False)
    assert rep["status"] in ("ok", "warn", "fail") and rep["checks"]


def test_it_catches_props_priced_on_last_seasons_data(monkeypatch):
    """2026-10-01: the NFL board priced week 4 of 2026 on 2025 week 18."""
    board = {"updated": "2026-10-01T12:00:00+00:00", "season": 2026, "week": 4,
             "player_data_through": {"season": 2025, "week": 18}}
    monkeypatch.setattr(hc, "_read", lambda rel: board if rel == "data/nfl/board.json" else {})
    r = hc.Report()
    hc.nfl(r, datetime(2026, 10, 1, 13, tzinfo=timezone.utc))
    assert any(c["check"] == "player data" and c["status"] == "fail" for c in r.checks)


def test_a_board_that_stopped_updating_fails(monkeypatch):
    monkeypatch.setattr(hc, "_read", lambda rel: {"updated": "2026-09-29T00:00:00+00:00"})
    r = hc.Report()
    hc.board_age(r, datetime(2026, 10, 1, tzinfo=timezone.utc), "NFL", "x", 14, 30)
    assert r.checks[0]["status"] == "fail"
