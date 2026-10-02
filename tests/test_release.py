"""Official or tracked, frozen with each pick (tracking/release.py)."""
from datetime import datetime, timezone

from tracking import release


def test_official_only_for_value_in_an_earned_market():
    now = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)
    fresh = "2026-10-02T11:00:00+00:00"
    rows = [{"market": "hits", "locked_at": fresh, "value": {"value": True}},
            {"market": "hits", "locked_at": fresh, "value": {"value": False}},
            {"market": "hr", "locked_at": fresh, "value": {"value": True}},
            {"market": "hits", "locked_at": "2026-09-20T11:00:00+00:00", "value": {"value": True}}]
    n = release.mark(rows, "mlb", lambda e: e["market"],
                     statuses={"hits": "promising", "hr": "testing"}, now=now)
    assert n == 1
    assert [r.get("release") for r in rows] == ["official", "tracked", "tracked", None]
    assert rows[2]["release_status_seen"] == "testing"
