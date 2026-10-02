"""Every newly frozen pick says which model made it (tracking/manifest.py)."""
from datetime import datetime, timezone

from tracking import manifest


def test_stamps_only_fresh_unstamped_picks(monkeypatch):
    monkeypatch.setattr(manifest, "commit", lambda: "abc1234567")
    monkeypatch.setattr(manifest, "report_hash", lambda sport: "r" * 12)
    now = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)
    fresh = {"locked_at": "2026-10-02T11:00:00+00:00", "trust_w": 0.0}
    old = {"locked_at": "2026-09-30T11:00:00+00:00"}
    done = {"locked_at": "2026-10-02T11:30:00+00:00", "model_version": {"commit": "x"}}
    assert manifest.stamp([fresh, old, done, {"no": "lock"}], "nfl", now=now) == 1
    assert fresh["model_version"] == {"commit": "abc1234567", "report": "rrrrrrrrrrrr", "trust": 0.0}
    assert "model_version" not in old                      # never back-dated
    assert done["model_version"] == {"commit": "x"}        # never rewritten
