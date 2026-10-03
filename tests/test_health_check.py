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


def test_an_unchanged_board_whose_refresh_ran_is_not_stale(monkeypatch):
    """2026-10-03: the off-season NBA board was 31h 'old' because it had not changed,
    while its refresh succeeded every few hours -- a false FAIL."""
    monkeypatch.setattr(hc, "_read", lambda rel: {"updated": "2026-10-01T12:00:00+00:00"})
    monkeypatch.setitem(hc.LAST_OK, "nba.yml", 2.0)
    r = hc.Report()
    hc.board_age(r, datetime(2026, 10, 2, 19, tzinfo=timezone.utc), "NBA", "x", 14, 30)
    assert r.checks[0]["status"] == "ok"
    monkeypatch.setitem(hc.LAST_OK, "nba.yml", 40.0)       # refresh itself stopped
    r2 = hc.Report()
    hc.board_age(r2, datetime(2026, 10, 2, 19, tzinfo=timezone.utc), "NBA", "x", 14, 30)
    assert r2.checks[0]["status"] == "fail"


def test_on_cloudflare_the_pipeline_is_read_from_the_job_queue(monkeypatch):
    """Since the move off GitHub Actions the jobs run on Cloudflare; reading `gh`
    there would report every pipeline unreadable and every quiet board stale."""
    import io
    import json as _json
    rows = [{"job": "nba", "status": "running", "enqueued_at": 1759500000},
            {"job": "nba", "status": "done", "enqueued_at": 1759490000},
            {"job": "nba", "status": "failed", "enqueued_at": 1759480000}]
    seen = {}

    def fake_open(req, timeout=0):
        seen["url"] = req.full_url
        return io.BytesIO(_json.dumps(rows).encode())

    monkeypatch.setenv("STATE_URL", "https://example.invalid")
    monkeypatch.setenv("STATE_TOKEN", "t")
    monkeypatch.setattr(hc.urllib.request, "urlopen", fake_open)
    runs = hc._runs("nba.yml")
    assert "job=nba" in seen["url"] and "ns=live" in seen["url"]
    assert [r["conclusion"] for r in runs] == ["success", "failure"]   # running is skipped
