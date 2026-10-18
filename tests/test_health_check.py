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


def test_mlb_board_is_not_stale_across_its_normal_overnight_gap(monkeypatch):
    from datetime import datetime, timezone
    now = datetime(2026, 10, 10, 12, 33, tzinfo=timezone.utc)
    boards = {"data/mlb/board.json": {"status": "live", "updated": "2026-10-09T22:02:00+00:00"},
              "data-raw/mlb/picks_log.json": {}}
    monkeypatch.setattr(hc, "_read", lambda rel: boards.get(rel))
    monkeypatch.setitem(hc.LAST_OK, "mlb.yml", 14.5)
    r = hc.Report()
    hc.line_sport(r, now, "MLB", "mlb", "tipoff")
    assert [c["status"] for c in r.checks if c["check"] == "board"] == ["ok"]


def test_a_board_not_built_by_the_rebuilt_model_fails(tmp_path, monkeypatch):
    import json
    from scripts import health_check as hc
    boards = {"data/nfl/board.json": {"updated": "x"},
              "data/nba/board.json": {"engine": {"name": "forecast-v1"}}}
    monkeypatch.setattr(hc, "_read", lambda rel: boards.get(rel))
    r = hc.Report()
    hc.engines(r)
    got = {c["sport"]: c["status"] for c in r.checks}
    assert got == {"NFL": "fail", "NBA": "ok"}


NOW = datetime(2026, 10, 18, 15, tzinfo=timezone.utc)


def _board(files):
    return lambda rel: files.get(rel)


def test_an_empty_nfl_props_board_on_game_day_is_flagged(monkeypatch):
    """2026-10-18: NFL game day, every market priced, nothing published -- the owner
    found it before the pipeline did."""
    board = {"games": [{"kickoff": "2026-10-18T17:00:00+00:00"}],
             "props_status": {"forecast": None, "markets": {
                 "receiving_yards": {"slate": 40, "with_book_price": 40, "in_forecast": 38, "priced": 38}}},
             "props": {"receiving_yards": {"released": True, "picks": []}}}
    monkeypatch.setattr(hc, "_read", _board({"data/nfl/board.json": board}))
    r = hc.Report()
    hc.props_output(r, NOW)
    c = [c for c in r.checks if c["sport"] == "NFL"][0]
    assert c["status"] == "warn" and "0 picks" in c["message"]


def test_a_broken_nfl_forecast_join_fails(monkeypatch):
    board = {"games": [{"kickoff": "2026-10-18T17:00:00+00:00"}],
             "props_status": {"forecast": None, "markets": {
                 "passing_yards": {"slate": 20, "with_book_price": 20, "in_forecast": 0, "priced": 0}}},
             "props": {}}
    monkeypatch.setattr(hc, "_read", _board({"data/nfl/board.json": board}))
    r = hc.Report()
    hc.props_output(r, NOW)
    assert r.checks[0]["status"] == "fail"


def test_ucl_props_missing_before_a_matchday_fails(monkeypatch):
    ucl = {"matches": [{"kickoff": "2026-10-19T19:00:00+00:00"}], "props": None}
    monkeypatch.setattr(hc, "_read", _board({"data/ucl/board.json": ucl}))
    r = hc.Report()
    hc.props_output(r, NOW)
    assert [c["status"] for c in r.checks if c["sport"] == "UCL"] == ["fail"]


def test_no_games_means_no_props_complaint(monkeypatch):
    monkeypatch.setattr(hc, "_read", _board({"data/nfl/board.json": {"games": [], "props": {}}}))
    r = hc.Report()
    hc.props_output(r, NOW)
    assert r.checks == []


def test_a_pick_frozen_after_kickoff_fails(monkeypatch):
    log = {"games": {"g": {"kickoff": "2026-10-18T13:00:00+00:00",
                           "locked_at": "2026-10-18T13:20:00+00:00"}},
           "props": {"p": {"kickoff": "2026-10-18T13:00:00+00:00",
                           "locked_at": "2026-10-18T12:00:00+00:00"}}}
    monkeypatch.setattr(hc, "_read", _board({"data-raw/nfl/picks_log.json": log}))
    r = hc.Report()
    hc.late_locks(r, NOW)
    assert r.checks[0]["status"] == "fail" and r.checks[0]["message"].startswith("1 pick")
