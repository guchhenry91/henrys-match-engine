"""An unreachable Understat must not eat the soccer publish's 20 minutes."""
import urllib.error

from scripts import sync_understat_matches as sync


def test_one_timeout_skips_the_other_leagues(monkeypatch, tmp_path):
    calls = []

    def down(code, year, base):
        calls.append(code)
        raise urllib.error.URLError("timed out")
    monkeypatch.setattr(sync, "sync_league", down)
    monkeypatch.setattr(sync, "UPCOMING", tmp_path / "upcoming.json")
    assert sync.main() == 0
    assert len(calls) == 1                       # the first failure stops the run


def test_a_bad_league_reply_does_not_stop_the_others(monkeypatch, tmp_path):
    calls = []

    def flaky(code, year, base):
        calls.append(code)
        if len(calls) == 1:
            raise KeyError("dates")
        return 0, []
    monkeypatch.setattr(sync, "sync_league", flaky)
    monkeypatch.setattr(sync, "UPCOMING", tmp_path / "upcoming.json")
    sync.main()
    assert len(calls) == len(sync.LEAGUES)


def test_live_requests_are_short(monkeypatch):
    seen = {}
    monkeypatch.setattr(sync, "_get", lambda url, tries, timeout: seen.update(tries=tries, timeout=timeout))
    sync.get("https://understat.com/getLeagueData/EPL/2026")
    assert seen == {"tries": 2, "timeout": 20}
