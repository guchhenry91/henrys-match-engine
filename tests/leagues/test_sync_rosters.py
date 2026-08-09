import io
import json

from scripts import sync_rosters


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_roster_request_retries_transient_failure(monkeypatch):
    calls = []

    def open_url(_request, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise OSError("temporary TLS failure")
        return _Response(b'{"ok": true}')

    monkeypatch.setattr(sync_rosters.urllib.request, "urlopen", open_url)
    sleeps = []
    assert sync_rosters.get_json("https://example.test", attempts=3,
                                 sleeper=sleeps.append) == {"ok": True}
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_failed_refresh_retains_complete_verified_snapshot(tmp_path, monkeypatch):
    out = tmp_path / "rosters.json"
    old = {"_verified_at": "2026-07-21T10:00:00+00:00"}
    old.update({key: {"Club": {"players": []}} for key in sync_rosters.LEAGUES})
    out.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(sync_rosters, "OUT", out)
    monkeypatch.setattr(sync_rosters, "fetch_league",
                        lambda *_args: (_ for _ in ()).throw(OSError("TLS")))

    assert sync_rosters.main() == 0
    assert json.loads(out.read_text(encoding="utf-8")) == old


def test_force_roster_leagues_catches_up_a_league_outside_its_rotation_slot(
        tmp_path, monkeypatch):
    """The daily pair rotation only refreshes two of four leagues at a time, so
    right after enabling a new API-Football key two leagues are stuck on the old
    ESPN fallback for up to 48h. FORCE_ROSTER_LEAGUES exists to catch those up
    immediately on a manual workflow_dispatch run, without waiting on the clock
    or touching the leagues the rotation already picked."""
    # Seed every league as already fresh so only the forced pair is due --
    # otherwise whichever pair the real day's date naturally rotates to would
    # also be fetched, making the test's outcome depend on today's date.
    out = tmp_path / "rosters.json"
    now_iso = sync_rosters.datetime.now(sync_rosters.timezone.utc).isoformat()
    out.write_text(json.dumps({
        "_league_verified_at": {key: now_iso for key in sync_rosters.LEAGUES},
    }), encoding="utf-8")
    monkeypatch.setattr(sync_rosters, "OUT", out)
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setenv("FORCE_ROSTER_LEAGUES", "pl,bundesliga")

    fetched = []

    def fake_fetch(_client, key):
        fetched.append(key)
        return {"Club": {"source": "api-football:team:1", "players": []}}

    monkeypatch.setattr(sync_rosters, "fetch_api_league", fake_fetch)
    monkeypatch.setattr(sync_rosters, "Client",
                        lambda **_kw: type("FakeClient", (), {"used": 0})())

    assert sync_rosters.main() == 0
    assert set(fetched) == {"PL", "BUNDESLIGA"}
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["_source"] == "API-Football current squad feeds"
    assert "PL" in written["_league_verified_at"]
    assert "BUNDESLIGA" in written["_league_verified_at"]


def test_one_leagues_api_football_failure_does_not_discard_a_sibling_success(
        tmp_path, monkeypatch):
    """Observed in production: PL fetched fine, then Bundesliga raised on an
    unmapped club name mid-loop, and the single try/except around the whole
    batch fell back to ESPN for ALL leagues -- discarding the already-good PL
    fetch too. Each league must succeed or fail independently."""
    out = tmp_path / "rosters.json"
    monkeypatch.setattr(sync_rosters, "OUT", out)
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setenv("FORCE_ROSTER_LEAGUES", "pl,bundesliga")

    def fake_fetch(_client, key):
        if key == "BUNDESLIGA":
            raise ValueError("'FSV Mainz 05' is not mapped")
        return {"Arsenal": {"source": "api-football:team:1", "players": []}}

    espn_calls = []
    monkeypatch.setattr(sync_rosters, "fetch_api_league", fake_fetch)
    monkeypatch.setattr(sync_rosters, "fetch_league",
                        lambda key, slug: espn_calls.append(key) or
                        {"SomeClub": {"source": "espn", "players": []}})
    monkeypatch.setattr(sync_rosters, "Client",
                        lambda **_kw: type("FakeClient", (), {"used": 0})())

    assert sync_rosters.main() == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["PL"] == {"Arsenal": {"source": "api-football:team:1", "players": []}}
    assert "PL" in written["_league_verified_at"]
    assert espn_calls == ["BUNDESLIGA"]
    assert written["BUNDESLIGA"] == {"SomeClub": {"source": "espn", "players": []}}
    assert "BUNDESLIGA" not in written["_league_verified_at"]
    assert "ESPN fallback for: BUNDESLIGA" in written["_source"]


class _FakeApiClient:
    """Records calls so a test can assert players/squads is never hit for a
    league that turns out to have an unmapped team."""

    def __init__(self, teams, squads):
        self._teams = teams
        self._squads = squads
        self.calls = []

    def get(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        if endpoint == "teams":
            return self._teams
        if endpoint == "players/squads":
            return self._squads[kwargs["team"]]
        raise AssertionError(f"unexpected endpoint {endpoint}")


def test_fetch_api_league_reports_every_unmapped_name_in_one_pass():
    """Previously this raised on the FIRST unmapped team, so fixing the alias
    file and re-running in CI only ever surfaced one missing name at a time --
    observed twice in a row (Mainz, then Hoffenheim) burning a CI round-trip
    and API quota each time. All unmapped names should surface together, and
    players/squads (the expensive per-team call) should never run once any
    name in the league can't be resolved."""
    teams = [
        {"team": {"id": 1, "name": "Bayern Munich"}},
        {"team": {"id": 2, "name": "Some Unmapped FC"}},
        {"team": {"id": 3, "name": "Another Unknown SV"}},
    ]
    client = _FakeApiClient(teams, squads={})
    try:
        sync_rosters.fetch_api_league(client, "BUNDESLIGA")
        assert False, "expected UnknownTeam"
    except sync_rosters.UnknownTeam as exc:
        assert "Some Unmapped FC" in str(exc)
        assert "Another Unknown SV" in str(exc)
    assert all(call[0] != "players/squads" for call in client.calls)
