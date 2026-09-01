import io
import json

from scripts import sync_rosters


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_roster_request_retries_transient_failure():
    """THE RETRY MOVED, IT WAS NOT DROPPED. It used to live in the ESPN fetcher,
    which was the only retry in the roster path; when that source was removed the
    API-Football client gained one, because a single dropped connection would
    otherwise cost a league its whole refresh cycle."""
    import io, json as _json
    from leagues.api_football import Client

    calls = []

    class _R(io.BytesIO):
        def __init__(self):
            super().__init__(_json.dumps({"response": [{"ok": True}]}).encode())
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_url(_request, timeout=None):
        calls.append(timeout)
        if len(calls) < 3:
            raise OSError("temporary TLS failure")
        return _R()

    sleeps = []
    client = Client(key="test", opener=open_url)
    assert client.get("teams", sleeper=sleeps.append) == [{"ok": True}]
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_an_api_error_is_not_retried():
    """The endpoint ANSWERED -- a bad league id or a spent allowance is a 'no',
    and asking again just spends the allowance twice on the same answer."""
    import io, json as _json
    import pytest as _pytest
    from leagues.api_football import Client

    calls = []

    class _R(io.BytesIO):
        def __init__(self):
            super().__init__(_json.dumps({"errors": {"token": "bad"}}).encode())
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_url(_request, timeout=None):
        calls.append(1)
        return _R()

    client = Client(key="test", opener=open_url)
    with _pytest.raises(RuntimeError, match="API-Football error"):
        client.get("teams")
    assert len(calls) == 1


def test_failed_refresh_retains_complete_verified_snapshot(tmp_path, monkeypatch):
    out = tmp_path / "rosters.json"
    old = {"_verified_at": "2026-07-21T10:00:00+00:00"}
    old.update({key: {"Club": {"players": []}} for key in sync_rosters.LEAGUES})
    out.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(sync_rosters, "OUT", out)
    monkeypatch.setenv("API_FOOTBALL_KEY", "test")
    monkeypatch.setattr(sync_rosters, "Client",
                        lambda **_kw: type("FakeClient", (),
                                          {"used": 0, "report": lambda self: "fake"})())
    monkeypatch.setattr(sync_rosters, "fetch_api_league",
                        lambda *_args: (_ for _ in ()).throw(OSError("TLS")))

    # EVERY league failed, so nothing was refreshed: the file must be byte
    # identical, not rewritten with new metadata around unchanged squads.
    assert sync_rosters.main() == 1
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
                        lambda **_kw: type("FakeClient", (),
                                          {"used": 0,
                                           # sync_rosters prints client.report()
                                           # so a run's spend is never invisible.
                                           "report": lambda self: "fake"})())

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
    batch discarded the already-good PL fetch too. Each league must succeed or
    fail independently -- and with the ESPN fallback removed, a failure now means
    the league simply keeps what it had rather than being rescued from a second,
    differently-shaped source."""
    out = tmp_path / "rosters.json"
    monkeypatch.setattr(sync_rosters, "OUT", out)
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setenv("FORCE_ROSTER_LEAGUES", "pl,bundesliga")

    def fake_fetch(_client, key):
        if key == "BUNDESLIGA":
            raise ValueError("'FSV Mainz 05' is not mapped")
        return {"Arsenal": {"source": "api-football:team:1", "players": []}}

    monkeypatch.setattr(sync_rosters, "fetch_api_league", fake_fetch)
    monkeypatch.setattr(sync_rosters, "Client",
                        lambda **_kw: type("FakeClient", (),
                                          {"used": 0,
                                           # sync_rosters prints client.report()
                                           # so a run's spend is never invisible.
                                           "report": lambda self: "fake"})())

    assert sync_rosters.main() == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["PL"] == {"Arsenal": {"source": "api-football:team:1", "players": []}}
    assert "PL" in written["_league_verified_at"]
    # THE FAILED LEAGUE IS LEFT ALONE, not rescued from a second source. There is
    # no fallback any more: it keeps whatever it had (nothing, here) and stays
    # unstamped, so it is due again on the next run.
    assert "BUNDESLIGA" not in written
    assert "BUNDESLIGA" not in written["_league_verified_at"]
    assert "NOT refreshed this run: BUNDESLIGA" in written["_source"]


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
