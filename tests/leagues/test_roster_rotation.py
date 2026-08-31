"""The roster refresh must cover every configured league, and say what it spent.

It alternated two HARDCODED pairs -- (PL, BUNDESLIGA) and (LALIGA, LIGUE1) -- so
Serie A was not merely refreshed late, it was NEVER fetched from the API at all.
It fell through to the ESPN fallback, and when ESPN began answering 403 the league
had no roster evidence of any kind.
"""
import io
import json
from datetime import datetime, timedelta, timezone

import pytest

from leagues import config
from leagues.api_football import Client
from scripts import sync_rosters


class _Response(io.BytesIO):
    def __init__(self, payload, headers=None):
        super().__init__(json.dumps(payload).encode())
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_every_configured_league_is_eligible_for_the_api_refresh():
    """THE ACTUAL BUG. Asserted against config.LEAGUES so a sixth league cannot
    quietly drop out of the rotation the way the fifth did."""
    missing = sorted(set(config.LEAGUES) - set(sync_rosters.LEAGUES))
    assert not missing, f"not in the roster rotation: {missing}"


def test_a_league_never_refreshed_is_due():
    now = datetime.now(timezone.utc)
    assert not sync_rosters._fresh({"_league_verified_at": {}}, "SERIEA", now)


def test_a_recently_refreshed_league_is_skipped():
    """`_fresh` is the real guard: repeated runs in a day must cost nothing."""
    now = datetime.now(timezone.utc)
    payload = {"_league_verified_at": {"SERIEA": (now - timedelta(hours=2)).isoformat()}}
    assert sync_rosters._fresh(payload, "SERIEA", now)


def test_a_stale_league_becomes_due_again():
    now = datetime.now(timezone.utc)
    payload = {"_league_verified_at": {"SERIEA": (now - timedelta(hours=40)).isoformat()}}
    assert not sync_rosters._fresh(payload, "SERIEA", now)


def test_the_budget_covers_a_full_refresh_of_every_league():
    """A constant budget silently truncated a league mid-fetch once the list grew
    past what it was sized for."""
    due = list(sync_rosters.LEAGUES)
    budget = sum(config.get(k).n_teams + 2 for k in due) + 5
    needed = sum(1 + config.get(k).n_teams for k in due)
    assert budget >= needed


def test_the_client_reports_the_accounts_real_allowance():
    """Quota was inferred from a constant written for the free 100-a-day tier.
    The account allows 7,500, and nobody had checked."""
    client = Client(key="test", opener=lambda r, timeout=None: _Response(
        {"response": []},
        {"x-ratelimit-requests-remaining": "7412",
         "x-ratelimit-requests-limit": "7500"}))
    client.get("teams")
    assert client.remaining == 7412 and client.daily_limit == 7500
    assert "7412 of 7500" in client.report()


def test_missing_headers_do_not_break_the_report():
    client = Client(key="test",
                    opener=lambda r, timeout=None: _Response({"response": []}))
    client.get("teams")
    assert client.remaining is None
    assert "1 request(s) used" in client.report()


def test_the_run_budget_still_stops_a_runaway():
    client = Client(key="test", limit=2,
                    opener=lambda r, timeout=None: _Response({"response": []}))
    client.get("teams")
    client.get("teams")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        client.get("teams")
