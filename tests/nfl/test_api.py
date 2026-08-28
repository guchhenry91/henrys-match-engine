"""The API client must count itself, stop when the allowance is gone, and never
let a ruled-out player onto the board.

Both halves exist because of what happened on the FOOTBALL key this month: 7,500
calls vanished in four hours and finding out where took reading thirty-one
workflow logs, because two of the four scripts using it reported nothing. Then
every run afterwards kept firing doomed requests at an allowance it already knew
was spent -- roughly 120 wasted calls a day.
"""
import io
import json
from datetime import datetime, timedelta, timezone

import pytest

from nfl import api
from scripts import sync_nfl_injuries as sync


class _Response(io.BytesIO):
    def __init__(self, payload, headers=None):
        super().__init__(json.dumps(payload).encode())
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(payload, headers=None):
    def open_it(request, timeout=None):
        return _Response(payload, headers)
    return open_it


@pytest.fixture(autouse=True)
def _isolate_breaker(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "BREAKER", tmp_path / "quota.json")
    monkeypatch.setenv("API_NFL_KEY", "test-key")


def test_it_counts_its_own_requests():
    client = api.Client(opener=_opener({"response": [1, 2]}))
    client.get("teams")
    client.get("teams")
    assert client.used == 2
    assert "2 request(s)" in client.report()


def test_it_reads_the_accounts_remaining_allowance():
    """The number that was invisible before, and had to be reconstructed by hand."""
    client = api.Client(opener=_opener(
        {"response": []},
        {"x-ratelimit-requests-remaining": "7412", "x-ratelimit-requests-limit": "7500"}))
    client.get("teams")
    assert client.remaining == 7412 and client.limit == 7500
    assert "7412 of 7500" in client.report()


def test_a_spent_allowance_trips_the_breaker_rather_than_raising_a_plain_error():
    """The API reports this as a NORMAL 200 with an error body, which is exactly
    how it slips past naive handling and gets retried forever."""
    client = api.Client(opener=_opener(
        {"errors": {"requests": "You have reached the request limit for the day"}}))
    with pytest.raises(api.QuotaExhausted):
        client.get("injuries", team=1)
    assert api.breaker_tripped()
    assert not api.available()


def test_the_breaker_only_applies_to_today():
    """UTC, matching the code and the API's own daily reset.

    This used date.today() -- the LOCAL date -- while breaker_tripped() uses UTC.
    For the hours each day when the two disagree, "yesterday local" IS "today
    UTC", and the test failed on a correct implementation. It duly broke the
    moment the date rolled over mid-session. A date-dependent test must use the
    same clock as the thing it is testing.
    """
    api.BREAKER.parent.mkdir(parents=True, exist_ok=True)
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    api.BREAKER.write_text(json.dumps({"exhausted_on": yesterday}), encoding="utf-8")
    assert not api.breaker_tripped(), "yesterday's exhaustion must not block today"


def test_the_breaker_does_apply_today():
    """The other half, which the original never checked."""
    api.BREAKER.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    api.BREAKER.write_text(json.dumps({"exhausted_on": today}), encoding="utf-8")
    assert api.breaker_tripped()


def test_a_run_budget_stops_a_runaway_loop():
    client = api.Client(budget=2, opener=_opener({"response": []}))
    client.get("teams")
    client.get("teams")
    with pytest.raises(api.QuotaExhausted):
        client.get("teams")


def test_no_key_means_unavailable_not_a_crash(monkeypatch):
    monkeypatch.delenv("API_NFL_KEY", raising=False)
    assert not api.available()


# --- injury classification ---------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    ("Out", "out"),
    ("Injured Reserve", "out"),
    ("IR", "out"),
    ("Suspended", "out"),
    ("Inactive", "out"),
    ("Doubtful", "doubt"),
    ("Questionable", "doubt"),
    ("Limited Participation", "doubt"),
    ("Full Participation", "fit"),
    ("Probable", "fit"),
    ("", "fit"),
    (None, "fit"),
])
def test_status_classification(status, expected):
    assert sync.classify(status) == expected


def test_classification_is_case_and_spacing_insensitive():
    """Status strings vary by source; a new spelling must not become 'fit'."""
    assert sync.classify("  OUT  ") == "out"
    assert sync.classify("doubtful") == "doubt"
