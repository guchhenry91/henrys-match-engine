import io
import json

import pytest

from leagues.api_football import Client


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_client_sends_secret_in_header_not_url():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["key"] = request.get_header("X-apisports-key")
        seen["timeout"] = timeout
        return _Response(json.dumps({"errors": [], "response": [{"id": 1}]}).encode())

    client = Client(key="secret-value", opener=opener)
    assert client.get("teams", league=39, season=2026) == [{"id": 1}]
    assert seen["key"] == "secret-value"
    assert "secret-value" not in seen["url"]


def test_client_stops_at_its_run_budget():
    client = Client(key="x", limit=0, opener=lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        client.get("teams")


def test_client_stops_at_the_account_floor():
    """Budget plan 2026-09-26: keep 1,500 of the 7,500 daily calls in reserve."""
    calls = []

    def opener(request, timeout):
        calls.append(1)
        r = _Response(json.dumps({"errors": [], "response": []}).encode())
        r.headers = {"x-ratelimit-requests-remaining": "1400"}
        return r

    client = Client(key="x", opener=opener, pace=0)
    client.get("teams")
    with pytest.raises(RuntimeError, match="floor"):
        client.get("teams")
    assert len(calls) == 1
