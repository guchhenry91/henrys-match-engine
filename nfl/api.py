"""Quota-aware API-NFL client.

WHY THIS IS NOT THE api_football CLIENT COPIED. That one counts its own requests
and tells nobody. When the API-Football account burned 7,500 calls in four hours
this month, working out where they went took reading thirty-one workflow logs,
because two of the four scripts using it reported nothing at all. This client
reads the rate-limit headers the API already sends and every caller prints what
it spent.

THE CIRCUIT BREAKER IS THE OTHER HALF. Once the daily limit is reached, every
subsequent run of the football scripts still fired four doomed requests at it --
about 120 wasted calls a day, more than the entire free allowance, spent learning
something the previous run already knew. Here the exhaustion is written down with
its date, and every caller checks it first.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://v1.american-football.api-sports.io"
ROOT = Path(__file__).resolve().parent.parent
BREAKER = ROOT / "data-raw" / "nfl" / "_quota.json"


class QuotaExhausted(RuntimeError):
    """The daily allowance is gone. Not an error to retry -- an error to stop on."""


class Client:
    def __init__(self, key=None, budget=60, opener=urllib.request.urlopen):
        self.key = key or os.environ.get("API_NFL_KEY")
        if not self.key:
            raise RuntimeError("API_NFL_KEY is not set")
        self.budget = budget
        self.used = 0
        self.remaining = None
        self.limit = None
        self.opener = opener

    def get(self, path: str, **params):
        if self.used >= self.budget:
            raise QuotaExhausted(f"this run's budget of {self.budget} is spent")
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{BASE}/{path.lstrip('/')}" + (f"?{query}" if query else "")
        request = urllib.request.Request(
            url, headers={"x-apisports-key": self.key,
                          "User-Agent": "henrys-match-engine/1.0"})
        try:
            with self.opener(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
                headers = response.headers
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"API-NFL HTTP {exc.code} for /{path}") from exc
        self.used += 1

        for attr, header in (("remaining", "x-ratelimit-requests-remaining"),
                             ("limit", "x-ratelimit-requests-limit")):
            try:
                setattr(self, attr, int(headers.get(header)))
            except (TypeError, ValueError):
                pass

        errors = payload.get("errors")
        # The API reports a spent allowance as a normal 200 with an error body,
        # which is exactly how it slips past naive error handling and gets retried
        # forever.
        if errors and any("limit" in str(v).lower() for v in
                          (errors.values() if isinstance(errors, dict) else errors)):
            trip_breaker(str(errors))
            raise QuotaExhausted(str(errors))
        if errors:
            raise RuntimeError(f"API-NFL error: {errors}")
        return payload.get("response") or []

    def report(self) -> str:
        """One line, printed by every caller. The thing that was missing before."""
        known = ""
        if self.remaining is not None and self.limit is not None:
            known = f"; account has {self.remaining} of {self.limit} left today"
        return f"API-NFL: {self.used} request(s) used this run{known}"


def trip_breaker(reason: str) -> None:
    BREAKER.parent.mkdir(parents=True, exist_ok=True)
    BREAKER.write_text(json.dumps({
        "exhausted_on": datetime.now(timezone.utc).date().isoformat(),
        "reason": reason[:300],
        "_note": ("Written when API-NFL reports the daily limit reached. Every "
                  "caller checks this first and no-ops for the rest of that UTC "
                  "day rather than firing doomed requests at a wall."),
    }, indent=2) + "\n", encoding="utf-8")


def breaker_tripped() -> bool:
    """True if the allowance is already known to be gone TODAY."""
    try:
        raw = json.loads(BREAKER.read_text(encoding="utf-8"))
    except Exception:
        return False
    today = datetime.now(timezone.utc).date().isoformat()
    return raw.get("exhausted_on") == today


def available() -> bool:
    """Can we usefully call the API at all right now?"""
    return bool(os.environ.get("API_NFL_KEY")) and not breaker_tripped()
