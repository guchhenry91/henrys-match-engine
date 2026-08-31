"""Small quota-aware client for the API-Football account.

IT NOW READS THE ACCOUNT'S OWN ALLOWANCE from the rate-limit headers, the way
nfl/api.py already did. Before, quota was inferred from a limit constant written
when the plan was the free 100-a-day tier -- so the roster rotation was built to
spend about 42 calls every other day against an account that actually allows
7,500. Every decision about how much to fetch was being made against a number
nobody had checked.

That mattered in the other direction too: when 7,500 calls disappeared in four
hours, finding out where meant reading thirty-one workflow logs, because the
scripts using this client reported nothing about what they had spent.
"""
import json
import os
import urllib.parse
import urllib.request


BASE = "https://v3.football.api-sports.io"


class Client:
    def __init__(self, key=None, limit=90, opener=urllib.request.urlopen):
        self.key = key or os.environ.get("API_FOOTBALL_KEY")
        if not self.key:
            raise RuntimeError("API_FOOTBALL_KEY is not set")
        self.limit = limit
        self.used = 0
        self.opener = opener
        # What the ACCOUNT says it has left, as opposed to this run's own budget.
        # None until the first response carries the headers.
        self.remaining = None
        self.daily_limit = None

    def get(self, path, **params):
        if self.used >= self.limit:
            raise RuntimeError(f"API-Football run budget exhausted ({self.limit})")
        query = urllib.parse.urlencode({k: v for k, v in params.items()
                                       if v is not None})
        url = f"{BASE}/{path.lstrip('/')}" + (f"?{query}" if query else "")
        request = urllib.request.Request(
            url, headers={"x-apisports-key": self.key,
                          "User-Agent": "henrys-match-engine/1.0"})
        with self.opener(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
            self._read_allowance(getattr(response, "headers", None))
        self.used += 1
        errors = payload.get("errors")
        if errors:
            raise RuntimeError(f"API-Football error: {errors}")
        return payload.get("response") or []

    def _read_allowance(self, headers) -> None:
        """Record the account's remaining daily allowance, if the headers say."""
        if not headers:
            return
        for name, attr in (("x-ratelimit-requests-remaining", "remaining"),
                           ("x-ratelimit-requests-limit", "daily_limit")):
            raw = headers.get(name)
            if raw is None:
                continue
            try:
                setattr(self, attr, int(raw))
            except (TypeError, ValueError):
                pass

    def report(self) -> str:
        """One line, printed by callers so a run's spend is never invisible."""
        out = f"API-Football: {self.used} request(s) used this run"
        if self.remaining is not None and self.daily_limit is not None:
            out += f"; account has {self.remaining} of {self.daily_limit} left today"
        return out
