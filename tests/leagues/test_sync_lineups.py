import datetime as dt
import json

from scripts import sync_lineups


NOW = dt.datetime(2026, 8, 15, 11, 30, tzinfo=dt.timezone.utc)


class FakeClient:
    instances = []

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.calls = []
        self.instances.append(self)

    def get(self, path, **params):
        self.used += 1
        self.calls.append((path, params))
        if path == "fixtures":
            return [{
                "fixture": {"id": 999}, "league": {"id": 39},
                "teams": {"home": {"name": "Arsenal"},
                          "away": {"name": "Chelsea"}},
            }]
        return [
            {"team": {"name": "Arsenal"},
             "startXI": [{"player": {"name": f"A{i}"}} for i in range(11)],
             "substitutes": []},
            {"team": {"name": "Chelsea"},
             "startXI": [{"player": {"name": f"C{i}"}} for i in range(11)],
             "substitutes": []},
        ]


def test_lineup_fetch_is_saved_and_never_repeated(tmp_path, monkeypatch):
    news = tmp_path / "news.json"
    news.write_text(json.dumps({"PL": {}}), encoding="utf-8")
    fixture = {"league_key": "PL", "date": "2026-08-15T12:00:00Z",
               "home": "Arsenal", "away": "Chelsea"}
    monkeypatch.setenv("API_FOOTBALL_KEY", "hidden")
    monkeypatch.setattr(sync_lineups, "NEWS_PATH", news)
    monkeypatch.setattr(sync_lineups, "upcoming_fixtures", lambda now: [fixture])
    monkeypatch.setattr(sync_lineups, "Client", FakeClient)

    assert sync_lineups.main(now=NOW) == 0
    saved = json.loads(news.read_text(encoding="utf-8"))
    assert saved["PL"]["Arsenal"]["lineup_confirmed"] is True
    assert len(saved["PL"]["Chelsea"]["starters"]) == 11
    assert FakeClient.instances[-1].used == 2  # one date + one lineup request

    assert sync_lineups.main(now=NOW) == 0
    assert FakeClient.instances[-1].used == 0


# --- the 60/45/30/15 ladder --------------------------------------------------
# One poll in a 40-minute window was too brittle: clubs release late and
# unevenly, so a single attempt that landed before a slow club published got
# nothing and never looked again.

import pytest
from scripts.sync_lineups import RUNGS


def _rung(minutes):
    return next((r for r in sorted(RUNGS) if r >= minutes), None)


@pytest.mark.parametrize("minutes,expected", [
    (58, 60), (47, 60),      # first look, an hour out
    (43, 45), (32, 45),
    (28, 30), (17, 30),
    (12, 15), (3, 15),
])
def test_each_run_maps_to_the_smallest_rung_at_or_above_it(minutes, expected):
    assert _rung(minutes) == expected


def test_consecutive_runs_consume_different_rungs():
    """The whole point of the ladder. Iterating RUNGS as written (descending)
    matched 60 for every time remaining, spending one rung and silently skipping
    the other three."""
    runs = [58, 43, 28, 12]                      # a 15-minute cadence
    assert sorted({_rung(m) for m in runs}) == [15, 30, 45, 60]


def test_nothing_is_polled_before_the_top_rung():
    assert _rung(62) is None
    assert _rung(90) is None


def test_ladder_is_ordered_and_starts_at_an_hour():
    assert RUNGS[0] == 60 and RUNGS[-1] == 15
    assert list(RUNGS) == sorted(RUNGS, reverse=True)
