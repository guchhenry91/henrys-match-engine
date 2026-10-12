import io
import json
from datetime import datetime, timezone

from nfl import weather


def test_indoor_rules():
    assert weather.indoor("dome", "ATL97")
    assert weather.indoor("closed", "DAL00")
    assert not weather.indoor("open", "DAL00")
    assert weather.indoor("", "DAL00")          # retractable, not declared yet
    assert not weather.indoor("", "BUF00")
    assert not weather.indoor("outdoors", "GNB00")


def test_refresh_reads_the_kickoff_hour(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE", tmp_path / "weather.json")
    payload = {"hourly": {"time": ["2026-10-18T17:00", "2026-10-18T18:00"],
                          "temperature_2m": [50.0, 52.0], "wind_speed_10m": [9.0, 12.5]}}
    calls = []

    def opener(url, timeout=0):
        calls.append(url)
        return io.BytesIO(json.dumps(payload).encode())

    now = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
    games = [{"game_id": "g1", "stadium_id": "GNB00", "roof": "outdoors",
              "kickoff": "2026-10-18T18:00:00Z"},
             {"game_id": "g2", "stadium_id": "ATL97", "roof": "dome",
              "kickoff": "2026-10-18T18:00:00Z"}]
    assert weather.refresh(games, now=now, opener=opener) == 1
    assert len(calls) == 1                          # the dome game is never fetched
    assert weather.load()["g1"] == (52.0, 12.5)
    # fresh inside REFRESH_HOURS: no second fetch
    assert weather.refresh(games, now=now, opener=opener) == 0
    assert len(calls) == 1


def test_weather_is_off_by_default():
    from nfl import forecast as fc
    assert fc.WEATHER is False
    assert "wind" not in fc.features("rushing_yards")
    assert "wind" in fc.features("rushing_yards", weather=True)
