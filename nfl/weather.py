"""Kick-off weather for upcoming NFL games: Open-Meteo's forecast (free, no key).

nflverse fills `temp`/`wind` only AFTER a game, so the live board had them blank while
training had them filled. This supplies what can actually be known before kick-off:
the forecast for the stadium at the kick-off hour. Training keeps nflverse's recorded
weather (a forecast hours out is close to what happens); the gate scores the feature
with forecast-sized error added to the test rows, so it is judged as it will be used.

Indoor games (dome / closed roof, and retractable roofs not yet declared) get no
forecast: wind 0, temperature left blank, as in training.
Cache: data-raw/nfl/weather.json {game_id: {temp, wind, fetched_at, kickoff}}.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data-raw" / "nfl" / "weather.json"
URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly=temperature_2m,wind_speed_10m&temperature_unit=fahrenheit"
       "&wind_speed_unit=mph&timezone=UTC&forecast_days=16")
REFRESH_HOURS = 3
HORIZON_DAYS = 10

# Stadium -> (lat, lon). Keyed by nflverse stadium_id.
STADIUMS = {
    "ATL97": (33.755, -84.401), "BAL00": (39.278, -76.623), "BOS00": (42.091, -71.264),
    "BUF00": (42.774, -78.787), "BUF01": (42.774, -78.787), "CAR00": (35.226, -80.853),
    "CHI98": (41.862, -87.617), "CIN00": (39.095, -84.516), "CLE00": (41.506, -81.700),
    "DAL00": (32.748, -97.093), "DEN00": (39.744, -105.020), "DET00": (42.340, -83.046),
    "FRA00": (50.069, 8.645), "GER00": (48.219, 11.625), "GNB00": (44.501, -88.062),
    "HOU00": (29.685, -95.411), "IND00": (39.760, -86.164), "JAX00": (30.324, -81.638),
    "KAN00": (39.049, -94.484), "LAX01": (33.953, -118.339), "LON00": (51.556, -0.280),
    "LON02": (51.604, -0.066), "MAD01": (40.453, -3.688), "MEL00": (-37.820, 144.983),
    "MEX00": (19.303, -99.150), "MIA00": (25.958, -80.239), "MIN01": (44.974, -93.258),
    "MUN01": (48.219, 11.625), "NAS00": (36.166, -86.771), "NOR00": (29.951, -90.081),
    "NYC01": (40.814, -74.074), "PAR00": (48.924, 2.360), "PHI00": (39.901, -75.168),
    "PHO00": (33.528, -112.263), "PIT00": (40.447, -80.016), "RIO00": (-22.912, -43.230),
    "SAO00": (-23.545, -46.474), "SEA00": (47.595, -122.332), "SFO01": (37.403, -121.970),
    "TAM00": (27.976, -82.503), "VEG00": (36.091, -115.184), "WAS00": (38.908, -76.864),
}
# Retractable roofs: an upcoming game with no roof declared yet is treated as indoor.
RETRACTABLE = {"ATL97", "DAL00", "HOU00", "IND00", "PHO00"}


def indoor(roof, stadium_id) -> bool:
    r = str(roof or "").lower()
    if r in ("dome", "closed"):
        return True
    if r in ("", "nan", "none") and stadium_id in RETRACTABLE:
        return True
    return False


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load() -> dict:
    """{game_id: (temp, wind)} from the cache."""
    return {k: (v.get("temp"), v.get("wind")) for k, v in _load().items() if isinstance(v, dict)}


def _hour_value(payload: dict, kickoff: datetime):
    h = payload.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        return None
    want = kickoff.strftime("%Y-%m-%dT%H:00")
    if want not in times:
        return None
    i = times.index(want)
    return h["temperature_2m"][i], h["wind_speed_10m"][i]


def refresh(games, now=None, opener=urllib.request.urlopen) -> int:
    """Fetch forecasts for upcoming OUTDOOR games inside HORIZON_DAYS. `games` rows need
    game_id, stadium_id, roof and a UTC kick-off (`kickoff`). Never fatal."""
    now = now or datetime.now(timezone.utc)
    cache = _load()
    n = 0
    for g in games:
        gid, sid = str(g.get("game_id")), g.get("stadium_id")
        ko = g.get("kickoff")
        if not gid or sid not in STADIUMS or indoor(g.get("roof"), sid) or ko is None:
            continue
        ko = ko if isinstance(ko, datetime) else datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        if not (now < ko <= now + timedelta(days=HORIZON_DAYS)):
            continue
        old = cache.get(gid) or {}
        try:
            fresh = (now - datetime.fromisoformat(old["fetched_at"])) < timedelta(hours=REFRESH_HOURS)
        except Exception:
            fresh = False
        if fresh:
            continue
        lat, lon = STADIUMS[sid]
        try:
            with opener(URL.format(lat=lat, lon=lon), timeout=20) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except Exception as exc:                            # noqa: BLE001
            print(f"  weather: {gid} unavailable ({type(exc).__name__})")
            continue
        v = _hour_value(payload, ko.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0))
        if v is None:
            continue
        cache[gid] = {"temp": round(float(v[0]), 1), "wind": round(float(v[1]), 1),
                      "kickoff": ko.isoformat(), "fetched_at": now.isoformat(timespec="seconds")}
        n += 1
    if n:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return n
