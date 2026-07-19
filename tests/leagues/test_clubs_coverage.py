import json
from pathlib import Path

import pytest

from leagues import config, fixtures

ROOT = Path(__file__).resolve().parents[2]
CLUBS = {"PL": "clubs.json", "LALIGA": "clubs_laliga.json",
         "BUNDESLIGA": "clubs_bundesliga.json", "LIGUE1": "clubs_ligue1.json"}


@pytest.mark.parametrize("league", list(config.LEAGUES))
def test_every_fixture_team_has_a_wellformed_colour(league):
    fx = fixtures.fetch_fixtures(league)            # cached feed; raises UnknownTeam on name gaps
    teams = sorted(set(fx["home"]) | set(fx["away"]))
    colours = json.loads((ROOT / "data" / "leagues" / CLUBS[league]).read_text("utf-8"))
    missing = [t for t in teams if t not in colours]
    assert not missing, f"{league}: no colour entry for {missing}"
    for t in teams:
        assert "primary" in colours[t] and "short" in colours[t], f"{league}/{t} malformed"
        assert colours[t]["primary"].startswith("#"), f"{league}/{t} bad hex"
