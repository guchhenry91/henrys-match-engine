import json
from pathlib import Path

import pytest

from leagues import config, fixtures
# IMPORTED, not restated. This file used to keep its own copy of the map, so
# adding Serie A had to update it in two places and failed here on the second --
# the duplication was the bug, not the missing entry.
from scripts.roster_integrity_check import CLUBS

ROOT = Path(__file__).resolve().parents[2]


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
