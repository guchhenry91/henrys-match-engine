"""Serie A must be wired everywhere a league is enumerated, not just configured.

Adding it found the four-league list written out in eleven separate places across
the engine, the scripts and the UI. A league that is configured but missing from
one of them fails silently and specifically: no board file, or picks that never
lock, or a parlay section that skips it.
"""
import json
import pathlib
import re

import pytest

from leagues import config, odds, parlays, players, publish
from scripts import lock_picks, refresh_results

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_serie_a_is_configured():
    lg = config.get("SERIEA")
    assert lg.fd_code == "I1" and lg.fd_code2 == "I2"
    assert lg.fixture_slug == "serie-a-2026"
    assert lg.understat == "ITA-Serie A"
    assert lg.n_teams == 20


def test_it_uses_head_to_head_like_the_official_table():
    """Serie A separates clubs level on points by their meetings, not goal
    difference. 'gd' would disagree with the official table at exactly the
    positions people care about."""
    assert config.get("SERIEA").tiebreak == "h2h"


@pytest.mark.parametrize("mapping,label", [
    (publish.FILE_FOR, "publish.FILE_FOR"),
    (lock_picks.FILES, "lock_picks.FILES"),
    (refresh_results.FILES, "refresh_results.FILES"),
    (odds.DIV, "odds.DIV"),
    (players._API_LEAGUE_IDS, "players._API_LEAGUE_IDS"),
])
def test_every_live_path_mapping_covers_every_configured_league(mapping, label):
    """The real risk is not a typo, it is a league configured and then forgotten
    in one of these. Asserted against config.LEAGUES so a sixth league fails here
    rather than silently losing its board or its lock."""
    missing = sorted(set(config.LEAGUES) - set(mapping))
    assert not missing, f"{label} is missing {missing}"


def test_the_parlay_order_covers_every_league():
    missing = sorted(set(config.LEAGUES) - set(parlays.LEAGUES_ORDER))
    assert not missing, f"parlays.LEAGUES_ORDER is missing {missing}"


def test_the_roster_sync_covers_every_league():
    """Without this Serie A published a data_warning saying no roster evidence
    existed at all, and player attribution fell back to last season's club."""
    from scripts import sync_rosters
    missing = sorted(set(config.LEAGUES) - set(sync_rosters.LEAGUES))
    assert not missing, f"sync_rosters.LEAGUES is missing {missing}"


def test_the_ui_knows_every_league():
    """index.html derives its slug list from one map; these are the two places a
    league still has to be named."""
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    files = re.search(r"const FILES=\{([^}]*)\}", html).group(1)
    names = re.search(r"const LG=\{([^}]*)\}", html).group(1)
    for key in config.LEAGUES:
        assert key in files, f"index.html FILES is missing {key}"
        assert key in names, f"index.html LG is missing {key}"


def test_every_league_has_a_board_file_name():
    """A duplicate output name would have two leagues overwriting each other."""
    assert len(set(publish.FILE_FOR.values())) == len(publish.FILE_FOR)


def test_the_published_serie_a_board_is_shaped_like_the_others():
    path = ROOT / "data" / "leagues" / "seriea.json"
    if not path.exists():
        pytest.skip("Serie A board not built in this checkout")
    board = json.loads(path.read_text(encoding="utf-8"))
    for key in ("league", "updated", "record", "matches", "table", "standings"):
        assert key in board, f"seriea.json is missing {key}"
    assert board["league"] == "Serie A"
    assert len(board["table"]) == config.get("SERIEA").n_teams
