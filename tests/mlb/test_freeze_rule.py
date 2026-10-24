"""MLB keeps its own freeze rule (regression, 2026-10-24).

MLB reuses nba.picks.freeze. When the NBA moved to a late freeze (last hour only,
after a pre-tip run MLB does not have), MLB inherited it: its last run before a
00:00 UTC first pitch is at 22:00, two hours out, so World Series Game 1 was never
frozen. MLB must still freeze on its last scheduled run before first pitch."""
from datetime import datetime, timezone

from mlb import book_lines as bl
from mlb import picks


def _t(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _payload():
    g = {"game_id": "849806", "tipoff": "2026-10-25T00:00:00Z", "home": "MIL", "away": "TBA",
         "pick": "MIL", "p_pick": 0.6}
    p = {"game_id": "849806", "tipoff": "2026-10-25T00:00:00Z", "market": "hits", "player": "A",
         "player_id": 1, "team": "MIL", "opponent": "TBA", "line": 0.5, "line_source": "pinnacle",
         "probability": 0.6}
    return {"games": [g], "props": {"hits": {"picks": [p]}}}


def test_the_22_utc_run_freezes_a_midnight_first_pitch():
    log = {}
    assert picks.freeze(_payload(), log, _t("2026-10-24T14:01"), bl.is_last_run_before) == 0
    assert picks.freeze(_payload(), log, _t("2026-10-24T22:01"), bl.is_last_run_before) == 2
    assert all(e["locked_at"].startswith("2026-10-24T22:01") for e in log.values())


def test_nba_still_freezes_only_in_its_last_hour():
    from nba import picks as nba_picks
    log = {}
    assert nba_picks.freeze(_payload(), log, _t("2026-10-24T22:01")) == 0
    assert nba_picks.freeze(_payload(), log, _t("2026-10-24T23:05")) == 2
