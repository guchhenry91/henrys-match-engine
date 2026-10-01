"""Phase 3: players inheriting an absent teammate's volume (nfl/vacancy.py)."""
import numpy as np
import pandas as pd

from nfl import features, vacancy


def _weeks(rows):
    return pd.DataFrame(rows, columns=["player_id", "team", "season", "week", "_share"])


def test_vacated_counts_only_players_who_played_last_game_and_are_absent_now():
    rows = [("a", "KC", 2025, 1, 0.30), ("b", "KC", 2025, 1, 0.20), ("c", "KC", 2025, 1, 0.10),
            ("a", "KC", 2025, 2, 0.35), ("c", "KC", 2025, 2, 0.15),        # b missed week 2
            ("a", "KC", 2025, 3, 0.30), ("c", "KC", 2025, 3, 0.20)]        # b still out: not re-counted
    v = features.vacated_share(_weeks(rows)).set_index("week")["vacated"]
    assert abs(v.loc[2] - 0.20) < 1e-9
    assert 3 not in v.index


def test_adjust_lifts_only_when_someone_is_out_and_c_is_set():
    p = np.array([0.40, 0.40])
    out = vacancy.adjust(p, form5=[50, 50], vacated=[0.0, 0.25], line=[45.5, 45.5], c=0.3)
    assert out[0] == 0.40 and out[1] > 0.40
    assert np.allclose(vacancy.adjust(p, [50, 50], [0.25, 0.25], [45.5, 45.5], None), p)


def test_boost_is_his_slice_of_the_freed_volume_against_the_line():
    b = vacancy.boost(form5=[60.0], vacated=[0.25], line=[50.0])
    assert abs(b[0] - 60.0 * (0.25 / 0.75) / 50.0) < 1e-9


def test_the_live_board_reads_the_season_being_played(monkeypatch):
    """2026-10-01: the board priced every prop on 2025's form; it must ask for 2026."""
    from nfl import config, data
    asked = []
    monkeypatch.setattr(data, "player_weeks", lambda seasons=None: asked.append(seasons) or "frame")
    assert data.live_player_weeks() == "frame"
    assert config.CURRENT_SEASON in asked[0]


def test_before_the_season_the_board_falls_back_to_completed_seasons(monkeypatch):
    from urllib.error import HTTPError
    from nfl import data

    def fake(seasons=None):
        if seasons:
            raise HTTPError("u", 404, "no file yet", {}, None)
        return "completed"
    monkeypatch.setattr(data, "player_weeks", fake)
    assert data.live_player_weeks() == "completed"
