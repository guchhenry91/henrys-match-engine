"""The only bug in the NFL feature layer that would invalidate everything.

A feature that includes the game it describes makes every backtest number look
excellent and every live prediction worthless, and it does not announce itself --
the report simply comes back better than it should. These tests assert that a
player's own performance cannot reach his own features.
"""
import pandas as pd
import pytest

from nfl import config, features


def _weeks(values, stat="receiving_yards"):
    """One player, one season, `values` in successive weeks."""
    rows = []
    for i, value in enumerate(values, start=1):
        row = {"player_id": "P1", "player_display_name": "Test Player",
               "position": "WR", "team": "AAA", "season": 2024, "week": i,
               "season_type": "REG", "opponent_team": "ZZZ",
               "passing_yards": 0.0, "rushing_yards": 0.0, "receiving_yards": 0.0,
               "receptions": 6.0, "carries": 6.0, "targets": 9.0,
               "attempts": 30.0, "completions": 20.0,
               "passing_tds": 0.0, "rushing_tds": 0.0, "receiving_tds": 0.0,
               "touchdowns": 0, "touches": 12.0}
        row[stat] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def test_history_never_includes_the_current_game():
    """A single enormous game must not raise the features OF that game."""
    frame = _weeks([10] * 12 + [999])
    built = features.build(frame, "receiving_yards")
    spike = built[built["week"] == 13]
    assert not spike.empty
    assert spike["hist_rate"].iloc[0] == pytest.approx(10.0)
    assert spike["form5"].iloc[0] == pytest.approx(10.0)
    assert spike["line"].iloc[0] == pytest.approx(10.5)


def test_the_spike_does_show_up_in_the_next_game():
    """...but it must reach the NEXT one, or the window is simply broken."""
    frame = _weeks([10] * 12 + [999, 10])
    built = features.build(frame, "receiving_yards")
    after = built[built["week"] == 14]
    assert after["form5"].iloc[0] > 100, "history is not updating at all"


def test_last_five_is_strictly_prior_and_ordered():
    frame = _weeks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    built = features.build(frame, "receiving_yards")
    row = built[built["week"] == 10].iloc[0]
    assert row["last_five"] == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert 10.0 not in row["last_five"], "the current game leaked into last five"


def test_last_five_is_what_the_model_used():
    """The board shows these five; the projection must come from the same five."""
    frame = _weeks([20] * 10)
    built = features.build(frame, "receiving_yards")
    row = built.iloc[-1]
    assert len(row["last_five"]) == config.FORM_GAMES
    assert row["form5"] == pytest.approx(sum(row["last_five"]) / len(row["last_five"]))


def test_a_player_below_the_games_threshold_is_not_published():
    built = features.build(_weeks([10] * 4), "receiving_yards")
    assert built.empty


def test_a_player_without_a_role_is_not_published():
    """A receiver with no carries must never acquire a rushing line."""
    frame = _weeks([10] * 12, stat="rushing_yards")
    frame["carries"] = 0.0
    assert features.build(frame, "rushing_yards").empty


def test_touchdown_market_is_a_flag_not_a_count():
    """Three touchdowns still settle one bet, and a passing TD settles none."""
    from nfl import data

    frame = pd.DataFrame([
        {"rushing_tds": 1.0, "receiving_tds": 2.0, "passing_tds": 0.0},   # hat-trick
        {"rushing_tds": 0.0, "receiving_tds": 0.0, "passing_tds": 4.0},   # QB threw 4
        {"rushing_tds": 0.0, "receiving_tds": 1.0, "passing_tds": 0.0},   # one catch
        {"rushing_tds": 0.0, "receiving_tds": 0.0, "passing_tds": 0.0},   # nothing
    ])
    assert list(data.anytime_touchdown(frame)) == [1, 0, 1, 0]
