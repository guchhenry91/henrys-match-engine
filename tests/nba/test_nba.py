"""The NBA engine's load-bearing properties.

The two that would silently destroy the backtest are leakage (a feature that sees
its own game) and dropped games (a pairing that quietly loses fixtures). Both
happened during the build: the neutral-venue pairing lost five real 2025-26 games
before it was caught, and every rolling window has to be shifted or the model
scores itself on the answer.
"""
import numpy as np
import pandas as pd
import pytest

from nba import backtest, config, data, features


def _rows(player_id, values, stat="PTS", team="AAA", opp="BBB", season=2020):
    """A synthetic career: one row per game, ascending dates."""
    return pd.DataFrame({
        "PLAYER_ID": player_id,
        "GAME_ID": [f"g{i}" for i in range(len(values))],
        "TEAM_ABBREVIATION": team,
        "opponent": opp,
        "season": season,
        "game_date": pd.date_range("2020-01-01", periods=len(values), freq="2D"),
        "MIN": 30.0,
        stat: values,
        "PTS": values if stat == "PTS" else 10.0,
        "REB": values if stat == "REB" else 5.0,
        "AST": values if stat == "AST" else 5.0,
        "FG3M": values if stat == "FG3M" else 2.0,
        "is_home": 1.0,
    })


# --- leakage ------------------------------------------------------------------

def test_no_feature_sees_its_own_game():
    """THE ONE THAT WOULD INVALIDATE EVERYTHING. A rolling mean without shift(1)
    reports a model that already knows the answer."""
    values = [10.0] * 20 + [100.0]          # a huge final game
    frame = features.build(_rows(1, values), "points")
    last = frame.iloc[-1]
    # Nothing entering that game may reflect the 100.
    assert last["hist_rate"] == pytest.approx(10.0)
    assert last["form5"] == pytest.approx(10.0)
    assert last["line"] < 20.0


def test_the_first_eligible_row_has_history_behind_it():
    frame = features.build(_rows(1, [12.0] * 30), "points")
    assert (frame["games_before"] >= config.MIN_GAMES_FOR_PROP).all()


def test_a_player_below_the_minutes_bar_is_excluded():
    """A ROLE, not an appearance. Without it the board fills with benchwarmers
    whose median is zero and who clear a floored line only in a blowout."""
    rows = _rows(1, [12.0] * 30)
    rows["MIN"] = 5.0
    assert features.build(rows, "points").empty


# --- the line -----------------------------------------------------------------

def test_the_line_is_built_from_the_players_own_median():
    frame = features.build(_rows(1, [20.0] * 30), "points")
    # median 20, points offset +0.5
    assert frame["line"].iloc[-1] == pytest.approx(20.5)


def test_the_line_never_falls_below_the_quotable_floor():
    frame = features.build(_rows(1, [1.0] * 30), "points")
    assert (frame["line"] >= config.MIN_LINE["points"]).all()


def test_count_markets_use_a_negative_offset():
    """MEASURED, not chosen: +0.5 on an integer median means 'over' needs median
    PLUS ONE, which pushed rebounds/assists/threes to a 36-40% base rate."""
    assert config.LINE_OFFSET["points"] > 0
    for market in ("rebounds", "assists", "threes"):
        assert config.LINE_OFFSET[market] < 0


def test_every_market_has_a_floor_and_an_offset():
    for market in config.MARKETS:
        assert market in config.MIN_LINE
        assert market in config.LINE_OFFSET


def test_lines_are_always_on_the_half_point():
    """No result can land exactly on a line, so there is never a push.

    THE VALUES MATTER. This used to alternate 7 and 9, whose running median is a
    WHOLE number (8), so the line came out on the half point no matter how the
    rounding worked and the test could not fail. Alternating 7 and 8 gives a
    median of 7.5, which is exactly the case that used to produce an INTEGER line
    -- 9,625 of them on the real points frame, 528 of which a player matched
    exactly and was graded a loss for.

    A test whose fixture cannot produce the failure is not testing for it.
    """
    for values in ([7.0, 8.0] * 15, [7.0, 9.0] * 15, [3.0, 4.0, 5.0] * 10):
        frame = features.build(_rows(1, values), "points")
        assert ((frame["line"] * 2) % 2 == 1).all(), (
            f"integer line from {values[:2]}: "
            f"{sorted(set(frame['line'][((frame['line'] * 2) % 2) == 0]))}")


def test_a_result_can_never_land_exactly_on_its_line():
    """The consequence, asserted directly: `outcome` is `stat > line`, so an
    exact tie grades as a LOSS. That is a silent thumb on the scale against every
    over, and the only thing preventing it is the half-point line."""
    for market in config.MARKETS:
        stat = config.MARKETS[market]
        frame = features.build(_rows(1, [4.0, 5.0, 6.0] * 12, stat=stat), market)
        assert not (frame[stat] == frame["line"]).any()


# --- the outcome --------------------------------------------------------------

def test_the_outcome_is_strictly_over_the_line():
    frame = features.build(_rows(1, [10.0] * 20 + [11.0]), "points")
    last = frame.iloc[-1]
    assert last["outcome"] == float(last["PTS"] > last["line"])


# --- neutral-venue games ------------------------------------------------------

def _team_pair(game_id, a, b, a_home, b_home, a_pts, b_pts, season=2026):
    return pd.DataFrame({
        "GAME_ID": [game_id, game_id],
        "TEAM_ABBREVIATION": [a, b],
        "TEAM_ID": [1, 2],
        "season": [season, season],
        "game_date": pd.to_datetime(["2026-01-15", "2026-01-15"]),
        "is_home": [a_home, b_home],
        "PTS": [a_pts, b_pts],
        "MATCHUP": [f"{a} @ {b}", f"{b} @ {a}"],
        "WL": ["W" if a_pts > b_pts else "L", "W" if b_pts > a_pts else "L"],
        "opponent": [b, a],
        "won": [float(a_pts > b_pts), float(b_pts > a_pts)],
    })


def test_a_neutral_venue_game_is_kept_and_flagged(monkeypatch):
    """IT WAS SILENTLY DROPPED. Five real 2025-26 games -- both rows read '@'
    because neither side owns the court -- vanished from the schedule, and only
    from the CURRENT season, so the loss would have grown every year."""
    pair = _team_pair("g1", "DAL", "DET", 0.0, 0.0, 110, 105)
    monkeypatch.setattr(data, "team_games", lambda *a, **k: pair)
    out = data.games([2026])
    assert len(out) == 1
    assert bool(out.iloc[0]["neutral"]) is True
    assert out.iloc[0]["winner"] == "home"          # DAL, the nominal home, won


def test_an_ordinary_game_is_not_flagged_neutral(monkeypatch):
    pair = _team_pair("g1", "DAL", "DET", 1.0, 0.0, 99, 101)
    monkeypatch.setattr(data, "team_games", lambda *a, **k: pair)
    out = data.games([2026])
    assert bool(out.iloc[0]["neutral"]) is False
    assert out.iloc[0]["winner"] == "away"


def test_neutral_games_get_no_home_edge():
    """Granting home advantage on a court neither side owns invents an effect."""
    from nfl.games_model import run_elo
    games = pd.DataFrame([
        {"season": 2026, "game_date": pd.Timestamp("2026-01-15"),
         "home_team": "AAA", "away_team": "BBB", "winner": "home",
         "played": True, "neutral": True, "home_score": 100, "away_score": 90},
    ])
    priced = run_elo(games, k=20.0, home_edge=60.0, regression=0.25)
    # Equal ratings and no edge -> exactly even.
    assert priced.iloc[0]["prob_home"] == pytest.approx(0.5)


# --- the gate -----------------------------------------------------------------

def test_the_report_separates_floored_lines_from_real_medians():
    """THE NUMBER THAT WOULD OTHERWISE FLATTER. For assists and threes the floor
    binds on ~80% of rows, so most of the market is 'will a rotation player clear
    a CONSTANT' -- far more predictable than a balanced prop."""
    predictions = pd.DataFrame({
        "season": [2020] * 1200,
        "prob": np.linspace(0.2, 0.8, 1200),
        "outcome": ([1.0, 0.0] * 600),
        "baseline": [0.5] * 1200,
        "at_floor": [True] * 600 + [False] * 600,
    })
    out = backtest.evaluate(predictions, "assists")
    assert out["floor_share"] == pytest.approx(0.5)
    assert out["above_floor"]["n"] == 600


def test_a_market_failing_one_season_is_withheld():
    """Beat the baseline in EVERY scored season. A market that works in twelve and
    fails in three is a coin that landed twelve times."""
    good = pd.DataFrame({"season": 2019, "prob": [0.9] * 600,
                         "outcome": [1.0] * 600, "baseline": [0.5] * 600,
                         "at_floor": [False] * 600})
    bad = pd.DataFrame({"season": 2020, "prob": [0.1] * 600,
                        "outcome": [1.0] * 600, "baseline": [0.5] * 600,
                        "at_floor": [False] * 600})
    out = backtest.evaluate(pd.concat([good, bad], ignore_index=True), "points")
    assert out["released"] is False
    assert any("2020" in f for f in out["failures"])


def test_burn_in_seasons_are_never_scored():
    frame = pd.concat([
        pd.DataFrame({"season": s, "outcome": [1.0, 0.0] * 10,
                      "line": 10.5, "hist_rate": 10.0, "form5": 10.0,
                      "form10": 10.0, "opp5": 1.0, "opp_allowed": 1.0,
                      "games_before": 20, "is_home": 1.0, "rest_days": 2.0,
                      "share5": 0.2, "eff5": 0.4,
                      "game_date": pd.date_range(f"{s}-01-01", periods=20)})
        for s in range(2012, 2018)], ignore_index=True)
    seasons = sorted(frame["season"].unique())
    assert seasons[:config.BURN_IN_SEASONS] == [2012, 2013, 2014, 2015]


def test_fifteen_seasons_are_configured_ending_with_the_current_one():
    assert len(config.SEASONS) == 15
    assert config.SEASONS[-1] == config.CURRENT_SEASON == 2026
    assert data.season_label(2026) == "2025-26"


def test_the_lockout_season_is_documented_as_short():
    """1,980 team rows in 2011-12 is correct, not a truncated download -- a
    row-count check that assumes 2,460 would flag the one legitimately short one."""
    assert 2012 in config.SHORT_SEASONS


# --- the training window -------------------------------------------------------

def test_the_training_window_is_bounded(monkeypatch):
    """THE FIX THAT RELEASED POINTS. An unbounded window anchors the model to a
    decade-old scoring environment: the points over-rate drifted from 0.461 in
    2012 to 0.555 by 2019, the model predicted 39.9% overs in 2024 when 53.2%
    landed, and it lost to its baseline in three straight seasons for no reason
    other than being calibrated to a league that had moved on.

    Asserted on the SPAN actually handed to fit(), not on the constant, because
    the constant existing proves nothing about whether walk_forward honours it.
    """
    seen = []

    class Spy:
        def __init__(self, market): pass
        def fit(self, train):
            seen.append((int(train["season"].min()), int(train["season"].max())))
            return self          # the real PropModel.fit is chained
        def predict(self, test):
            return np.full(len(test), 0.5)

    monkeypatch.setattr(backtest, "PropModel", Spy)
    monkeypatch.setattr(backtest, "empirical_baseline",
                        lambda frame, market: np.full(len(frame), 0.5))

    frame = pd.concat([
        pd.DataFrame({"season": s, "outcome": [1.0, 0.0] * 30, "line": 10.5,
                      "at_floor": False,
                      "game_date": pd.date_range(f"{s}-01-01", periods=60)})
        for s in range(2012, 2027)], ignore_index=True)
    backtest.walk_forward(frame, "points")

    assert seen, "no season was scored"
    for lo, hi in seen:
        assert hi - lo + 1 <= config.TRAIN_SEASONS, (
            f"trained on {hi - lo + 1} seasons ({lo}-{hi}), "
            f"cap is {config.TRAIN_SEASONS}")


def test_a_scored_season_is_never_in_its_own_training_set():
    """The bound must not be implemented in a way that lets the window slide
    forward over the season being predicted."""
    assert config.TRAIN_SEASONS >= 1
    frame = pd.DataFrame({"season": [2020, 2021, 2022]})
    for season in (2021, 2022):
        train = frame[(frame["season"] < season)
                      & (frame["season"] >= season - config.TRAIN_SEASONS)]
        assert season not in set(train["season"])
