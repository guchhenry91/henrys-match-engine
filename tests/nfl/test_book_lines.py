"""The model is asked about the BOOKMAKER'S line, not only its own.

WHY. Every NFL prop line used to be the player's own career median, which backup
seasons drag far below what a book quotes -- Parker Washington's receiving line
was 26.5 against recent games of 71, 26, 53, 145 and 115. The model now trains
across a spread of lines so it can be asked about the book's number, and the
board uses the book's line wherever bet365 quotes one.
"""
import pandas as pd
import pytest

from nfl import config, features, odds


def _weeks(values, stat="receiving_yards"):
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


def test_each_game_is_asked_at_every_line_in_the_spread():
    built = features.build(_weeks([40] * 12), "receiving_yards")
    spread = features.augment_lines(built, "receiving_yards")
    assert set(spread["line_mult"]) <= set(config.LINE_MULTIPLIERS)
    assert len(spread) > len(built)


def test_the_outcome_is_resettled_at_each_line():
    """40 yards beats a 20.5 line and loses to an 80.5 one -- same game."""
    built = features.build(_weeks([40] * 12), "receiving_yards")
    spread = features.augment_lines(built, "receiving_yards")
    for _, row in spread.iterrows():
        assert row["outcome"] == float(row["receiving_yards"] > row["line"])


def test_a_spread_line_below_the_floor_is_still_refused():
    built = features.build(_weeks([40] * 12), "receiving_yards")
    spread = features.augment_lines(built, "receiving_yards")
    assert (spread["line"] >= config.MIN_LINE["receiving_yards"]).all()


def test_touchdowns_have_no_line_to_spread():
    built = features.build(_weeks([1] * 12, stat="touchdowns"), "anytime_touchdown")
    assert features.augment_lines(built, "anytime_touchdown").equals(built)


def test_at_line_moves_only_the_line():
    built = features.build(_weeks([40] * 12), "receiving_yards")
    moved = features.at_line(built, 50.5)
    assert (moved["line"] == 50.5).all()
    assert moved.drop(columns="line").equals(built.drop(columns="line"))


@pytest.mark.parametrize("book,ours", [
    ("Kenneth Walker III", "Kenneth Walker"),
    ("Marvin Harrison Jr.", "Marvin Harrison"),
    ("D.J. Moore", "DJ Moore"),
    ("Ja'Marr Chase", "Ja'Marr Chase"),
])
def test_names_join_across_the_two_feeds(book, ours):
    assert odds.norm_name(book) == odds.norm_name(ours)


def test_a_player_is_matched_within_his_own_game_only():
    quotes = {"kenneth walker": {"line": 64.5}, "zach charbonnet": {"line": 22.5}}
    assert odds.match_player(quotes, "Kenneth Walker III") == {"line": 64.5}
    assert odds.match_player(quotes, "Someone Else") is None


# The live shape, copied from the 2026-09-13 probe of TB @ CIN.
BET365 = {"name": "Bet365", "bets": [
    {"id": 328, "name": "Player Rushing Yards", "values": [
        {"value": "Bucky Irving - Over 50.5", "odd": "1.90"},
        {"value": "Bucky Irving - Under 50.5", "odd": "1.90"},
        {"value": "Kenneth Gainwell - Over 21.5", "odd": "1.80"},
        {"value": "Kenneth Gainwell - Under 21.5", "odd": "2.00"},
        {"value": "Baker Mayfield - Over 14.5", "odd": "1.90"}]},     # one side only
    {"id": 47, "name": "Anytime Goal Scorer", "values": [
        {"value": "Bucky Irving", "odd": "2.10"}]},
    {"id": 336, "name": "Player Passing Yards", "values": [
        {"value": "Joe Burrow - Over 274.5", "odd": "1.90"},
        {"value": "Joe Burrow - Under 274.5", "odd": "1.90"}]},
    {"id": 1, "name": "Home/Away", "values": [{"value": "Home", "odd": "1.50"}]}]}


def test_the_live_bet365_shape_parses():
    props = odds.player_props(BET365)
    irving = props["rushing_yards"]["Bucky Irving"]
    assert irving["line"] == 50.5 and irving["over"] == pytest.approx(0.5)
    assert props["passing_yards"]["Joe Burrow"]["line"] == 274.5
    assert props["anytime_touchdown"]["Bucky Irving"]["raw_yes"] == pytest.approx(1 / 2.10, abs=1e-4)


def test_the_over_price_is_devigged():
    g = odds.player_props(BET365)["rushing_yards"]["Kenneth Gainwell"]
    assert g["over"] + g["under"] == pytest.approx(1.0, abs=1e-3)
    assert g["over"] > 0.5            # 1.80 over vs 2.00 under


def test_a_one_sided_yards_quote_is_refused():
    assert "Baker Mayfield" not in odds.player_props(BET365)["rushing_yards"]


def test_the_main_line_wins_over_an_alternate():
    book = {"name": "Bet365", "bets": [{"id": 328, "values": [
        {"value": "A B - Over 30.5", "odd": "1.30"}, {"value": "A B - Under 30.5", "odd": "3.40"},
        {"value": "A B - Over 50.5", "odd": "1.90"}, {"value": "A B - Under 50.5", "odd": "1.90"}]}]}
    assert odds.player_props(book)["rushing_yards"]["A B"]["line"] == 50.5


def test_an_ambiguous_name_is_never_guessed():
    """Two quotes normalising to the same name: refuse, never pick one."""
    quotes = {"a": {"line": 40.5, "_name": "Josh Allen"},
              "b": {"line": 12.5, "_name": "Josh Allen"}}
    assert odds.match_player(quotes, "Josh Allen") is None
