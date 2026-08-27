"""Odds maths, where a silent bug does not lose accuracy -- it invents money.

Every failure mode here produces a LARGER apparent edge, which is the direction
that gets acted on. A missing price read as zero says the book thinks the outcome
is impossible; a raw implied probability compared against a model probability
shows an edge on almost everything, because the vig is doing the work.
"""
import pytest

from nfl import odds


def test_decimal_prices_convert_to_probabilities():
    assert odds.decimal_to_prob(2.0) == pytest.approx(0.5)
    assert odds.decimal_to_prob(1.25) == pytest.approx(0.8)
    assert odds.decimal_to_prob(5.0) == pytest.approx(0.2)


def test_impossible_prices_are_refused_not_coerced():
    """A zero here would read as 'the book thinks this cannot happen'."""
    for bad in (None, "", "evens", 0, 1.0, 0.5, -3):
        assert odds.decimal_to_prob(bad) is None


def test_devig_removes_the_margin_and_sums_to_one():
    raw = {"home": 0.55, "away": 0.50}          # 105% book
    fair = odds.devig(raw)
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair["home"] > fair["away"]


def test_devig_is_proportional_not_additive():
    """Additive de-vigging takes the same absolute slice off a heavy favourite and
    a big dog, which badly overstates the dog's true chance."""
    raw = {"home": 0.90, "away": 0.15}          # 105%
    fair = odds.devig(raw)
    assert fair["home"] / fair["away"] == pytest.approx(0.90 / 0.15)
    assert fair["away"] == pytest.approx(0.15 / 1.05)


def test_devig_survives_an_empty_or_broken_market():
    assert odds.devig({}) == {}
    assert odds.devig({"home": None, "away": None}) == {}


def test_bet365_is_preferred_when_it_quotes():
    book = odds.pick_bookmaker([{"name": "Pinnacle"}, {"name": "Bet365"},
                                {"name": "Unibet"}])
    assert book["name"] == "Bet365"


def test_a_fallback_book_is_used_when_bet365_is_absent():
    book = odds.pick_bookmaker([{"name": "Unibet"}, {"name": "Pinnacle"}])
    assert book["name"] == "Pinnacle", "should prefer the sharper fallback"


def test_no_bookmakers_means_none():
    assert odds.pick_bookmaker([]) is None
    assert odds.pick_bookmaker(None) is None


def _book(values, name="Bet365", market="Home/Away"):
    return {"name": name, "bets": [{"name": market, "values": values}]}


def test_a_moneyline_is_parsed_and_devigged():
    line = odds.moneyline(_book([{"value": "Home", "odd": "1.80"},
                                 {"value": "Away", "odd": "2.10"}]), "KC", "BUF")
    assert line["book"] == "Bet365"
    assert line["home"] + line["away"] == pytest.approx(1.0)
    assert line["overround"] > 1.0, "the raw prices should show a margin"
    assert line["home"] > line["away"]


def test_a_half_priced_market_is_refused():
    """One side missing cannot be de-vigged, and inventing the other side's price
    from 1-p would hand the model a free edge on the leg that is present."""
    assert odds.moneyline(_book([{"value": "Home", "odd": "1.80"}]), "KC", "BUF") is None


def test_an_unrecognised_market_is_refused():
    assert odds.moneyline(_book([{"value": "Home", "odd": "1.8"},
                                 {"value": "Away", "odd": "2.1"}],
                                market="Total Field Goals"), "KC", "BUF") is None


def test_a_missing_book_is_refused():
    assert odds.moneyline(None, "KC", "BUF") is None


def test_edge_is_the_gap_over_the_fair_price():
    assert odds.edge(0.62, 0.55) == pytest.approx(0.07)
    assert odds.edge(0.50, 0.55) == pytest.approx(-0.05)


def test_value_needs_a_real_gap_not_a_rounding_difference():
    assert odds.value_verdict(0.62, 0.55)[0] == "value"
    assert odds.value_verdict(0.560, 0.550)[0] == "no edge"
    assert odds.value_verdict(0.50, 0.60)[0] == "book favours"


def test_agreement_with_the_book_is_a_real_answer():
    """A board that finds an edge on every game is describing its own error."""
    verdict, gap = odds.value_verdict(0.55, 0.55)
    assert verdict == "no edge" and gap == 0.0
