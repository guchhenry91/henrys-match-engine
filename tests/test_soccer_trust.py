"""Soccer value verdicts are judged on the model pulled toward bet365
(tracking/trust.soccer_*), fitted on soccer's own graded record."""
import json

import pytest

from tracking import trust, value


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(trust, "SOCCER_DIR", tmp_path)
    monkeypatch.setattr(trust, "_SOCCER", None)
    (tmp_path / "pl").mkdir()
    return tmp_path


def test_rows_use_the_frozen_fair_chance_or_the_pre_match_line(logs):
    (logs / "pl" / "picks_log.json").write_text(json.dumps({
        "a": {"p_pick": 0.6, "book_p_pick": 0.55, "graded": "correct"},
        "b": {"p_pick": 0.6, "open": {"p": 0.5}, "graded": "wrong"},
        "c": {"p_pick": 0.6, "graded": "correct"},                       # no book: skipped
        "d": {"p_pick": 0.6, "book_p_pick": 0.5, "graded": "correct", "void": True},
        "_released": {}}))
    assert trust.soccer_rows("winner") == [(0.6, 0.55, 1.0), (0.6, 0.5, 0.0)]


def test_too_few_graded_uses_the_default(logs):
    assert trust.soccer_weights()["winner"] == {"w": trust.DEFAULT_W, "n": 0, "basis": "default (too few graded)"}


def test_a_model_that_only_adds_noise_earns_no_weight(logs):
    # the book is right (55%), the model wildly over-confident (80%): w should fall to ~0
    rows = {str(i): {"p_pick": 0.8, "book_p_pick": 0.55, "graded": "correct" if i % 20 < 11 else "wrong"}
            for i in range(300)}
    (logs / "pl" / "picks_log.json").write_text(json.dumps(rows))
    w = trust.soccer_weights()["winner"]
    assert w["n"] == 300 and w["w"] <= 0.1


def test_value_is_judged_on_the_blend(logs):
    rows = {str(i): {"p_pick": 0.8, "book_p_pick": 0.55, "graded": "correct" if i % 20 < 11 else "wrong"}
            for i in range(300)}
    (logs / "pl" / "picks_log.json").write_text(json.dumps(rows))
    # raw model 0.70 at bet365 1.80 looks like +26% EV; pulled to the book (~0.55) it is no bet
    p = trust.soccer_value_p(0.70, 0.55, "winner")
    assert p < 0.58
    assert value.assess(p, 0.55, 1.80, True)["value"] is False
    assert trust.soccer_value_p(0.70, None, "winner") == 0.70          # nothing to blend toward


def test_one_sided_markets_are_never_blended_toward_a_vigged_price():
    payload = {"props": {"anytime_touchdown": {"picks": [{"probability": 0.55, "book_p": 0.6897}]},
                         "rushing_yards": {"picks": [{"probability": 0.60, "book_p": 0.50}]}},
               "games": []}
    trust.apply_line_board(payload, {"prop": {"w": 0.0}, "winner": {"w": 0.0}})
    assert payload["props"]["anytime_touchdown"]["picks"][0]["probability"] == 0.55
    assert payload["props"]["rushing_yards"]["picks"][0]["probability"] == 0.50
    rows = trust.rows([{"kind": "prop", "market": "anytime_touchdown", "graded": "correct",
                        "p_pick": 0.5, "book_p": 0.69}], "prop")
    assert rows == []


def test_a_pick_pulled_below_fifty_percent_leaves_a_fifty_percent_board():
    payload = {"games": [], "props": {"rushing_yards": {"picks": [
        {"probability": 0.58, "book_p": 0.47},                     # 0.58 -> 0.47: drops
        {"probability": 0.62, "book_p": 0.55},                     # stays above 50%
        {"probability": 0.15, "book_p": 0.12, "market": "hr"},     # long shot: untouched
        {"probability": 0.58, "book_p": 0.47, "locked": True}]}}}  # frozen: never removed
    trust.apply_line_board(payload, {"prop": {"w": 0.0}, "winner": {"w": 0.0}})
    probs = [p["probability"] for p in payload["props"]["rushing_yards"]["picks"]]
    assert probs == [0.55, 0.12, 0.58]
