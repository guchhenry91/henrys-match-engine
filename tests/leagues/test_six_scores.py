"""The 6 Scores board.

Correct score is the one market where an unhonest presentation would be easiest
and most tempting, so these lock down the two things that keep it truthful: it
never invents a selection, and it always publishes its own measured hit rate
next to the picks.
"""
import json

import pandas as pd
import pytest

from leagues import six_scores


def _match(mid, home, away, score="1-1", pct=13.0, agrees=False, date="2026-09-12T14:00:00+00:00"):
    return {"id": mid, "date": date, "home": home, "away": away,
            "prediction": {"pick": home, "provisional": True,
                           "top_scores": [{"score": score, "pct": pct, "agrees_with_pick": agrees},
                                          {"score": "2-1", "pct": 9.9},
                                          {"score": "1-0", "pct": 8.8}]}}


@pytest.fixture
def sel(tmp_path, monkeypatch):
    def _write(fixtures, matchweek=1):
        p = tmp_path / "six_scores.json"
        p.write_text(json.dumps({"fixtures": fixtures, "matchweek": matchweek,
                                 "_verified_on": "2026-09-10"}), encoding="utf-8")
        monkeypatch.setattr(six_scores, "SELECTION", p)
    return _write


NOW = pd.Timestamp("2026-09-10T12:00:00Z")


def test_board_is_empty_when_bet365_has_not_published(sel):
    """The board must never invent six fixtures. An empty selection publishes an
    empty board, which is the truth; a plausible guess would produce a scoreboard
    measuring nothing and nobody could tell."""
    sel([])
    out = six_scores.build({"matches": [_match(1, "Arsenal", "Coventry")]}, {}, NOW)
    assert out["picks"] == []
    assert out["missing_fixtures"] == []


def test_board_uses_the_selection_in_bet365_order(sel):
    sel(["Everton|Crystal Palace", "Arsenal|Coventry"])
    payload = {"matches": [_match(1, "Arsenal", "Coventry"),
                           _match(2, "Everton", "Crystal Palace")]}
    out = six_scores.build(payload, {}, NOW)
    assert [(p["home"], p["away"]) for p in out["picks"]] == [
        ("Everton", "Crystal Palace"), ("Arsenal", "Coventry")]


def test_a_fixture_not_in_this_matchweek_is_reported_not_silently_dropped(sel):
    sel(["Arsenal|Coventry", "Chelsea|Fulham"])
    out = six_scores.build({"matches": [_match(1, "Arsenal", "Coventry")]}, {}, NOW)
    assert len(out["picks"]) == 1
    assert out["missing_fixtures"] == ["Chelsea|Fulham"]


def test_scoreline_is_grid_mode_with_its_alternatives(sel):
    sel(["Arsenal|Coventry"])
    out = six_scores.build({"matches": [_match(1, "Arsenal", "Coventry",
                                               score="2-0", pct=13.9)]}, {}, NOW)
    p = out["picks"][0]
    assert p["score"] == "2-0" and p["score_pct"] == 13.9
    assert [a["score"] for a in p["alternatives"]] == ["2-1", "1-0"]


def test_disagreement_with_the_match_pick_is_flagged(sel):
    """Grid mode answers a different question from the 1X2 pick and often
    disagrees. The card must say so rather than look self-contradictory."""
    sel(["Arsenal|Coventry"])
    out = six_scores.build({"matches": [_match(1, "Arsenal", "Coventry", agrees=False)]},
                           {}, NOW)
    assert out["picks"][0]["agrees_with_match_pick"] is False


def test_measured_expectation_ships_with_every_board(sel):
    """The honesty guarantee: the hit rate and the always-1-1 baseline travel
    WITH the picks, so the claim and its evidence cannot be separated."""
    sel([])
    out = six_scores.build({"matches": []}, {}, NOW)
    assert out["hit_rate_pct"] == six_scores.PL_HIT_RATE_PCT
    assert out["baseline_pct"] == six_scores.PL_BASELINE_PCT
    assert out["expected_of_six"] == pytest.approx(6 * six_scores.PL_HIT_RATE_PCT / 100, abs=0.01)


def test_the_board_does_not_claim_an_edge_it_does_not_have():
    """PL correct score is BELOW the always-1-1 baseline. If a future refit ever
    flips that, this test fails and the section's copy needs rewriting rather
    than quietly becoming out of date."""
    assert six_scores.PL_HIT_RATE_PCT < six_scores.PL_BASELINE_PCT


def test_settled_scorelines_grade_on_exact_match(sel):
    sel([])
    log = {"2026:six:7": {"pick": "2-1"}}
    payload = {"matches": [], "season": [
        {"id": 7, "home": "Arsenal", "away": "Coventry", "date": "2026-09-12",
         "result": {"home_goals": 2, "away_goals": 1}},
        {"id": 8, "home": "Everton", "away": "Fulham", "date": "2026-09-12",
         "result": {"home_goals": 0, "away_goals": 0}}]}
    out = six_scores.build(payload, log, NOW)
    assert out["record"] == {"correct": 1, "total": 1}     # id 8 was never predicted
    assert out["settled"][0]["graded"] == "correct"


def test_a_missing_selection_file_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(six_scores, "SELECTION", tmp_path / "absent.json")
    assert six_scores.build({"matches": []}, {}, NOW)["picks"] == []
