"""The rebuilt NBA board's decision rule (nba/forecast_board.py)."""
from nba import forecast_board as fb


def test_combine_leans_on_both_and_is_neutral_at_even():
    w = (0.5, 0.5, 0.0)
    assert abs(fb.combine(0.5, 0.5, w) - 0.5) < 1e-9
    both = fb.combine(0.7, 0.7, w)
    assert 0.65 < both < 0.75
    assert fb.combine(0.7, 0.4, w) < both          # a disagreeing book pulls it down


def _p(name, prob, p_model, game="g1"):
    return {"player": name, "probability": prob, "p_model": p_model, "game_id": game,
            "book_price": 2.0, "book_p": p_model, "book": "bet365"}


def test_select_needs_the_bar_and_the_forecast_on_the_same_side():
    got = fb.select([_p("A", 0.70, 0.66), _p("B", 0.68, 0.42),    # B: the book's pick, not ours
                     _p("C", 0.60, 0.70), _p("D", 0.64, 0.55)])
    assert [p["player"] for p in got] == ["A", "D"]


def test_select_caps_each_game():
    priced = [_p(f"P{i}", 0.70 + i / 100, 0.6) for i in range(5)]
    assert len(fb.select(priced)) == fb.TOP_PER_MARKET


def test_select_keeps_a_short_priced_contender_as_proving_only():
    """70% at 1.30 loses money (needs 1.43): the price check keeps it off the board."""
    short = dict(_p("short", 0.70, 0.70), book_price=1.30, book="bet365", book_p=0.66)   # model backs it
    fair = dict(_p("fair", 0.70, 0.70), book_price=1.60, book="bet365")
    out = fb.select([short, fair])
    # Owner 2026-10-12: a likely pick the model backs is published as a PROVING pick --
    # frozen and graded, never a bet at that price. Only the fair-priced one is value.
    assert sorted(p["player"] for p in out) == ["fair", "short"]
    assert short["pick_kind"] == "proving" and fair["pick_kind"] == "value"
    assert short["price_check"]["reason"] == "price too short"
    assert short["price_check"]["break_even_price"] == 1.43


def test_no_under_on_a_player_whose_minutes_dipped_late_last_season():
    """Opening-night audit, 2026-10-19: such players were under-forecast by 1.7 points
    over four openers -- an under on him is the model's error, not an edge."""
    from nba import forecast_board as fb
    base = {"market": "rebounds", "player": "X", "player_id": 1, "game_id": "g", "line": 5.5,
            "probability": 0.60, "p_model": 0.70, "book_p": 0.57, "book_price": 1.75,
            "season_games_before": 0, "min_dip": 8.0}
    assert fb._dip_under({**base, "side": "under"})
    assert not fb._dip_under({**base, "side": "over"})
    assert not fb._dip_under({**base, "side": "under", "season_games_before": 1})
    assert not fb._dip_under({**base, "side": "under", "min_dip": 2.0})
    assert fb.select([{**base, "side": "under"}]) == []


def _rung(**kw):
    base = {"market": "points", "player": "X", "player_id": 1, "game_id": "g1",
            "tipoff": "2026-10-21T00:00:00Z", "line": 24.5, "milestone": 25, "side": "over",
            "p_model": 0.75, "probability": 0.70, "book_p": 0.66, "book_price": 1.45,
            "avg": 22.0, "pick_kind": "proving", "price_check": {"ev": 0.01}}
    return {**base, **kw}


def test_custom_play_takes_the_highest_milestone_the_model_backs():
    from nba import forecast_board as fb
    out = fb.custom_plays([_rung(line=19.5, milestone=20, p_model=0.85),
                           _rung(line=24.5, milestone=25, p_model=0.72),
                           _rung(line=29.5, milestone=30, p_model=0.40, pick_kind=None)])
    assert [c["milestone"] for c in out["custom"]] == [25] and out["custom"][0]["play"] == "custom"


def test_custom_play_needs_the_bar_and_the_price_check():
    from nba import forecast_board as fb
    assert fb.custom_plays([_rung(p_model=0.65)])["custom"] == []
    assert fb.custom_plays([_rung(pick_kind=None)])["custom"] == []


def test_monster_night_needs_value_and_a_real_big_night():
    from nba import forecast_board as fb
    big = _rung(line=29.5, milestone=30, p_model=0.20, book_price=7.0, pick_kind="value", avg=18.0,
                price_check={"ev": 0.15})
    assert fb.custom_plays([big])["monster"][0]["play"] == "monster"
    assert fb.custom_plays([dict(big, pick_kind="proving")])["monster"] == []      # value only
    assert fb.custom_plays([dict(big, avg=28.0)])["monster"] == []                 # 30 is no monster for a 28-a-game man
    many = [dict(big, player_id=i, game_id=f"g{i}") for i in range(6)]
    assert len(fb.custom_plays(many)["monster"]) == fb.MONSTER_PER_NIGHT


def test_plays_freeze_apart_and_record_apart():
    from datetime import datetime, timezone
    from nba import picks
    p = dict(_rung(), play="custom", team="BOS", opponent="NYK", engine="forecast-v1")
    log = {}
    n = picks.freeze({"plays": {"custom": [p]}}, log, datetime(2026, 10, 20, 23, tzinfo=timezone.utc),
                     lambda t, n: True)
    assert n == 1 and list(log.values())[0]["kind"] == "play"
    e = list(log.values())[0]
    picks.grade(log, {}, {("g1", 1): {"PTS": 27}}, {"g1"})
    assert e["graded"] == "correct"
    assert picks.record(log)["props"]["correct"] == 0           # not in the main record
    assert picks.record_plays(log)["custom"]["points"]["correct"] == 1


def test_custom_play_needs_a_price_worth_betting():
    from nba import forecast_board as fb
    assert fb.custom_plays([_rung(book_price=1.10, p_model=0.95)])["custom"] == []


def test_playoff_correction_only_in_the_playoffs():
    from nba import current, forecast_board as fb
    assert fb.playoff_factor(current.REGULAR, False, "AST") == 1.0
    assert fb.playoff_factor(current.PLAYOFFS, False, "AST") < 0.85
    assert fb.playoff_factor(current.PLAYOFFS, True, "REB") > 1.0


def test_playoff_games_count_for_picks_and_grading():
    from nba import config, current
    assert current.PLAYOFFS in current.COUNTED and current.PLAYIN in current.COUNTED
    assert current._odds_stage(config.PLAYOFFS_START) == current.PLAYOFFS
    assert current._odds_stage("2027-01-10") == current.REGULAR
    assert current._path(current.PLAYOFFS).name.startswith("playoffs_")


def test_proving_picks_are_judged_on_the_published_chance():
    """Review 2026-10-20: judged on the model's own 62%+, they stated 67-70% and landed
    57-65% against an informed book. Now the published (book-led) chance decides."""
    from nba import forecast_board as fb
    assert fb._clears({"pick_kind": "proving", "probability": 0.56, "p_model": 0.58, "book_p": 0.5})
    assert not fb._clears({"pick_kind": "proving", "probability": 0.53, "p_model": 0.80, "book_p": 0.5})


def test_markets_go_live_on_the_new_models_gate(tmp_path, monkeypatch):
    import json
    from nba import publish
    ok = {"main": {"0.62": {"pass": True, "n": 900, "said": 0.64, "landed": 0.66}},
          "balanced_55": {"pass": True}}
    bad = {"main": {"0.62": {"pass": False, "n": 900, "said": 0.70, "landed": 0.60}},
           "balanced_55": {"pass": True}}
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"stats": {"PTS": {"live": ok}, "REB": {"live": bad},
                                         "AST": {"live": ok}, "FG3M": {"live": ok}}}))
    monkeypatch.setattr(publish, "FORECAST_REPORT", rep)
    released, why = publish.released_markets()
    assert released == ["assists", "points", "threes"] and not why["rebounds"]["released"]
    assert "landed 60%" in publish.withheld_caveat(why, ["rebounds"])[0]
    monkeypatch.setattr(publish, "FORECAST_REPORT", tmp_path / "missing.json")
    assert publish.released_markets()[0] == []          # no gate, nothing live


def test_the_live_report_releases_all_four_markets():
    from nba import publish
    assert publish.released_markets()[0] == ["assists", "points", "rebounds", "threes"]
