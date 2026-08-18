"""The parlay engine: honest combination and all-or-nothing grading."""
import pandas as pd
import pytest

from leagues import parlays


def _best(upcoming, settled=None):
    return {"upcoming": upcoming, "settled": settled or []}


def _pp(upcoming, settled=None):
    return {"upcoming": upcoming, "settled": settled or []}


def _m(lk, mid, home, away, pick, p, date="2099-08-20T18:00:00+00:00"):
    return {"league_key": lk, "league": lk, "id": mid, "home": home, "away": away,
            "pick": pick, "p_pick": p, "date": date}


def _pr(lk, mid, home, away, player, market, line, p, date="2099-08-20T18:00:00+00:00"):
    return {"league_key": lk, "league": lk, "id": mid, "home": home, "away": away,
            "player": player, "market": market, "line": line, "p_pick": p, "date": date}


def test_combined_probability_is_the_product_of_legs():
    legs = [{"p": 0.8}, {"p": 0.5}, {"p": 0.5}]
    assert parlays._combined(legs) == pytest.approx(0.2)


def test_one_leg_per_match_never_doubles_a_fixture():
    # Two picks from the SAME match must never both appear in a parlay.
    team = [parlays._match_leg(_m("PL", 1, "A", "B", "A", 0.8))]
    prop = [parlays._prop_leg(_pr("PL", 1, "A", "B", "Striker", "goal", "Anytime", 0.6))]
    legs = parlays._independent(team + prop, 2)
    mids = {(l["league_key"], l["mid"]) for l in legs}
    assert len(legs) == 1          # only one leg survives -- same fixture
    assert len(mids) == 1


def test_independent_picks_span_distinct_matches():
    team = [parlays._match_leg(_m("PL", 1, "A", "B", "A", 0.8)),
            parlays._match_leg(_m("PL", 2, "C", "D", "C", 0.7)),
            parlays._match_leg(_m("PL", 3, "E", "F", "E", 0.6))]
    legs = parlays._independent(team, 3)
    assert len(legs) == 3
    assert len({(l["league_key"], l["mid"]) for l in legs}) == 3
    assert [l["p"] for l in legs] == [0.8, 0.7, 0.6]     # highest first


def test_four_fold_takes_one_leg_from_each_league():
    team = [_m("PL", 1, "A", "B", "A", 0.77), _m("LALIGA", 2, "C", "D", "C", 0.73),
            _m("BUNDESLIGA", 3, "E", "F", "E", 0.69), _m("LIGUE1", 4, "G", "H", "G", 0.68)]
    legs = parlays._one_per_league([parlays._match_leg(m) for m in team], [])
    assert {l["league_key"] for l in legs} == {"PL", "LALIGA", "BUNDESLIGA", "LIGUE1"}


def test_pl_only_section_contains_only_premier_league_legs():
    best = _best([_m("PL", 1, "Arsenal", "Coventry", "Arsenal", 0.77),
                  _m("LALIGA", 2, "Real", "Sociedad", "Real", 0.68)])
    pp = _pp([_pr("PL", 3, "City", "Newcastle", "Haaland", "shots", "2+ shot attempts", 0.77),
              _pr("PL", 4, "Villa", "Wolves", "Watkins", "goal", "Anytime", 0.45)])
    out = parlays.build_parlays(best, pp, "/tmp/nonexistent_parlay_log_test.json",
                                now=pd.Timestamp("2099-01-01T00:00:00+00:00"))
    pl_sec = next(s for s in out["sections"] if s["title"] == "Premier League only")
    for pa in pl_sec["parlays"]:
        for leg in pa["legs"]:
            assert leg["league_key"] == "PL"


def test_parlay_locks_once_and_freezes_probabilities():
    legs = [{"id": "PL#1#w", "selection": "A to win", "p": 0.8, "date": "2099-08-20T18:00:00+00:00"},
            {"id": "PL#2#w", "selection": "C to win", "p": 0.6, "date": "2099-08-20T18:00:00+00:00"}]
    log = {}
    now = pd.Timestamp("2099-08-19T18:00:00+00:00")     # a day before -> not tainted
    e1 = parlays.lock_parlay(log, legs, now=now)
    assert e1["tainted"] is False
    assert e1["combined"] == pytest.approx(0.48)
    # A second call, even with moved probabilities, must not overwrite.
    legs2 = [dict(legs[0], p=0.99), dict(legs[1], p=0.99)]
    e2 = parlays.lock_parlay(log, legs2, now=now)
    assert e2["combined"] == pytest.approx(0.48)         # frozen, not re-chosen


def test_parlay_first_locked_after_kickoff_is_tainted():
    legs = [{"id": "PL#1#w", "selection": "A", "p": 0.8, "date": "2099-08-20T18:00:00+00:00"},
            {"id": "PL#2#w", "selection": "C", "p": 0.6, "date": "2099-08-20T20:00:00+00:00"}]
    log = {}
    after = pd.Timestamp("2099-08-20T19:00:00+00:00")    # past the EARLIEST kickoff
    e = parlays.lock_parlay(log, legs, now=after)
    assert e["tainted"] is True
    assert parlays.grade_parlay(e, {"PL#1#w": "correct", "PL#2#w": "correct"}) == "void"


def test_grade_is_all_or_nothing():
    entry = {"tainted": False, "legs": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
    assert parlays.grade_parlay(entry, {"a": "correct", "b": "correct", "c": "correct"}) == "correct"
    assert parlays.grade_parlay(entry, {"a": "correct", "b": "wrong", "c": "correct"}) == "wrong"
    assert parlays.grade_parlay(entry, {"a": "correct", "b": "correct"}) == "pending"   # c unsettled
    assert parlays.grade_parlay(entry, {"a": "correct", "b": "void", "c": "correct"}) == "void"


def test_ungradeable_prop_legs_are_never_stacked(tmp_path):
    """Bundesliga player picks publish gradeable=false (no shot feed) and never
    enter the settled board, so a parlay leg from one could never grade -- it
    would sit pending forever. Such legs must be excluded from every parlay."""
    best = _best([_m("PL", 1, "A", "B", "A", 0.77), _m("LALIGA", 2, "C", "D", "C", 0.73),
                  _m("LIGUE1", 3, "E", "F", "E", 0.69)])
    pp = _pp([
        {**_pr("BUNDESLIGA", 4, "G", "H", "Kane", "shots", "2+ shot attempts", 0.90), "gradeable": False},
        {**_pr("PL", 5, "I", "J", "Haaland", "shots", "2+ shot attempts", 0.77), "gradeable": True}])
    out = parlays.build_parlays(best, pp, tmp_path / "log.json",
                                now=pd.Timestamp("2099-01-01T00:00:00+00:00"))
    sels = [l["selection"] for s in out["sections"] for pa in s["parlays"] for l in pa["legs"]]
    assert not any("Kane" in x for x in sels)        # ungradeable BL prop excluded
    assert any("Haaland" in x for x in sels)         # gradeable PL prop kept


def test_props_treble_prefers_three_distinct_markets(tmp_path):
    best = _best([])
    pp = _pp([{**_pr("PL", 1, "A", "B", "P1", "shots", "2+ shot attempts", 0.80), "gradeable": True},
              {**_pr("PL", 2, "C", "D", "P2", "shots", "2+ shot attempts", 0.79), "gradeable": True},
              {**_pr("LALIGA", 3, "E", "F", "P3", "sot", "1+ shot on target", 0.75), "gradeable": True},
              {**_pr("LIGUE1", 4, "G", "H", "P4", "goal", "Anytime", 0.55), "gradeable": True}])
    out = parlays.build_parlays(best, pp, tmp_path / "log.json",
                                now=pd.Timestamp("2099-01-01T00:00:00+00:00"))
    treble = next(pa for s in out["sections"] for pa in s["parlays"] if pa["tier"] == "Props treble")
    tags = [l["tag"] for l in treble["legs"]]
    assert sorted(tags) == ["a", "g", "o"]           # attempt, goal, on-target -- all distinct


def test_settled_legs_grade_the_parlay_end_to_end():
    # A locked parlay whose both legs settled correct -> the parlay is correct.
    best = _best(
        upcoming=[],
        settled=[{"league_key": "PL", "id": 1, "graded": "correct"},
                 {"league_key": "PL", "id": 2, "graded": "correct"}])
    pp = _pp([])
    log = {parlays._parlay_key([{"id": "PL#1#w"}, {"id": "PL#2#w"}]): {
        "legs": [{"id": "PL#1#w"}, {"id": "PL#2#w"}], "combined": 0.48,
        "tainted": False, "earliest_kickoff": "2099-08-20T18:00:00+00:00"}}
    import leagues.picks as pk
    # inject the log by monkeypatching load_log via a temp file
    import json, tempfile, os
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump(log, open(path, "w"))
    out = parlays.build_parlays(best, pp, path, now=pd.Timestamp("2099-09-01T00:00:00+00:00"))
    os.remove(path)
    assert out["record"]["correct"] == 1
    assert out["record"]["total"] == 1


# --- leg window --------------------------------------------------------------
# Added after the board built a four-fold running 19 to 29 August: not placeable,
# and it froze "Dortmund to win" eleven days before kickoff because a parlay locks
# on its EARLIEST leg.

def _leg(lid, date, p=0.7):
    return {"id": lid, "selection": f"{lid} to win", "match": lid, "league": "PL",
            "league_key": "PL", "tag": "w", "p": p, "date": date}


def test_legs_beyond_the_window_are_dropped():
    now = pd.Timestamp("2026-08-18T04:00:00Z")
    team = [_leg("a", "2026-08-19T19:00:00Z"), _leg("b", "2026-08-21T19:00:00Z"),
            _leg("c", "2026-08-29T16:30:00Z")]      # 10 days after the anchor
    kept, _ = parlays._within_window(team, [], now)
    assert [l["id"] for l in kept] == ["a", "b"]


def test_window_is_anchored_on_the_next_fixture_not_a_past_one():
    """A fixture that has already kicked off must not anchor the window, or the
    board trails a round behind."""
    now = pd.Timestamp("2026-08-22T12:00:00Z")
    team = [_leg("old", "2026-08-19T19:00:00Z"),     # already played
            _leg("new", "2026-08-23T13:00:00Z"),
            _leg("far", "2026-08-29T16:30:00Z")]
    kept, _ = parlays._within_window(team, [], now)
    assert [l["id"] for l in kept] == ["old", "new"]   # anchor is 'new', not 'old'


def test_props_and_match_legs_share_one_anchor():
    """Both lists are filtered against the SAME cutoff, so a mixed parlay cannot
    straddle two rounds."""
    now = pd.Timestamp("2026-08-18T04:00:00Z")
    team = [_leg("t1", "2026-08-19T19:00:00Z")]
    prop = [_leg("p1", "2026-08-20T19:00:00Z"), _leg("p2", "2026-08-26T19:00:00Z")]
    kept_t, kept_p = parlays._within_window(team, prop, now)
    assert [l["id"] for l in kept_t] == ["t1"]
    assert [l["id"] for l in kept_p] == ["p1"]


def test_window_covers_a_normal_weekend_round():
    """Friday night to Monday night must survive intact -- the constraint exists
    to stop nine-day accas, not to break ordinary ones."""
    now = pd.Timestamp("2026-08-20T12:00:00Z")
    team = [_leg("fri", "2026-08-21T19:00:00Z"), _leg("sat", "2026-08-22T14:00:00Z"),
            _leg("sun", "2026-08-23T15:30:00Z"), _leg("mon", "2026-08-24T19:00:00Z")]
    kept, _ = parlays._within_window(team, [], now)
    assert len(kept) == 4


def test_empty_input_is_safe():
    assert parlays._within_window([], [], pd.Timestamp("2026-08-18T04:00:00Z")) == ([], [])
