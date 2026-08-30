import leagues.publish as publish

_EMPTY_BEST = {"record": {"correct": 0, "wrong": 0}, "upcoming": [], "settled": [],
               "_incomplete": []}
_EMPTY_PLAYERS = {"record": {"correct": 0, "wrong": 0}, "upcoming": [], "settled": [],
                  "record_by_market": {}, "min_probability": {}, "_incomplete": []}

_STUB = lambda lg: {"league": lg, "matches": [], "table": [],
                    "missing_squads": [], "data_warnings": []}


def test_main_writes_one_atomic_file_per_league(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)
    monkeypatch.setattr(publish, "build", _STUB)
    monkeypatch.setattr(publish, "build_best_picks", lambda: _EMPTY_BEST)
    monkeypatch.setattr(publish, "build_player_picks", lambda: _EMPTY_PLAYERS)
    publish.main([])                       # no arg -> all leagues
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    # DERIVED FROM THE CONFIG, not restated. This list was written out by hand and
    # adding Serie A broke it -- the test was pinning the number of leagues rather
    # than the property, which is "one file per league plus the cross-league
    # boards". A sixth league should not need this line edited.
    cross = ["best.json", "parlays.json", "player_picks.json",
             "record_history.json",
             # the bet365 6 Scores board, written from the PL payload
             "six_scores.json"]
    assert written == sorted(list(publish.FILE_FOR.values()) + cross)
    assert not list(tmp_path.glob("*.tmp"))          # no leftover temp files


def test_one_league_failing_does_not_block_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)

    def flaky(lg):
        if lg == "LALIGA":
            raise RuntimeError("simulated fetch failure")
        return _STUB(lg)

    monkeypatch.setattr(publish, "build", flaky)
    monkeypatch.setattr(publish, "build_best_picks", lambda: _EMPTY_BEST)
    monkeypatch.setattr(publish, "build_player_picks", lambda: _EMPTY_PLAYERS)
    # Seed a PRE-EXISTING laliga.json. Asserting against an empty directory could
    # not tell "correctly skipped" from "silently left stale", which is the actual
    # hazard -- an aborted league leaving last week's file for the gate to pass.
    (tmp_path / "laliga.json").write_text('{"league": "STALE"}', encoding="utf-8")
    import pytest
    # One league of however many are configured fails; the rest still publish.
    expected = rf"{len(publish.FILE_FOR) - 1}/{len(publish.FILE_FOR)}"
    with pytest.raises(RuntimeError, match=expected):
        publish.main([])                   # partial files stay local; no deployment
    written = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert "pl" in written and "bundesliga" in written and "ligue1" in written
    # the failing league's file is untouched, NOT overwritten with partial data
    import json as _json
    assert _json.loads((tmp_path / "laliga.json").read_text())["league"] == "STALE"
    assert not (tmp_path / "best.json").exists()
    assert not (tmp_path / "player_picks.json").exists()


def test_single_league_arg_writes_only_that_file(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "OUT", tmp_path)
    monkeypatch.setattr(publish, "build", _STUB)
    monkeypatch.setattr(publish, "build_best_picks", lambda: _EMPTY_BEST)
    monkeypatch.setattr(publish, "build_player_picks", lambda: _EMPTY_PLAYERS)
    publish.main(["PL"])                    # quick-iteration path
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["pl.json"]


def test_actual_standings_points_order_and_preseason():
    import pandas as pd
    from leagues import publish
    played = pd.DataFrame([
        {"home": "A", "away": "B", "home_goals": 2, "away_goals": 0},   # A win
        {"home": "C", "away": "A", "home_goals": 1, "away_goals": 1},   # draw
    ])
    st = publish.actual_standings(played, ["A", "B", "C", "D"])
    by = {r["team"]: r for r in st}
    assert by["A"]["points"] == 4 and by["A"]["played"] == 2 and by["A"]["gd"] == 2
    assert by["C"]["points"] == 1 and by["B"]["points"] == 0
    assert by["D"]["played"] == 0 and by["D"]["points"] == 0        # never played
    assert st[0]["team"] == "A"                                     # most points first
    # pre-season: everyone zero -> alphabetical, no crash on empty played
    empty = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals"])
    st0 = publish.actual_standings(empty, ["Z", "A", "M"])
    assert [r["team"] for r in st0] == ["A", "M", "Z"]
    assert all(r["points"] == 0 and r["played"] == 0 for r in st0)
