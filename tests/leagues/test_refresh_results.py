"""The fast results path must update what a score changes and nothing else.

It exists because grading and the table needed an eighteen-minute model refit,
and GitHub starts 4-8 scheduled runs a day against the 56 the crons ask for. That
left the boards four hours stale on 2026-08-29 with about fifteen finished games
missing.

Its whole safety argument is that it CANNOT invent a pick: it reads the frozen log
and applies scores. These tests hold it to that.
"""
import json

import pandas as pd
import pytest

from scripts import refresh_results


def _fixtures(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_results, "OUT", tmp_path / "data")
    monkeypatch.setattr(refresh_results, "PICKS_DIR", tmp_path / "raw")
    (tmp_path / "data").mkdir()
    payload = {
        "league": "Premier League",
        "record": {"correct": 0, "wrong": 0, "total": 0, "void": 0, "pending": 1},
        "standings": [],
        "table": [{"team": "Everton"}, {"team": "Crystal Palace"},
                  {"team": "Bayern"}, {"team": "Stuttgart"}],
        "matches": [{"id": 2, "home": "Everton", "away": "Crystal Palace"},
                    {"id": 9, "home": "Bayern", "away": "Stuttgart"}],
        # Things a finished match must NOT change.
        "six_scores": ["untouched"], "backtest": {"rps": 0.19},
    }
    (tmp_path / "data" / "pl.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _log(tmp_path, entries):
    d = tmp_path / "raw" / "pl"
    d.mkdir(parents=True, exist_ok=True)
    (d / "picks_log.json").write_text(json.dumps(entries), encoding="utf-8")


PLAYED = [{"match_id": 2, "date": "2026-08-22T14:00:00+00:00", "home": "Everton",
           "away": "Crystal Palace", "home_goals": 2, "away_goals": 0,
           "played": True}]


def _patch_feed(monkeypatch, rows):
    monkeypatch.setattr(refresh_results.fixtures, "fetch_fixtures",
                        lambda league: _fixtures(rows))


def test_it_grades_a_frozen_pick_against_the_score(board, monkeypatch):
    _log(board, {"2026:2": {"pick": "Everton", "confidence": 4,
                            "kickoff": "2026-08-22T14:00:00+00:00",
                            "tainted": False}})
    _patch_feed(monkeypatch, PLAYED)
    info = refresh_results.refresh_league("PL")
    assert info["newly_graded"] == 1
    out = json.loads((board / "data" / "pl.json").read_text())
    assert out["record"]["correct"] == 1


def test_a_losing_pick_grades_wrong(board, monkeypatch):
    _log(board, {"2026:2": {"pick": "Crystal Palace", "confidence": 3,
                            "kickoff": "2026-08-22T14:00:00+00:00",
                            "tainted": False}})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    out = json.loads((board / "data" / "pl.json").read_text())
    assert out["record"]["wrong"] == 1


def test_a_played_fixture_with_no_frozen_pick_is_unrecorded_never_a_loss(
        board, monkeypatch):
    """THE RULE THIS MUST NOT BREAK. A fixture nobody froze a pick for is absent
    from the record and listed as unrecorded -- inventing a loss for it would be
    exactly the fabrication the 0-18 incident taught."""
    _log(board, {})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    out = json.loads((board / "data" / "pl.json").read_text())
    assert out["record"]["wrong"] == 0 and out["record"]["correct"] == 0
    assert len(out["unrecorded"]) == 1
    assert out["unrecorded"][0]["home"] == "Everton"


def test_it_never_writes_a_pick(board, monkeypatch):
    """It grades; it does not lock. An empty log must stay empty."""
    _log(board, {})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    log = json.loads((board / "raw" / "pl" / "picks_log.json").read_text())
    assert log == {}


def test_a_tainted_pick_stays_void(board, monkeypatch):
    _log(board, {"2026:2": {"pick": "Everton", "confidence": 4,
                            "kickoff": "2026-08-22T14:00:00+00:00",
                            "tainted": True}})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    out = json.loads((board / "data" / "pl.json").read_text())
    assert out["record"]["void"] == 1
    assert out["record"]["correct"] == 0 and out["record"]["wrong"] == 0


def test_a_finished_fixture_leaves_the_upcoming_list(board, monkeypatch):
    _log(board, {"2026:2": {"pick": "Everton", "confidence": 4,
                            "kickoff": "2026-08-22T14:00:00+00:00",
                            "tainted": False}})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    out = json.loads((board / "data" / "pl.json").read_text())
    assert [m["id"] for m in out["matches"]] == [9]


def test_the_standings_are_redrawn(board, monkeypatch):
    _log(board, {})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    out = json.loads((board / "data" / "pl.json").read_text())
    ev = next(r for r in out["standings"] if r["team"] == "Everton")
    assert ev["played"] == 1 and ev["won"] == 1 and ev["points"] == 3


def test_it_touches_nothing_the_score_did_not_change(board, monkeypatch):
    """Predictions, props and the gate report are the model's, not this script's."""
    _log(board, {})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    out = json.loads((board / "data" / "pl.json").read_text())
    assert out["six_scores"] == ["untouched"]
    assert out["backtest"] == {"rps": 0.19}
    assert out["league"] == "Premier League"


def test_pending_picks_stay_in_the_record(board, monkeypatch):
    """A frozen pick for a fixture not yet played must still count as pending,
    not silently vanish because it is not in the played frame."""
    _log(board, {"2026:2": {"pick": "Everton", "confidence": 4,
                            "kickoff": "2026-08-22T14:00:00+00:00",
                            "tainted": False},
                 "2026:9": {"pick": "Bayern", "confidence": 3,
                            "kickoff": "2026-09-05T14:00:00+00:00",
                            "tainted": False}})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    out = json.loads((board / "data" / "pl.json").read_text())
    assert out["record"]["correct"] == 1
    assert out["record"]["pending"] == 1


def test_regrading_is_stable(board, monkeypatch):
    """Running twice must not double-count or change a verdict."""
    _log(board, {"2026:2": {"pick": "Everton", "confidence": 4,
                            "kickoff": "2026-08-22T14:00:00+00:00",
                            "tainted": False}})
    _patch_feed(monkeypatch, PLAYED)
    refresh_results.refresh_league("PL")
    second = refresh_results.refresh_league("PL")
    out = json.loads((board / "data" / "pl.json").read_text())
    assert second["newly_graded"] == 0
    assert out["record"]["correct"] == 1 and out["record"]["total"] == 1


def test_a_missing_board_is_skipped_not_crashed(board, monkeypatch):
    _patch_feed(monkeypatch, PLAYED)
    assert refresh_results.refresh_league("LALIGA")["skipped"] == "no published board"


def test_one_league_failing_does_not_stop_the_others(board, monkeypatch, capsys):
    """A feed outage in one league must not hold back three working ones."""
    def flaky(league):
        if league == "PL":
            raise RuntimeError("feed down")
        return _fixtures(PLAYED)
    monkeypatch.setattr(refresh_results.fixtures, "fetch_fixtures", flaky)
    refresh_results.main(["PL", "LALIGA"])
    out = capsys.readouterr().out
    assert "PL: FAILED" in out
    assert "newly graded" in out or "no published board" in out


def test_it_refuses_to_regress_the_table_on_a_stale_snapshot(board, monkeypatch):
    """fetch_fixtures falls back to a snapshot when the feed times out and returns
    it with NO flag saying so -- 5 hours old in the run that prompted this guard.
    A season's played count only goes up, so a decrease proves stale input."""
    payload = json.loads((board / "data" / "pl.json").read_text())
    payload["standings"] = [
        {"team": "Everton", "played": 3, "won": 3, "drawn": 0, "lost": 0,
         "gf": 6, "ga": 0, "gd": 6, "points": 9},
        {"team": "Crystal Palace", "played": 3, "won": 0, "drawn": 0, "lost": 3,
         "gf": 0, "ga": 6, "gd": -6, "points": 0}]
    (board / "data" / "pl.json").write_text(json.dumps(payload), encoding="utf-8")
    _log(board, {})
    _patch_feed(monkeypatch, PLAYED)          # the snapshot knows only ONE match
    info = refresh_results.refresh_league("PL")
    assert "refusing to regress" in info["skipped"]
    after = json.loads((board / "data" / "pl.json").read_text())
    assert sum(r["played"] for r in after["standings"]) // 2 == 3


def test_an_equal_count_still_updates(board, monkeypatch):
    """Equal is not a regression -- a re-run on the same data must still refresh
    grades, or a league would freeze after its first refusal."""
    payload = json.loads((board / "data" / "pl.json").read_text())
    payload["standings"] = [
        {"team": "Everton", "played": 1, "won": 1, "drawn": 0, "lost": 0,
         "gf": 2, "ga": 0, "gd": 2, "points": 3},
        {"team": "Crystal Palace", "played": 1, "won": 0, "drawn": 0, "lost": 1,
         "gf": 0, "ga": 2, "gd": -2, "points": 0}]
    (board / "data" / "pl.json").write_text(json.dumps(payload), encoding="utf-8")
    _log(board, {"2026:2": {"pick": "Everton", "confidence": 4,
                            "kickoff": "2026-08-22T14:00:00+00:00",
                            "tainted": False}})
    _patch_feed(monkeypatch, PLAYED)
    info = refresh_results.refresh_league("PL")
    assert not info.get("skipped")
    assert json.loads((board / "data" / "pl.json").read_text())["record"]["correct"] == 1
