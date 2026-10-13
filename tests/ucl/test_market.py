"""The Champions League board checks every pick against bet365 (ucl/market.py), the
fast locker freezes it (scripts/lock_picks.lock_ucl), and played games with no pick
are published (ucl.publish.unrecorded)."""
import json

import pandas as pd
import pytest

from scripts import lock_picks, sync_ucl_odds
from ucl import market, publish

KICKOFF = "2026-10-14T19:00:00+00:00"


def _match(**kw):
    m = {"id": 77, "kickoff": KICKOFF, "date": "2026-10-14", "home": "Slovan Bratislava",
         "away": "VfB Stuttgart", "p_home": 0.45, "p_draw": 0.27, "p_away": 0.28,
         "pick": "Slovan Bratislava", "p_pick": 0.45, "confidence": 2, "best_pick": False,
         "lockable": True}
    m.update(kw)
    return m


QUOTE = {"odds": {"home": 9.0, "draw": 5.5, "away": 1.30}, "checked_at": "2026-10-13T10:00:00+00:00"}


def test_the_book_leads_and_a_wild_model_pick_is_overruled():
    m = _match()
    market.apply(m, QUOTE, publish._confidence)
    assert m["pick"] == "VfB Stuttgart"
    assert m["model_pick"] == "Slovan Bratislava"
    assert m["disagrees"] is True and m["market_note"]
    assert m["p_model_home"] == 0.45
    assert abs(m["p_home"] + m["p_draw"] + m["p_away"] - 1) < 0.01
    assert m["market"]["pick_odds"] == 1.30


def test_agreeing_model_moves_the_book_only_a_little():
    m = _match(p_home=0.12, p_draw=0.2, p_away=0.68, pick="VfB Stuttgart", p_pick=0.68)
    market.apply(m, QUOTE, publish._confidence)
    book = market.devig(QUOTE["odds"])["away"]
    assert m["disagrees"] is False
    assert abs(m["p_away"] - book) < 0.03


def test_no_price_means_model_only_and_never_a_best_pick():
    m = _match(p_home=0.8, p_draw=0.1, p_away=0.1, pick="Slovan Bratislava", p_pick=0.8, best_pick=True)
    market.apply(m, None, publish._confidence)
    assert m["basis"] == "model only" and m["best_pick"] is False


def test_parse_reads_bet365_match_winner_only():
    rows = [{"fixture": {"id": 5}, "bookmakers": [
        {"id": 6, "bets": [{"id": 1, "values": [{"value": "Home", "odd": "2"}]}]},
        {"id": 8, "bets": [{"id": 5, "values": []},
                           {"id": 1, "values": [{"value": "Home", "odd": "1.5"},
                                                {"value": "Draw", "odd": "4"},
                                                {"value": "Away", "odd": "6"}]}]}]}]
    assert market.parse(rows) == {"5": {"home": 1.5, "draw": 4.0, "away": 6.0}}


def test_sync_pages_and_keeps_the_first_price(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "STORE", tmp_path / "odds.json")
    calls = []

    class Fake:
        def get(self, path, **kw):
            calls.append(kw["page"])
            return [{"fixture": {"id": 5}, "bookmakers": [{"id": 8, "bets": [{"id": 1, "values": [
                {"value": "Home", "odd": "1.5"}, {"value": "Draw", "odd": "4"}, {"value": "Away", "odd": "6"}]}]}]}]
    now = pd.Timestamp("2026-10-13T10:00:00Z").to_pydatetime()
    sync_ucl_odds.main(now=now, client=Fake())
    assert calls == [1, 1]          # match winner, then anytime scorer
    first = json.loads((tmp_path / "odds.json").read_text())["fixtures"]["5"]
    assert first["open"] == first["odds"]
    # inside the recheck interval: no call
    sync_ucl_odds.main(now=now + pd.Timedelta(hours=1), client=Fake())
    assert calls == [1, 1]


@pytest.fixture
def ucl_files(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "PICKS_LOG", tmp_path / "picks_log.json")
    monkeypatch.setattr(publish, "FIXTURES", tmp_path / "fixtures.json")
    monkeypatch.setattr(publish.data, "results_by_id", lambda season=None: {})
    monkeypatch.setattr(lock_picks, "ROOT", tmp_path)
    (tmp_path / "data" / "ucl").mkdir(parents=True)
    return tmp_path


def _board(root, matches):
    (root / "data" / "ucl" / "board.json").write_text(json.dumps({"matches": matches}))


def test_fast_locker_freezes_the_displayed_pick_inside_the_window(ucl_files):
    m = _match()
    market.apply(m, QUOTE, publish._confidence)
    _board(ucl_files, [m])
    now = pd.Timestamp(KICKOFF) - pd.Timedelta(hours=1)
    assert lock_picks.lock_ucl(now, 2.0)
    entry = json.loads((ucl_files / "picks_log.json").read_text())["77"]
    assert entry["pick"] == "VfB Stuttgart" and entry["pick_side"] == "away"
    assert entry["basis"] == "with market" and entry["odds"] == 1.30
    assert entry["tainted"] is False
    # idempotent
    assert lock_picks.lock_ucl(now, 2.0) == []


def test_fast_locker_leaves_distant_and_started_games(ucl_files):
    _board(ucl_files, [_match()])
    assert lock_picks.lock_ucl(pd.Timestamp(KICKOFF) - pd.Timedelta(hours=5), 2.0) == []
    assert lock_picks.lock_ucl(pd.Timestamp(KICKOFF) + pd.Timedelta(minutes=5), 2.0) == []
    assert not (ucl_files / "picks_log.json").exists()


def test_played_games_without_a_pick_are_listed(ucl_files):
    (ucl_files / "fixtures.json").write_text(json.dumps({"fixtures": [
        {"id": 1, "home": "A", "away": "B", "kickoff": "2026-09-09T19:00:00+00:00", "played": True},
        {"id": 2, "home": "C", "away": "D", "kickoff": "2026-09-09T19:00:00+00:00", "played": True},
        {"id": 3, "home": "E", "away": "F", "kickoff": "2026-10-14T19:00:00+00:00", "played": False}]}))
    out = publish.unrecorded({"2": {"pick": "C"}})
    assert [u["id"] for u in out] == ["1"]


def test_record_splits_by_basis():
    log = {"1": {"graded": "correct", "basis": "with market"},
           "2": {"graded": "wrong"}, "_released": {}}
    assert publish.record(log, "with market")["correct"] == 1
    assert publish.record(log, "model only")["wrong"] == 1
    assert publish.record(log)["settled"] == 2


def test_same_pick_but_model_less_sure_is_not_called_overrating():
    m = _match(home="Viking", away="Bayern München", p_home=0.3, p_draw=0.24, p_away=0.46,
               pick="Bayern München", p_pick=0.46)
    market.apply(m, {"odds": {"home": 15.0, "draw": 12.0, "away": 1.10}}, publish._confidence)
    assert m["pick"] == "Bayern München"
    assert m["market_note_kind"] == "less_sure"
    assert "overrates" not in m["market_note"]


def test_every_drawn_club_has_a_pot_and_one_history():
    """A club renamed by the feed (Bayern Munich -> Bayern München) must not be split
    into two clubs, or the current name carries only its recent matches."""
    from ucl import data
    fixtures = json.loads(publish.FIXTURES.read_text(encoding="utf-8"))["fixtures"]
    clubs = {f["home"] for f in fixtures} | {f["away"] for f in fixtures}
    drawn = data.drawn_clubs()
    assert clubs <= set(drawn)
    depth = data.history_depth(data.matches())
    assert depth.get("Bayern München", 0) > 150
    assert "Bayern Munich" not in depth
