"""Gap-fill NFL lines from The Odds API: at most two checks a game, bet365 first,
and the model asked about the book's number wherever any book quotes one."""
from datetime import datetime, timezone

import pytest

from nfl import book_lines as bl
from scripts import sync_nfl_book_lines as sync

UTC = timezone.utc


def _t(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# --- when to spend -----------------------------------------------------------------

def test_next_scheduled_run_follows_the_workflow_crons():
    assert bl.next_scheduled_run(_t("2026-09-27T10:00")) == _t("2026-09-27T16:00")
    assert bl.next_scheduled_run(_t("2026-09-27T16:05")) == _t("2026-09-28T09:00")
    # Monday 16:05 -> Tuesday 08:30 (the backtest run)
    assert bl.next_scheduled_run(_t("2026-09-28T16:05")) == _t("2026-09-29T08:30")


def test_a_sunday_1pm_game_is_checked_on_saturday_then_at_the_last_run():
    kick = "2026-09-27T17:00:00+00:00"
    assert bl.due([], kick, _t("2026-09-24T09:05")) is None           # 80h out
    assert bl.due([], kick, _t("2026-09-26T16:05")) == "board"        # 25h out
    checks = ["2026-09-26T16:05:00+00:00"]
    assert bl.due(checks, kick, _t("2026-09-27T09:05")) is None       # not the last run
    assert bl.due(checks, kick, _t("2026-09-27T16:05")) == "lock"     # last run before
    assert bl.due(checks + ["2026-09-27T16:05:00+00:00"], kick,
                  _t("2026-09-27T16:20")) is None                     # never a third


def test_sunday_night_football_gets_its_lock_check_at_16_utc():
    kick = "2026-09-28T00:20:00+00:00"
    assert bl.due([], kick, _t("2026-09-27T09:05")) == "board"
    assert bl.due(["2026-09-27T09:05:00+00:00"], kick, _t("2026-09-27T16:05")) == "lock"


def test_nothing_is_spent_on_a_game_about_to_start():
    assert bl.due([], "2026-09-27T17:00:00+00:00", _t("2026-09-27T16:40")) is None


# --- parsing ------------------------------------------------------------------------

def _book(key, market, rows):
    return {"key": key, "markets": [{"key": market, "outcomes": [
        {"name": side, "description": name, "point": pt, "price": price}
        for name, side, pt, price in rows]}]}


def test_parse_devigs_and_prefers_pinnacle_then_draftkings():
    payload = {"bookmakers": [
        _book("draftkings", "player_reception_yds", [
            ("Khalil Shakir", "Over", 48.5, 1.87), ("Khalil Shakir", "Under", 48.5, 1.95),
            ("Keenan Allen", "Over", 55.5, 1.90), ("Keenan Allen", "Under", 55.5, 1.90)]),
        _book("pinnacle", "player_reception_yds", [
            ("Khalil Shakir", "Over", 47.5, 1.93), ("Khalil Shakir", "Under", 47.5, 1.93)]),
    ]}
    got = bl.parse_event(payload)["receiving_yards"]
    assert got["Khalil Shakir"]["line"] == 47.5 and got["Khalil Shakir"]["source"] == "pinnacle"
    assert got["Keenan Allen"]["source"] == "draftkings"
    assert got["Keenan Allen"]["over"] == pytest.approx(0.5)


def test_a_one_sided_quote_is_refused():
    payload = {"bookmakers": [_book("fanduel", "player_rush_yds",
                                    [("James Cook", "Over", 60.5, 1.9)])]}
    assert bl.parse_event(payload) == {}


def test_bet365_is_never_overwritten_by_a_gap_fill_book():
    bet365 = {"BUF|LAC": {"rushing_yards": {"James Cook": {"_name": "James Cook", "line": 62.5}}}}
    extra = {"BUF|LAC": {"rushing_yards": {
                "James Cook III": {"_name": "James Cook III", "line": 60.5, "source": "pinnacle"},
                "Ty Johnson": {"_name": "Ty Johnson", "line": 12.5, "source": "pinnacle"}},
             "receiving_yards": {"Khalil Shakir": {"_name": "Khalil Shakir", "line": 47.5}}}}
    merged = bl.merge(bet365, extra)
    rush = merged["BUF|LAC"]["rushing_yards"]
    assert rush["James Cook"]["line"] == 62.5 and "James Cook III" not in rush
    assert "Ty Johnson" in rush and "Khalil Shakir" in merged["BUF|LAC"]["receiving_yards"]


# --- the sync -----------------------------------------------------------------------

class FakeClient:
    def __init__(self, events, payload):
        self.events, self.payload, self.paid = events, payload, []

    def get(self, path, *, sport, purpose, est=None, **params):
        if path.endswith("/events"):
            return self.events
        self.paid.append((path, est, params))
        return self.payload


def test_sync_spends_only_on_due_games_and_asks_three_books_three_markets():
    events = [
        {"id": "e1", "home_team": "Buffalo Bills", "away_team": "Los Angeles Chargers",
         "commence_time": "2026-09-27T17:00:00Z"},
        {"id": "e2", "home_team": "Denver Broncos", "away_team": "Los Angeles Rams",
         "commence_time": "2026-10-05T00:15:00Z"},                       # too far out
    ]
    payload = {"bookmakers": [_book("pinnacle", "player_reception_yds", [
        ("Khalil Shakir", "Over", 47.5, 1.93), ("Khalil Shakir", "Under", 47.5, 1.93)])]}
    client = FakeClient(events, payload)
    store = sync.run(client, now=_t("2026-09-26T16:05"), store={})
    assert len(client.paid) == 1
    path, est, params = client.paid[0]
    assert "e1" in path and est == 3
    assert params["bookmakers"] == "pinnacle,draftkings,fanduel"
    assert set(params["markets"].split(",")) == {"player_pass_yds", "player_rush_yds",
                                                 "player_reception_yds"}
    game = store["games"]["BUF|LAC"]
    assert game["checks"] and "Khalil Shakir" in game["props"]["receiving_yards"]
    # The very next run, same afternoon: nothing new is due.
    sync.run(client, now=_t("2026-09-26T16:30"), store=store)
    assert len(client.paid) == 1


def test_publish_merges_the_gap_fill_file(tmp_path, monkeypatch):
    import json
    from nfl import publish
    raw = tmp_path / "data-raw" / "nfl"
    raw.mkdir(parents=True)
    (raw / "odds.json").write_text(json.dumps({"props": {}}), encoding="utf-8")
    (raw / "odds_api_props.json").write_text(json.dumps({"games": {"BUF|LAC": {
        "props": {"receiving_yards": {"Khalil Shakir": {
            "_name": "Khalil Shakir", "line": 47.5, "over": 0.5, "source": "pinnacle",
            "book": "Pinnacle"}}}}}}), encoding="utf-8")
    monkeypatch.setattr(publish, "ROOT", tmp_path)
    got = publish.book_props()
    assert got["BUF|LAC"]["receiving_yards"]["Khalil Shakir"]["source"] == "pinnacle"


def test_the_card_names_the_book_whose_line_it_is():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[2] / "index.html").read_text(encoding="utf-8")
    assert "pinnacle:\"Pinnacle\"" in html and "function lineBook(p)" in html


def test_an_empty_reply_is_free_and_does_not_use_up_a_check():
    events = [{"id": "e1", "home_team": "Buffalo Bills", "away_team": "Los Angeles Chargers",
               "commence_time": "2026-09-27T17:00:00Z"}]
    client = FakeClient(events, {"bookmakers": []})
    store = sync.run(client, now=_t("2026-09-26T16:05"), store={})
    assert store["games"]["BUF|LAC"]["checks"] == []


def test_the_board_publishes_yardage_picks_on_book_lines_only():
    from nfl import publish
    assert publish.REQUIRE_BOOK_LINE is True
