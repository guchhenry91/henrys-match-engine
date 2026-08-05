"""Newly-locked-picks Telegram notifier: only locked, only once, never blocks."""
import json

from scripts import telegram_picks as tp


def _write(tmp_path, name, payload):
    (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_only_locked_picks_are_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "OUT", tmp_path)
    _write(tmp_path, "best", {"upcoming": [
        {"league_key": "PL", "id": 1, "home": "A", "away": "B", "pick": "A",
         "p_pick": 0.77, "provisional": False},
        {"league_key": "PL", "id": 2, "home": "C", "away": "D", "pick": "C",
         "p_pick": 0.70, "provisional": True},   # not locked yet -- excluded
    ]})
    out = tp.newly_locked_best(set())
    keys = [k for k, _ in out]
    assert "best:PL:1" in keys
    assert "best:PL:2" not in keys


def test_already_sent_picks_are_not_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "OUT", tmp_path)
    _write(tmp_path, "best", {"upcoming": [
        {"league_key": "PL", "id": 1, "home": "A", "away": "B", "pick": "A",
         "p_pick": 0.77, "provisional": False}]})
    out = tp.newly_locked_best({"best:PL:1"})
    assert out == []


def test_player_picks_key_includes_market_and_player(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "OUT", tmp_path)
    _write(tmp_path, "player_picks", {"upcoming": [
        {"league_key": "BUNDESLIGA", "id": 4, "player": "Harry Kane",
         "team": "Bayern Munich", "market": "shots", "p_pick": 0.78,
         "provisional": False}]})
    out = tp.newly_locked_players(set())
    assert len(out) == 1
    key, msg = out[0]
    assert key == "player:BUNDESLIGA:4:shots:Harry Kane"
    assert "Kane" in msg and "78%" in msg


def test_main_is_a_silent_noop_without_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert tp.main() == 0


def test_main_never_fails_the_pipeline_on_a_send_error(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "OUT", tmp_path)
    monkeypatch.setattr(tp, "SENT_LOG", tmp_path / "sent.json")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
    _write(tmp_path, "best", {"upcoming": [
        {"league_key": "PL", "id": 1, "home": "A", "away": "B", "pick": "A",
         "p_pick": 0.77, "provisional": False}]})
    def boom(*a, **k): raise RuntimeError("network down")
    monkeypatch.setattr(tp, "send", boom)
    assert tp.main() == 0                          # never raises, never fails the run
    assert not (tmp_path / "sent.json").exists()    # not marked sent -- will retry


def test_sent_picks_persist_and_dedupe_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "OUT", tmp_path)
    monkeypatch.setattr(tp, "SENT_LOG", tmp_path / "sent.json")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
    _write(tmp_path, "best", {"upcoming": [
        {"league_key": "PL", "id": 1, "home": "A", "away": "B", "pick": "A",
         "p_pick": 0.77, "provisional": False}]})
    sent_calls = []
    monkeypatch.setattr(tp, "send", lambda *a: sent_calls.append(a) or 200)
    assert tp.main() == 0
    assert len(sent_calls) == 1
    # second run, same picks still upcoming and locked -> must NOT resend
    assert tp.main() == 0
    assert len(sent_calls) == 1
