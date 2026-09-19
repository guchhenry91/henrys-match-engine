"""News inputs for NFL, NBA and the Champions League, kept by the cloud routine.

Each can only ever make a board MORE cautious (NFL, NBA) or add information to a
card (Champions League) -- never loosen what an official feed already says.
"""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from nfl import news

NOW = datetime(2026, 9, 19, 14, 0, tzinfo=timezone.utc)


def _file(tmp_path, players):
    p = tmp_path / "news.json"
    p.write_text(json.dumps({"players": players}), encoding="utf-8")
    return p


def _entry(status, days_ago=0, **extra):
    return {"status": status, "checked": (NOW - timedelta(days=days_ago)).isoformat(), **extra}


def test_fresh_out_and_doubt_are_loaded(tmp_path):
    got = news.load_player_news(_file(tmp_path, {"A": _entry("out"), "B": _entry("doubt")}), NOW)
    assert got["A"]["status"] == "out" and got["B"]["status"] == "doubt"


def test_stale_undated_and_unknown_status_are_ignored(tmp_path):
    got = news.load_player_news(_file(tmp_path, {
        "Old": _entry("out", days_ago=9),
        "Undated": {"status": "out"},
        "Weird": _entry("active")}), NOW)
    assert got == {}


def test_manual_out_overrides_the_api():
    merged = news.merge({"A": {"status": "doubt"}}, {"A": {"status": "out", "detail": "m"}})
    assert merged["A"]["status"] == "out"


def test_manual_doubt_never_softens_an_official_out():
    merged = news.merge({"A": {"status": "out"}}, {"A": {"status": "doubt", "detail": "m"}})
    assert merged["A"]["status"] == "out"


def test_nfl_availability_includes_manual_news(tmp_path, monkeypatch):
    from nfl import publish
    raw = tmp_path / "data-raw" / "nfl"
    raw.mkdir(parents=True)
    (raw / "injuries.json").write_text(json.dumps({"players": {}}), encoding="utf-8")
    (raw / "news.json").write_text(json.dumps({"players": {
        "Late Scratch": {"status": "out", "checked": datetime.now(timezone.utc).isoformat()}}}),
        encoding="utf-8")
    monkeypatch.setattr(publish, "ROOT", tmp_path)
    assert publish.availability()["Late Scratch"]["status"] == "out"


def test_nba_news_removes_out_and_flags_doubt():
    from nba import publish
    picks = [{"player": "Out Guy"}, {"player": "Doubt Guy"}, {"player": "Fine Guy"}]
    kept = publish.apply_news(picks, {"Out Guy": {"status": "out", "detail": "x"},
                                      "Doubt Guy": {"status": "doubt", "detail": "y"}})
    assert [p["player"] for p in kept] == ["Doubt Guy", "Fine Guy"]
    assert kept[0]["availability"] == "doubt"


def test_ucl_club_news_skips_stale_and_undated(tmp_path, monkeypatch):
    from ucl import publish
    f = tmp_path / "news.json"
    f.write_text(json.dumps({"clubs": {
        "Arsenal": {"out": ["Saka"], "checked": "2026-09-18T10:00:00Z"},
        "Lille": {"out": ["X"], "checked": "2026-09-01T10:00:00Z"},
        "Nowhen": {"out": ["Y"]}}}), encoding="utf-8")
    monkeypatch.setattr(publish, "NEWS", f)
    got = publish.club_news(now=pd.Timestamp("2026-09-19T14:00:00Z"))
    assert list(got) == ["Arsenal"] and got["Arsenal"]["out"] == ["Saka"]


def test_ucl_override_fills_a_missed_result_but_never_beats_the_feed(tmp_path, monkeypatch):
    from ucl import data
    hist = tmp_path / "history.json"
    hist.write_text(json.dumps({"seasons": {"2026": [
        {"id": 1, "home": "A", "away": "B", "home_goals": 1, "away_goals": 0}]}}),
        encoding="utf-8")
    over = tmp_path / "override.json"
    over.write_text(json.dumps({"results": {
        "1": {"home_goals": 9, "away_goals": 9, "status": "FT", "sources": ["s1", "s2"]},
        "2": {"home": "C", "away": "D", "home_goals": 2, "away_goals": 2,
              "status": "FT", "sources": ["s1", "s2"]},
        "3": {"home_goals": 1, "away_goals": 0, "status": "FT", "sources": ["only one"]}}}),
        encoding="utf-8")
    monkeypatch.setattr(data, "HISTORY", hist)
    monkeypatch.setattr(data, "OVERRIDE", over)
    got = data.results_by_id(season=2026)
    assert got["1"]["home_goals"] == 1                  # the feed wins
    assert got["2"]["home_goals"] == 2                  # a missed result is filled
    assert "3" not in got                               # one source is not enough


def test_the_workflows_rebuild_when_news_is_pushed():
    import yaml
    for wf, path in (("leagues", "data-raw/leagues/news.json"),
                     ("leagues", "data-raw/ucl/news.json"),
                     ("nfl", "data-raw/nfl/news.json")):
        doc = yaml.safe_load(open(f".github/workflows/{wf}.yml", encoding="utf-8"))
        on = doc.get("on") or doc.get(True)
        assert path in on["push"]["paths"], f"{wf}.yml does not rebuild on {path}"
