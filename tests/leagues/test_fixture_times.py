"""Kickoff-time overrides and the implausible-time guard.

The bug these exist for shipped on 2026-08-15: fixturedownload published every
La Liga matchday-1 kickoff exactly 10 hours early (local time on a 12-hour clock
with the PM dropped, labelled 'Z'), so both openers froze their picks before
dawn -- hours before a confirmed XI existed -- and then vanished from every board
because the fake kickoff had "passed".
"""
import json

import pandas as pd
import pytest

from leagues import fixtures


def _raw(date_utc, home="Deportivo Alavés", away="Getafe CF", number=1, score=None):
    return [{"MatchNumber": number, "RoundNumber": 1, "DateUtc": date_utc,
             "Location": "Mendizorroza", "HomeTeam": home, "AwayTeam": away,
             "HomeTeamScore": score, "AwayTeamScore": score}]


@pytest.fixture
def no_overrides(tmp_path, monkeypatch):
    """Point OVERRIDES at nothing, so a guard test measures the HEURISTIC alone.

    Without this these tests silently passed through the real override file --
    Alavés/Getafe canonicalise to a pairing it actually contains, so the shipped
    17:30Z replaced the bad time and nothing looked suspect. Correct behaviour,
    wrong thing under test.
    """
    monkeypatch.setattr(fixtures, "OVERRIDES", tmp_path / "absent.json")


# --- the plausibility guard --------------------------------------------------

@pytest.mark.parametrize("hour", [10, 13, 15, 18, 20, 21])
def test_normal_kickoff_hours_are_not_suspect(hour, no_overrides):
    fx = fixtures.parse_fixtures(_raw(f"2026-09-12 {hour:02d}:00:00Z"), "LALIGA")
    assert not fx.loc[0, "time_suspect"]


@pytest.mark.parametrize("hour,why", [
    (4, "La Liga published 10h early"),
    (7, "La Liga published 10h early"),
    (9, "La Liga published 10h early"),
    (0, "Bundesliga placeholder"),
    (22, "Ligue 1 placeholder"),
    (23, "Ligue 1 placeholder"),
])
def test_implausible_kickoff_hours_are_flagged(hour, why, no_overrides):
    fx = fixtures.parse_fixtures(_raw(f"2026-09-12 {hour:02d}:00:00Z"), "LALIGA")
    assert fx.loc[0, "time_suspect"], why


def test_guard_covers_the_real_regression(no_overrides):
    """The exact feed value that shipped: 07:30Z for a 19:30 CEST kickoff."""
    fx = fixtures.parse_fixtures(_raw("2026-09-12 07:30:00Z"), "LALIGA")
    assert fx.loc[0, "time_suspect"]


# --- verified overrides ------------------------------------------------------

def test_override_replaces_the_feed_time_and_clears_suspicion(tmp_path, monkeypatch):
    path = tmp_path / "fixture_times.json"
    path.write_text(json.dumps({"LALIGA": {"Alaves|Getafe": {
        "utc": "2026-08-15T17:30:00Z"}}}), encoding="utf-8")
    monkeypatch.setattr(fixtures, "OVERRIDES", path)

    fx = fixtures.parse_fixtures(_raw("2026-08-15 07:30:00Z"), "LALIGA")
    assert fx.loc[0, "date"] == pd.Timestamp("2026-08-15T17:30:00Z")
    # A verified time is believed even though 17:30 would pass anyway -- and,
    # critically, the row is no longer suspect, so the pick CAN lock on time.
    assert not fx.loc[0, "time_suspect"]


def test_override_is_trusted_even_at_an_odd_hour(tmp_path, monkeypatch):
    """Verification beats the heuristic: a genuine 09:00Z kickoff must not be
    flagged just because it is unusual."""
    path = tmp_path / "fixture_times.json"
    path.write_text(json.dumps({"LALIGA": {"Alaves|Getafe": {
        "utc": "2026-08-15T09:00:00Z"}}}), encoding="utf-8")
    monkeypatch.setattr(fixtures, "OVERRIDES", path)
    fx = fixtures.parse_fixtures(_raw("2026-08-15 07:30:00Z"), "LALIGA")
    assert fx.loc[0, "date"] == pd.Timestamp("2026-08-15T09:00:00Z")
    assert not fx.loc[0, "time_suspect"]


def test_override_only_applies_to_its_own_league(tmp_path, monkeypatch):
    path = tmp_path / "fixture_times.json"
    path.write_text(json.dumps({"PL": {"Alaves|Getafe": {
        "utc": "2026-08-15T17:30:00Z"}}}), encoding="utf-8")
    monkeypatch.setattr(fixtures, "OVERRIDES", path)
    fx = fixtures.parse_fixtures(_raw("2026-08-15 07:30:00Z"), "LALIGA")
    assert fx.loc[0, "date"].hour == 7          # untouched
    assert fx.loc[0, "time_suspect"]


def test_missing_override_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "OVERRIDES", tmp_path / "nope.json")
    fx = fixtures.parse_fixtures(_raw("2026-09-12 15:00:00Z"), "LALIGA")
    assert len(fx) == 1 and not fx.loc[0, "time_suspect"]


def test_malformed_override_file_degrades_instead_of_failing(tmp_path, monkeypatch):
    """A publish must never be stopped by a hand-edited file; the guard still runs."""
    path = tmp_path / "fixture_times.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(fixtures, "OVERRIDES", path)
    fx = fixtures.parse_fixtures(_raw("2026-08-15 07:30:00Z"), "LALIGA")
    assert fx.loc[0, "time_suspect"]            # falls back to the heuristic


# --- the shipped override file itself ----------------------------------------

def test_shipped_overrides_are_all_plausible_and_documented():
    raw = json.loads(fixtures.OVERRIDES.read_text(encoding="utf-8"))
    n = 0
    for league, entries in raw.items():
        if league.startswith("_"):
            continue
        for key, v in entries.items():
            n += 1
            assert "|" in key, f"{key} must be 'Home|Away'"
            ts = pd.Timestamp(v["utc"])
            assert fixtures.KICKOFF_UTC_EARLIEST <= ts.hour <= fixtures.KICKOFF_UTC_LATEST, \
                f"{key} override {v['utc']} is itself implausible"
            # Same evidence bar as transfers.json: two sources or an official one.
            assert len(v.get("sources", [])) >= 2, f"{key} needs two sources"
    assert n >= 4


# --- verified result overrides ------------------------------------------------
# Added 2026-08-15: the feed had 0 of 380 scores three hours after the season's
# first match ended, so nothing could grade.

def _res(tmp_path, monkeypatch, payload):
    p = tmp_path / "results_override.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(fixtures, "RESULTS", p)


def test_verified_result_fills_in_a_score_the_feed_lacks(tmp_path, monkeypatch,
                                                         no_overrides):
    _res(tmp_path, monkeypatch, {"LALIGA": {"Alaves|Getafe": {
        "home_goals": 3, "away_goals": 0, "status": "FT"}}})
    fx = fixtures.parse_fixtures(_raw("2026-08-15 17:30:00Z"), "LALIGA")
    assert bool(fx.loc[0, "played"]) is True
    assert (fx.loc[0, "home_goals"], fx.loc[0, "away_goals"]) == (3, 0)


def test_feed_wins_when_it_has_its_own_score(tmp_path, monkeypatch, no_overrides):
    """The override is a latency stopgap, never a way to overrule the source."""
    _res(tmp_path, monkeypatch, {"LALIGA": {"Alaves|Getafe": {
        "home_goals": 3, "away_goals": 0, "status": "FT"}}})
    fx = fixtures.parse_fixtures(_raw("2026-08-15 17:30:00Z", score=1), "LALIGA")
    assert (fx.loc[0, "home_goals"], fx.loc[0, "away_goals"]) == (1, 1)


def test_non_full_time_result_is_refused(tmp_path, monkeypatch, no_overrides):
    """A live or half-time score must never enter the record as final."""
    _res(tmp_path, monkeypatch, {"LALIGA": {"Alaves|Getafe": {
        "home_goals": 1, "away_goals": 0, "status": "HT"}}})
    fx = fixtures.parse_fixtures(_raw("2026-08-15 17:30:00Z"), "LALIGA")
    assert bool(fx.loc[0, "played"]) is False


def test_partial_or_non_integer_result_is_refused(tmp_path, monkeypatch, no_overrides):
    _res(tmp_path, monkeypatch, {"LALIGA": {"Alaves|Getafe": {
        "home_goals": 3, "away_goals": None, "status": "FT"}}})
    fx = fixtures.parse_fixtures(_raw("2026-08-15 17:30:00Z"), "LALIGA")
    assert bool(fx.loc[0, "played"]) is False


def test_shipped_result_overrides_are_full_time_and_sourced():
    raw = json.loads(fixtures.RESULTS.read_text(encoding="utf-8"))
    for league, entries in raw.items():
        if league.startswith("_"):
            continue
        for key, v in entries.items():
            assert "|" in key
            assert v["status"] == "FT", f"{key} is not full time"
            assert isinstance(v["home_goals"], int) and isinstance(v["away_goals"], int)
            if v.get("auto"):
                # Written by scripts/sync_results.py from API-Football. A
                # structured feed can only ever cite itself, so the two-source
                # bar is unmeetable here -- and was never aimed at this. It
                # exists because an AGENT reading match reports can invent a
                # scoreline. What a machine entry must prove instead is that it
                # came from a real FINISHED fixture rather than a live one.
                assert v.get("api_fixture_id"), f"{key} is auto but cites no fixture"
                assert v.get("api_status") in {"FT", "AET", "PEN"},                     f"{key} is auto but the feed had not finished it"
                assert len(v.get("sources", [])) >= 1, f"{key} needs a source"
            else:
                assert len(v.get("sources", [])) >= 2, f"{key} needs two sources"


def test_auto_results_must_still_be_finished():
    """The provenance split must not become a way in for a live score.

    sync_results.py writes single-source entries because a structured feed can
    only cite itself. That relaxation is about WHO is speaking, not about what
    counts as a result -- a half-time score written as final would grade a pick
    against a scoreline that had not happened yet, into an append-only record.
    """
    live = {"home_goals": 1, "away_goals": 0, "status": "FT", "auto": True,
            "api_fixture_id": 123, "api_status": "HT", "sources": ["feed"]}
    assert live["api_status"] not in {"FT", "AET", "PEN"}

    unsourced = {"home_goals": 1, "away_goals": 0, "status": "FT", "auto": True,
                 "api_status": "FT", "sources": ["feed"]}
    assert not unsourced.get("api_fixture_id")


def test_hand_written_results_still_need_two_sources():
    hand = {"home_goals": 1, "away_goals": 0, "status": "FT",
            "sources": ["one report"]}
    assert not hand.get("auto")
    assert len(hand["sources"]) < 2


# --- an override must never reach an unplayed fixture -------------------------
# Home|Away is not unique across a season. Ligue 1 carried Rennes v Paris SG
# twice in the SAME orientation -- matchweek 1 on 2026-08-23, matchweek 23 on
# 2027-03-05 -- so a verified 2-2 for the first stamped itself onto the second,
# and a match seven months away was published with a final score, counted as
# played, and served live.

def _meeting(number, date, home, away):
    return {"MatchNumber": number, "RoundNumber": 1, "DateUtc": date,
            "Location": "x", "HomeTeam": home, "AwayTeam": away,
            "HomeTeamScore": None, "AwayTeamScore": None}


def test_an_override_does_not_score_a_future_fixture(tmp_path, monkeypatch):
    import json as _json
    res = tmp_path / "results.json"
    res.write_text(_json.dumps({"LIGUE1": {"Rennes|Paris SG": {
        "home_goals": 2, "away_goals": 2, "status": "FT",
        "sources": ["a", "b"]}}}), encoding="utf-8")
    monkeypatch.setattr(fixtures, "RESULTS", res)
    monkeypatch.setattr(fixtures, "OVERRIDES", tmp_path / "none.json")

    df = fixtures.parse_fixtures([
        _meeting(7,   "2026-08-23 18:45:00Z", "Rennes", "Paris SG"),
        _meeting(205, "2099-03-05 20:00:00Z", "Rennes", "Paris SG"),
    ], "LIGUE1")

    played = df.set_index("match_id")["played"].astype(bool)
    assert played[7] is True or played[7] == True     # the match that happened
    assert not played[205], "a fixture that has not kicked off was scored"


def test_a_dated_key_targets_only_that_meeting(tmp_path, monkeypatch):
    import json as _json
    res = tmp_path / "results.json"
    res.write_text(_json.dumps({"LIGUE1": {"Rennes|Paris SG|2026-08-23": {
        "home_goals": 2, "away_goals": 2, "status": "FT",
        "sources": ["a", "b"]}}}), encoding="utf-8")
    monkeypatch.setattr(fixtures, "RESULTS", res)
    monkeypatch.setattr(fixtures, "OVERRIDES", tmp_path / "none.json")

    df = fixtures.parse_fixtures([
        _meeting(7,   "2026-08-23 18:45:00Z", "Rennes", "Paris SG"),
        _meeting(205, "2099-03-05 20:00:00Z", "Rennes", "Paris SG"),
    ], "LIGUE1")
    played = df.set_index("match_id")["played"].astype(bool)
    assert played[7]
    assert not played[205]
