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
