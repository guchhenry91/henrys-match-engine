import pytest

from leagues import fixtures as fixtures_mod
from leagues.fixtures import parse_fixtures


@pytest.fixture(autouse=True)
def _isolate_override_files(tmp_path, monkeypatch):
    """Point the override files at nothing for every test in this module.

    These tests use Arsenal v Coventry as their sample fixture, and once that
    pairing appeared in the real results_override.json -- 3-0, written the night
    it was played -- parse_fixtures dutifully marked the unplayed sample as
    played and the test failed. A unit test asserting on parser behaviour must
    not read whatever the live data files happen to contain today.
    """
    monkeypatch.setattr(fixtures_mod, "OVERRIDES", tmp_path / "no_times.json")
    monkeypatch.setattr(fixtures_mod, "RESULTS", tmp_path / "no_results.json")

RAW = [
    {"MatchNumber": 1, "RoundNumber": 1, "DateUtc": "2026-08-21 19:00:00Z",
     "Location": "Emirates Stadium", "HomeTeam": "Arsenal", "AwayTeam": "Coventry",
     "HomeTeamScore": None, "AwayTeamScore": None},
    {"MatchNumber": 2, "RoundNumber": 1, "DateUtc": "2026-08-22 14:00:00Z",
     "Location": "Anfield", "HomeTeam": "Liverpool", "AwayTeam": "Man City",
     "HomeTeamScore": 2, "AwayTeamScore": 1},
]


def test_parse_fixtures_normalizes_names_and_types():
    df = parse_fixtures(RAW, "PL")
    assert len(df) == 2
    assert list(df.columns) == ["match_id", "round", "date", "venue",
                                "home", "away", "home_goals", "away_goals", "played",
                                # time_suspect: the feed's kickoff time is not
                                # believable, so publish must not freeze the pick
                                # on it (see test_fixture_times.py).
                                "time_suspect"]
    assert df.loc[0, "home"] == "Arsenal"
    assert df.loc[1, "away"] == "Manchester City"


def test_parse_fixtures_marks_played_only_when_both_scores_present():
    df = parse_fixtures(RAW, "PL")
    assert bool(df.loc[0, "played"]) is False
    assert bool(df.loc[1, "played"]) is True
    assert df.loc[1, "home_goals"] == 2
