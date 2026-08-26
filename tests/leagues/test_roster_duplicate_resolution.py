"""A duplicate we have already verified must not fail the pipeline.

On 2026-08-26 the whole leagues run stopped at the roster audit: API-Football
listed Taiwo Awoniyi at Nottingham Forest AND Coventry simultaneously, because
he had just moved and the registration had not settled in the feed. transfers.json
already recorded him at Coventry, on the project's evidence bar -- Nottingham
Forest's own announcement plus three independent outlets.

Failing on a conflict we have verified and written down trains the operator to
ignore the check. It stays a WARNING rather than becoming silence, because the
snapshot really is inconsistent until the feed catches up.
"""
import json

import pytest

from scripts import roster_integrity_check as ric


@pytest.fixture(autouse=True)
def _transfers(tmp_path, monkeypatch):
    path = tmp_path / "transfers.json"
    path.write_text(json.dumps({"PL": {"Taiwo Awoniyi": "Coventry"}}), encoding="utf-8")
    monkeypatch.setattr(ric, "TRANSFERS", path)


def test_the_abbreviated_feed_name_matches_the_written_out_one():
    assert ric._same_player("T. Awoniyi", "Taiwo Awoniyi")


def test_a_different_player_with_the_same_surname_does_not_match():
    """Must stay narrow -- a loose match here would hide a real conflict."""
    assert not ric._same_player("R. Neves", "Joao Neves")


def test_a_different_surname_never_matches():
    assert not ric._same_player("T. Awoniyi", "Taiwo Adebayo")


def test_verified_clubs_reads_the_override_list():
    assert ric._verified_clubs("PL") == {"taiwo awoniyi": "Coventry"}


def test_an_unverified_duplicate_is_still_an_error():
    """The check must keep doing the job it was written for."""
    payload = {"PL": {
        "Arsenal": {"players": [{"id": "1", "name": "A. Nobody", "position": "X"}]},
        "Chelsea": {"players": [{"id": "1", "name": "A. Nobody", "position": "X"}]},
    }}
    errors = [e for e in ric.audit(payload)[0] if "listed for" in e]
    assert errors, "an unknown duplicate identity was not reported"


def test_a_verified_duplicate_is_a_warning_not_an_error():
    payload = {"PL": {
        "Nottingham Forest": {"players": [
            {"id": "8598", "name": "T. Awoniyi", "position": "Attacker"}]},
        "Coventry": {"players": [
            {"id": "8598", "name": "T. Awoniyi", "position": "Attacker"}]},
    }}
    errors, warnings = ric.audit(payload)
    assert not [e for e in errors if "listed for" in e]
    assert any("VERIFIED at Coventry" in w for w in warnings)
