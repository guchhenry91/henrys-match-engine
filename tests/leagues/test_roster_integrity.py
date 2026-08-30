import json
from pathlib import Path

from scripts.roster_integrity_check import audit


ROOT = Path(__file__).resolve().parents[2]


def test_roster_snapshot_covers_every_configured_club_without_duplicates():
    payload = json.loads(
        (ROOT / "data-raw" / "leagues" / "rosters.json").read_text(encoding="utf-8"))
    errors, _warnings = audit(payload)
    assert errors == []


def test_roster_snapshot_is_dated_and_documents_that_it_is_provisional():
    payload = json.loads(
        (ROOT / "data-raw" / "leagues" / "rosters.json").read_text(encoding="utf-8"))
    assert payload["_verified_at"]
    assert payload["_source"]
    assert payload["_provisional"]


def test_a_league_absent_from_the_snapshot_warns_rather_than_errors():
    """ABSENCE OF EVIDENCE IS NOT CONTRADICTORY EVIDENCE -- the same distinction
    the engine draws for a thin club roster.

    A league the snapshot does not cover (a newly added one, or an outage at the
    source) must not fail the audit as though its data were WRONG. publish already
    detects the gap and prints a data_warning, and attribution falls back to last
    season plus transfer overrides rather than deleting anyone.
    """
    payload = json.loads(
        (ROOT / "data-raw" / "leagues" / "rosters.json").read_text(encoding="utf-8"))
    payload.pop("PL", None)
    errors, warnings = audit(payload)
    assert not [e for e in errors if e.startswith("PL:")]
    assert any(w.startswith("PL:") and "no roster evidence" in w for w in warnings)


def test_a_league_present_but_WRONG_is_still_an_error():
    """The dangerous case must keep failing: evidence that contradicts the league
    is how clubs get deleted."""
    payload = json.loads(
        (ROOT / "data-raw" / "leagues" / "rosters.json").read_text(encoding="utf-8"))
    if not payload.get("PL"):
        import pytest
        pytest.skip("no PL snapshot in this checkout")
    payload["PL"] = {next(iter(payload["PL"])): payload["PL"][next(iter(payload["PL"]))]}
    errors, _ = audit(payload)
    assert any(e.startswith("PL:") for e in errors)
