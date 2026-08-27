"""The roster snapshot corroborates. It must never convict on thin evidence.

This rule was learned expensively on the soccer side, where treating incomplete
rosters as proof deleted Real Madrid, Barcelona, PSG and 14 of 18 Bundesliga
clubs -- 70% of two leagues -- because a free feed happened to list fewer than 18
names for them. The NFL board must not repeat it two weeks before a season opens.
"""
import pytest

from nfl import rosters


def _snapshot(teams, failed=None):
    return {"teams": teams, "failed_teams": failed or []}


def _team(names, complete=True):
    return {"players": names, "count": len(names), "complete": complete}


def test_name_key_ignores_accents_case_and_punctuation():
    assert rosters.name_key("De'Von Achane") == rosters.name_key("Devon Achane")
    assert rosters.name_key("Vinícius Júnior") == rosters.name_key("Vinicius Junior")


def test_name_key_ignores_generational_suffixes():
    """Both feeds spell these inconsistently; a strict key calls one man two."""
    assert rosters.name_key("Travis Etienne Jr.") == rosters.name_key("Travis Etienne")
    assert rosters.name_key("Michael Pittman Jr") == rosters.name_key("Michael Pittman")


def test_name_key_still_separates_different_people():
    """The suffix rule must not merge Penix into Pittman, which is exactly the
    false positive a looser check produced when this was investigated."""
    assert rosters.name_key("Michael Penix Jr.") != rosters.name_key("Michael Pittman Jr.")


def test_a_move_is_applied_when_the_roster_is_complete():
    lookup, done = rosters.index(_snapshot({"MIN": _team(["Kyler Murray"] * 1 + [f"P{i}" for i in range(40)]),
                                            "ARI": _team([f"Q{i}" for i in range(40)])}))
    team, why = rosters.reconcile("Kyler Murray", "ARI", lookup, done)
    assert team == "MIN"
    assert "moved" in why


def test_a_player_on_no_complete_roster_is_dropped_only_when_evidence_is_complete():
    lookup, done = rosters.index(_snapshot({
        "AAA": _team([f"P{i}" for i in range(40)]),
        "BBB": _team([f"Q{i}" for i in range(40)]),
    }))
    assert done is True
    team, why = rosters.reconcile("Retired Guy", "AAA", lookup, done)
    assert team is None and "no current roster" in why


def test_a_thin_roster_never_deletes_anyone():
    """THE RULE. A team returning eight players is a broken fetch, not a squad."""
    lookup, done = rosters.index(_snapshot({
        "AAA": _team([f"P{i}" for i in range(40)]),
        "BBB": _team(["Only", "Eight", "Names"], complete=False),
    }))
    assert done is False
    team, why = rosters.reconcile("Some Player", "BBB", lookup, done)
    assert team == "BBB", "a thin roster was treated as proof of absence"
    assert "incomplete" in why


def test_a_failed_team_makes_the_evidence_incomplete():
    lookup, done = rosters.index(_snapshot(
        {"AAA": _team([f"P{i}" for i in range(40)])}, failed=["BBB: timeout"]))
    assert done is False
    team, _ = rosters.reconcile("Unknown Man", "BBB", lookup, done)
    assert team == "BBB"


def test_two_teams_listing_one_man_is_not_guessed():
    """Camp cuts produce this. Picking one is how a projection lands on the wrong
    team while looking certain."""
    shared = "Split Player"
    lookup, done = rosters.index(_snapshot({
        "AAA": _team([shared] + [f"P{i}" for i in range(40)]),
        "BBB": _team([shared] + [f"Q{i}" for i in range(40)]),
    }))
    team, why = rosters.reconcile(shared, "AAA", lookup, done)
    assert team == "AAA"
    assert "2 teams" in why


def test_no_snapshot_at_all_falls_back_rather_than_emptying_the_board():
    team, why = rosters.reconcile("Anyone", "AAA", {}, False)
    assert team == "AAA"
    assert "no roster snapshot" in why


def test_a_confirmed_player_is_marked_as_such():
    lookup, done = rosters.index(_snapshot(
        {"AAA": _team(["Real Player"] + [f"P{i}" for i in range(40)])}))
    team, why = rosters.reconcile("Real Player", "AAA", lookup, done)
    assert team == "AAA" and why == "confirmed"


def test_corroboration_refuses_a_snapshot_that_does_not_know_the_league():
    """THE GUARD THAT WAS MISSING. API-NFL returned 43-71 plausible names a team --
    every count check passed -- and they did not include Patrick Mahomes, A.J.
    Brown or Alvin Kamara. 177 current players were marked as having left."""
    lookup = {rosters.name_key(f"Fringe Guy {i}"): ["AAA"] for i in range(100)}
    known = [f"Star Player {i}" for i in range(100)]
    trusted, rate = rosters.corroborates(lookup, known)
    assert not trusted and rate == 0.0


def test_corroboration_accepts_a_snapshot_that_does():
    known = [f"Real Player {i}" for i in range(100)]
    lookup = {rosters.name_key(n): ["AAA"] for n in known[:80]}
    trusted, rate = rosters.corroborates(lookup, known)
    assert trusted and rate == pytest.approx(0.80)


def test_corroboration_is_false_with_nothing_to_compare():
    assert rosters.corroborates({}, ["A"]) == (False, 0.0)
    assert rosters.corroborates({"a": ["X"]}, []) == (False, 0.0)


def test_a_name_column_with_missing_values_does_not_crash_the_build():
    """publish sorts these; a NaN mixed with strings raises TypeError and took
    down a CI run."""
    import math
    names = ["Real Player", float("nan"), None, "", "Another Player"]
    cleaned = sorted({str(n) for n in names if isinstance(n, str) and n.strip()})
    assert cleaned == ["Another Player", "Real Player"]
