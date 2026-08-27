"""Clubs are reconciled by ID against the nflverse roster. Never by name.

This was built against API-NFL first and it failed in the worst possible way: its
player endpoint returned 43-71 plausible names a team -- enough to pass every
count-based completeness check -- and those names did not include Patrick Mahomes,
A.J. Brown, Alvin Kamara or Austin Ekeler. Philadelphia came back as Andy Dalton,
Britain Covey and Danny Gray. The board dropped 177 current players as having left
the league and every guard reported success, because I had measured QUANTITY and
concluded IDENTITY.

nflverse publishes 91.6 players a team with gsis_id, the same key the stats use,
so the join needs no name matching at all.
"""
import pandas as pd
import pytest

from nfl import rosters


def _roster(rows):
    return pd.DataFrame(rows, columns=["season", "team", "position", "status",
                                       "full_name", "gsis_id"])


def _row(gsis, team, status="ACT", name="A Player"):
    return {"season": 2026, "team": team, "position": "WR", "status": status,
            "full_name": name, "gsis_id": gsis}


def test_an_active_player_is_placed_on_his_team():
    index = rosters.build_index(_roster([_row("00-1", "KC")]))
    assert index == {"00-1": "KC"}


def test_cut_and_retired_players_are_not_placed():
    """Exactly the people the board kept projecting before any of this existed."""
    index = rosters.build_index(_roster([
        _row("00-1", "KC", "CUT"), _row("00-2", "KC", "RET"),
        _row("00-3", "KC", "RES"), _row("00-4", "KC", "ACT")]))
    assert index == {"00-4": "KC"}


def test_a_player_listed_by_two_teams_is_placed_by_neither():
    """Happens mid-camp. Guessing is how a projection lands on the wrong team
    while looking certain."""
    index = rosters.build_index(_roster([_row("00-1", "KC"), _row("00-1", "BUF")]))
    assert "00-1" not in index


def test_an_empty_roster_places_nobody():
    assert rosters.build_index(_roster([])) == {}
    assert rosters.build_index(None) == {}


def test_a_move_is_applied():
    index = {"00-1": "MIN"}
    team, why = rosters.reconcile("00-1", "ARI", index, trusted=True)
    assert team == "MIN" and why.startswith("moved")


def test_a_confirmed_player_keeps_his_team():
    team, why = rosters.reconcile("00-1", "KC", {"00-1": "KC"}, trusted=True)
    assert team == "KC" and why == "confirmed"


def test_a_player_on_no_active_roster_is_dropped_when_the_file_is_trusted():
    team, why = rosters.reconcile("00-9", "KC", {"00-1": "KC"}, trusted=True)
    assert team is None and "not on an active roster" in why


def test_nothing_is_dropped_when_the_file_is_not_trusted():
    """THE GUARD. An untrusted file must never delete anyone."""
    team, why = rosters.reconcile("00-9", "KC", {"00-1": "KC"}, trusted=False)
    assert team == "KC" and "unusable" in why


def test_corroboration_refuses_a_file_that_does_not_know_the_league():
    """The API-NFL failure, reproduced: plausible rows, wrong people."""
    lookup = {f"00-{i}": "AAA" for i in range(100)}
    known = [f"90-{i}" for i in range(100)]
    trusted, rate = rosters.corroborates(lookup, known)
    assert not trusted and rate == 0.0


def test_corroboration_accepts_a_file_that_does():
    known = [f"00-{i}" for i in range(100)]
    lookup = {i: "AAA" for i in known[:86]}
    trusted, rate = rosters.corroborates(lookup, known)
    assert trusted and rate == pytest.approx(0.86)


def test_corroboration_is_false_with_nothing_to_compare():
    assert rosters.corroborates({}, ["00-1"]) == (False, 0.0)
    assert rosters.corroborates({"00-1": "AAA"}, []) == (False, 0.0)


def test_ids_compare_as_strings_whatever_the_frame_holds():
    index = rosters.build_index(_roster([_row("00-1", "KC")]))
    team, _ = rosters.reconcile(0o1 if False else "00-1", "KC", index, trusted=True)
    assert team == "KC"
