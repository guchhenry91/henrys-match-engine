"""Form strips must come from the same rows the record is graded on.

A board showing one form window while the model and the grader used another is
explaining itself with numbers nobody scored it by. These also pin the merge
rule, because two feeds disagreeing about one match would otherwise show the same
game twice in a five-game strip.
"""
import pandas as pd
import pytest

from leagues import players


def _stats(rows):
    return pd.DataFrame(rows, columns=["date", "team", "player", "goals",
                                       "shots", "sot"])


def _row(day, player, goals=0, shots=0, sot=0, team="Arsenal"):
    return {"date": day, "team": team, "player": player,
            "goals": goals, "shots": shots, "sot": sot}


@pytest.fixture
def patched(monkeypatch):
    def apply(primary, fallback):
        monkeypatch.setattr(players, "match_player_stats", lambda lg: primary)
        monkeypatch.setattr(players, "api_match_stats", lambda lg: (fallback, set()))
    return apply


def test_it_returns_the_last_five_oldest_first(patched):
    patched(_stats([_row(f"2026-0{i}-01", "Saka", shots=i) for i in range(1, 8)]),
            _stats([]))
    form = players.recent_form("PL")
    assert form["Saka"]["shots"] == [3, 4, 5, 6, 7]


def test_a_player_with_fewer_than_five_matches_gets_what_he_has(patched):
    patched(_stats([_row("2026-01-01", "New Guy", goals=1)]), _stats([]))
    assert players.recent_form("PL")["New Guy"]["goals"] == [1]


def test_the_fallback_fills_matches_the_primary_lacks(patched):
    patched(_stats([_row("2026-01-01", "Saka", shots=1)]),
            _stats([_row("2026-02-01", "Saka", shots=4)]))
    assert players.recent_form("PL")["Saka"]["shots"] == [1, 4]


def test_understat_wins_a_match_both_feeds_report(patched):
    """Otherwise one game appears twice in a five-game strip."""
    patched(_stats([_row("2026-01-01", "Saka", shots=9)]),
            _stats([_row("2026-01-01", "Saka", shots=2)]))
    form = players.recent_form("PL")
    assert form["Saka"]["shots"] == [9], "the fallback overwrote the shot feed"


def test_no_feeds_at_all_yields_nothing_rather_than_raising(patched):
    patched(_stats([]), _stats([]))
    assert players.recent_form("PL") == {}


def test_a_broken_primary_still_yields_the_fallback(patched, monkeypatch):
    monkeypatch.setattr(players, "match_player_stats",
                        lambda lg: (_ for _ in ()).throw(TimeoutError("blip")))
    monkeypatch.setattr(players, "api_match_stats",
                        lambda lg: (_stats([_row("2026-01-01", "Saka", sot=2)]), set()))
    assert players.recent_form("PL")["Saka"]["sot"] == [2]


def test_every_market_stat_is_present(patched):
    patched(_stats([_row("2026-01-01", "Saka", goals=1, shots=4, sot=2)]), _stats([]))
    form = players.recent_form("PL")["Saka"]
    assert set(form) == {"goals", "shots", "sot"}
