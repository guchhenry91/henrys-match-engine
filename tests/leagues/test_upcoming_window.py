"""The current matchweek is chosen from live-or-future fixtures only.

On 2026-09-16 Levante v Athletic Club was postponed (PST). It kept its original
date and no score, became "the soonest unplayed fixture", and La Liga's board
shrank to that one stale game -- which failed the sanity check and blocked the
refresh of all five leagues for two days.
"""
import pandas as pd

from leagues import fixtures

NOW = pd.Timestamp("2026-09-19T13:00:00Z")


def _fx(rows):
    return pd.DataFrame([{"round": r, "date": pd.Timestamp(d, tz="UTC"), "home": h, "away": "X"}
                         for r, d, h in rows])


def test_a_postponed_fixture_does_not_anchor_the_matchweek():
    rem = _fx([(6, "2026-09-16 19:30", "Levante"),        # postponed, no score
               (7, "2026-09-19 14:15", "Athletic"),
               (7, "2026-09-20 14:15", "Atletico"),
               (8, "2026-09-26 14:15", "Next week")])
    got = fixtures.upcoming_window(rem, now=NOW)
    assert list(got["home"]) == ["Athletic", "Atletico"]


def test_a_match_in_play_is_still_current():
    rem = _fx([(7, "2026-09-19 12:00", "Osasuna"),        # kicked off an hour ago
               (7, "2026-09-19 16:30", "Celta")])
    assert set(fixtures.upcoming_window(rem, now=NOW)["home"]) == {"Osasuna", "Celta"}


def test_a_rescheduled_early_round_game_comes_back():
    rem = _fx([(6, "2026-09-23 19:00", "Levante"),        # new date, still round 6
               (7, "2026-09-19 14:15", "Athletic")])
    assert set(fixtures.upcoming_window(rem, now=NOW)["home"]) == {"Levante", "Athletic"}


def test_only_overdue_fixtures_means_an_empty_window_not_a_stale_one():
    rem = _fx([(6, "2026-09-16 19:30", "Levante")])
    assert fixtures.upcoming_window(rem, now=NOW).empty


def test_publish_uses_the_window():
    src = open("leagues/publish.py", encoding="utf-8").read()
    assert "fixtures.upcoming_window(remaining" in src
