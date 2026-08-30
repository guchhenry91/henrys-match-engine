"""The depth-chart gate removes real players, so it has to be provably safe.

It exists because six of nineteen passing picks on the 2026 week 1 board were
BACKUP quarterbacks -- Marcus Mariota (WAS QB2) was the second-highest passing
pick at 62%, and Cleveland published its QB2 and QB3 while its starter appeared
nowhere. A backup's line is his own entering median, set in relief, so "over"
looks easy right up until he takes no snap.

The failure mode to guard against is the opposite one: an August roster feed that
looked complete while omitting Patrick Mahomes deleted 177 current players. So
absence from the chart must never remove anyone, and a chart that does not
recognise the board is ignored wholesale.
"""
import pandas as pd
import pytest

from nfl import config, data, depth


def _chart(rows):
    return pd.DataFrame(rows, columns=["dt", "team", "player_name", "gsis_id",
                                       "pos_abb", "pos_rank"])


ROWS = [
    ("2026-08-30T12:30:54Z", "WAS", "Starter Guy", "00-0000001", "QB", 1),
    ("2026-08-30T12:30:54Z", "WAS", "Marcus Mariota", "00-0000002", "QB", 2),
    ("2026-08-30T12:30:54Z", "CLE", "Dillon Gabriel", "00-0000003", "QB", 3),
    ("2026-08-30T12:30:54Z", "LV", "Mack Hollins", "00-0000004", "WR", 4),
    ("2026-08-30T12:30:54Z", "GB", "Slot Receiver", "00-0000005", "WR", 3),
    ("2026-08-30T12:30:54Z", "ARI", "Trey Benson", "00-0000006", "RB", 7),
]


@pytest.fixture
def index():
    return depth.build_index(_chart(ROWS))


# --- the gate -----------------------------------------------------------------

def test_a_backup_quarterback_is_removed_from_the_passing_market(index):
    """THE ACTUAL BUG. One quarterback takes essentially every drop-back, so a
    QB2 is usually none of this market rather than a smaller share of it."""
    ok, why = depth.verdict("passing_yards", index["00-0000002"])
    assert ok is False
    assert "QB2" in why and "drop-back" in why


def test_the_starting_quarterback_is_kept(index):
    ok, _ = depth.verdict("passing_yards", index["00-0000001"])
    assert ok is True


def test_a_third_string_quarterback_is_removed(index):
    assert depth.verdict("passing_yards", index["00-0000003"])[0] is False


def test_a_wr3_is_kept_because_that_market_genuinely_shares(index):
    """A WR3 plays real snaps. Only the passing market is winner-take-all."""
    assert depth.verdict("receiving_yards", index["00-0000005"])[0] is True


def test_a_wr4_is_removed_from_receiving(index):
    """Mack Hollins at WR4 was the highest receiving pick on the board."""
    ok, why = depth.verdict("receiving_yards", index["00-0000004"])
    assert ok is False and "WR4" in why


def test_a_deep_reserve_back_is_removed_from_rushing(index):
    assert depth.verdict("rushing_yards", index["00-0000006"])[0] is False


def test_a_qb2_is_NOT_removed_from_a_rushing_market(index):
    """The cap is per market. A quarterback who does not start still has a
    rushing line only if he plays, but that is the passing gate's job -- this
    test pins that the caps are not applied across markets by accident."""
    assert depth.verdict("rushing_yards", index["00-0000002"])[0] is True


# --- the safety properties ----------------------------------------------------

def test_a_player_absent_from_the_chart_is_KEPT(index):
    """ABSENCE IS NOT EVIDENCE. Dropping on absence is exactly how 177 current
    players were deleted in August; a thin chart at one position must not be
    read as proof a man was cut."""
    ok, why = depth.verdict("passing_yards", None)
    assert ok is True and why == ""


def test_a_player_with_an_unreadable_rank_is_kept():
    """A rank that will not parse is not a rank, and must not remove anyone."""
    idx = depth.build_index(_chart([
        ("2026-08-30T12:30:54Z", "WAS", "Odd One", "00-0000009", "QB", "n/a")]))
    assert "00-0000009" not in idx
    assert depth.verdict("passing_yards", idx.get("00-0000009"))[0] is True


def test_a_market_with_no_cap_never_removes_anyone(index):
    assert depth.verdict("some_future_market", index["00-0000006"])[0] is True


# --- corroboration ------------------------------------------------------------

def test_a_chart_that_recognises_the_board_is_trusted(index):
    known = [r[3] for r in ROWS]
    trusted, rate, size = depth.usable(index, known)
    assert trusted is True and rate == 1.0 and size == len(ROWS)


def test_a_chart_that_does_not_recognise_the_board_is_refused(index):
    """The rosters.corroborates lesson: a file that does not contain the players
    we independently know are active is more likely broken than right."""
    strangers = [f"00-999{i:04d}" for i in range(20)]
    trusted, rate, _ = depth.usable(index, strangers)
    assert trusted is False and rate == 0.0


def test_the_bar_is_the_configured_one(index):
    known = [r[3] for r in ROWS] + [f"00-888{i:04d}" for i in range(len(ROWS))]
    trusted, rate, _ = depth.usable(index, known)
    assert rate == pytest.approx(0.5)
    assert trusted is (0.5 >= config.MIN_DEPTH_COVERAGE)


def test_an_empty_chart_is_never_trusted():
    trusted, rate, size = depth.usable({}, ["00-0000001"])
    assert trusted is False and rate == 0.0 and size == 0


# --- the loader ---------------------------------------------------------------

def test_only_the_latest_snapshot_is_used(monkeypatch):
    """The file holds every snapshot of the season -- 485k rows in 2026. An older
    one would reinstate players who have since been cut, which is the exact
    staleness this is here to remove."""
    frame = _chart([
        ("2026-07-01T00:00:00Z", "WAS", "Cut Guy", "00-0000077", "QB", 1),
        ("2026-08-30T12:30:54Z", "WAS", "Current Guy", "00-0000001", "QB", 1),
    ])
    monkeypatch.setattr(data, "_read_csv", lambda *a, **k: frame)
    out = data.depth_charts(2026)
    assert list(out["gsis_id"]) == ["00-0000001"]


def test_a_player_listed_twice_keeps_his_best_rank(monkeypatch):
    """A returner shows up at WR and KR. Being a starter somewhere is what
    decides whether he plays."""
    frame = _chart([
        ("2026-08-30T12:30:54Z", "GB", "Two Hats", "00-0000050", "KR", 4),
        ("2026-08-30T12:30:54Z", "GB", "Two Hats", "00-0000050", "WR", 1),
    ])
    monkeypatch.setattr(data, "_read_csv", lambda *a, **k: frame)
    out = data.depth_charts(2026)
    assert len(out) == 1 and int(out.iloc[0]["pos_rank"]) == 1


def test_a_missing_depth_file_degrades_to_no_gate(monkeypatch):
    """No chart must mean no filtering, never an empty board."""
    def boom(*a, **k):
        raise RuntimeError("404")
    monkeypatch.setattr(data, "_read_csv", boom)
    out = data.depth_charts(2026)
    assert out.empty
    assert depth.build_index(out) == {}
    assert depth.usable({}, ["00-0000001"])[0] is False


# --- the card -----------------------------------------------------------------

def test_the_depth_label_is_published_for_kept_players(index):
    pick = depth.annotate({"player": "Starter Guy"}, index["00-0000001"])
    assert pick["depth_label"] == "QB1"
    assert pick["depth_pos"] == "QB" and pick["depth_rank"] == 1


def test_an_unknown_player_gets_a_null_label_not_a_crash():
    pick = depth.annotate({"player": "Nobody"}, None)
    assert pick["depth_label"] is None and pick["depth_rank"] is None
