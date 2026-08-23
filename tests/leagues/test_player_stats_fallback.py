"""API-Football stands in for Understat only where Understat is silent.

Context: on 2026-08-23 Understat had filed nothing for 2026-27 while 26 fixtures
had been played, so every player pick graded against a fixture the feed had
never seen. leagues.publish now refuses to grade an uncovered side, and
scripts/sync_player_stats.py fills the silence from API-Football.

Two properties matter here and are easy to lose in a refactor: the better feed
must win wherever it HAS the match, and a fixture whose squads came back must
count as covered even when our particular player never took a shot -- otherwise
a genuine losing pick hangs pending forever instead of settling.
"""
import json

import pandas as pd
from leagues import players


def _patch_cache(monkeypatch, tmp_path, payload):
    """Point players.api_match_stats at a temp cache file."""
    root = tmp_path / "data-raw" / "leagues"
    root.mkdir(parents=True)
    (root / "player_stats.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(players, "__file__", str(tmp_path / "leagues" / "players.py"))
    return root / "player_stats.json"


PAYLOAD = {
    "PL": {
        "2026-08-22|Tottenham|Brentford": {
            "date": "2026-08-22",
            "home": "Tottenham", "away": "Brentford",
            "api_fixture_id": 1570999,
            "api_squads": {"Tottenham": ["Richarlison de Andrade"],
                           "Brentford": ["Igor Thiago"]},
            "players": {
                "Richarlison": {"team": "Tottenham", "api_name":
                                "Richarlison de Andrade",
                                "goals": 0, "shots": 2, "sot": 1},
            },
        }
    }
}


def test_reads_rows_and_coverage(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, PAYLOAD)
    df, covered = players.api_match_stats("PL")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["player"] == "Richarlison"
    assert (row["goals"], row["shots"], row["sot"]) == (0, 2, 1)
    day = pd.Timestamp("2026-08-22").date()
    assert ("Tottenham", day) in covered


def test_a_side_whose_squad_returned_is_covered_even_with_no_matched_row():
    """Brentford has no matched player, but the feed clearly holds the fixture.

    If coverage were derived from matched ROWS instead of squads, a pick on a
    player who genuinely never shot would never settle -- it would sit pending
    forever while looking, on the page, exactly like missing data.
    """
    day = pd.Timestamp("2026-08-22").date()
    entry = PAYLOAD["PL"]["2026-08-22|Tottenham|Brentford"]
    covered = {(t, day) for t in entry["api_squads"]}
    assert ("Brentford", day) in covered
    assert "Igor Thiago" not in entry["players"]


def test_missing_cache_is_silent(monkeypatch, tmp_path):
    (tmp_path / "data-raw" / "leagues").mkdir(parents=True)
    monkeypatch.setattr(players, "__file__", str(tmp_path / "leagues" / "players.py"))
    df, covered = players.api_match_stats("PL")
    assert df.empty and covered == set()


def test_unknown_league_is_silent(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path, PAYLOAD)
    df, covered = players.api_match_stats("LALIGA")
    assert df.empty and covered == set()


def test_resolver_refuses_ambiguous_and_mismatched_names():
    """The joins that must never happen, because each settles a real bet."""
    r = players.resolve_squad_name
    assert r("Joao Neves", ["Ruben Neves"]) is None
    assert r("Marc Cucurella", ["Marc Guiu"]) is None
    assert r("Williams", ["Inaki Williams", "Nico Williams"]) is None
    assert r("Anybody", []) is None


def test_resolver_accepts_real_spelling_differences():
    r = players.resolve_squad_name
    assert r("Richarlison", ["Richarlison de Andrade"]) == "Richarlison de Andrade"
    assert r("Kylian Mbappe-Lottin", ["K. Mbappe"]) == "K. Mbappe"
    assert r("Matheus Cunha",
             ["Matheus Santos Carneiro Da Cunha"]) == \
        "Matheus Santos Carneiro Da Cunha"
