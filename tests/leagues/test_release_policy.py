"""publish.py reading the gate's generated artefacts.

The failure these guard against is silent by nature: a missing or malformed
policy file must fall back to the model's own defaults and keep publishing, and
a missing report must make the page DROP its backtested claim rather than show a
stale one.
"""
import json

import pytest

from leagues import publish
from leagues.model import XG_WEIGHT
from leagues.weights import XI_PER_DAY


@pytest.fixture
def picks_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "PICKS_DIR", tmp_path)
    return tmp_path


def _write(d, name, payload):
    (d / name).write_text(json.dumps(payload), encoding="utf-8")


# --- release policy ----------------------------------------------------------

def test_model_params_reads_the_generated_policy(picks_dir):
    _write(picks_dir, "release_policy.json",
           {"leagues": {"LALIGA": {"xi": 0.0018, "xg_weight": 1.0}}})
    assert publish.model_params("LALIGA") == {"xi": 0.0018, "xg_weight": 1.0}


def test_missing_policy_falls_back_to_model_defaults(picks_dir):
    """No gate artefact means nothing was promoted -- publish must still run, on
    the same parameters it used before the policy existed."""
    assert publish.model_params("PL") == {}
    from leagues.model import LeagueModel
    m = LeagueModel(**publish.model_params("PL"))
    assert (m.xi, m.xg_weight) == (XI_PER_DAY, XG_WEIGHT)


def test_league_absent_from_policy_falls_back(picks_dir):
    _write(picks_dir, "release_policy.json", {"leagues": {"PL": {"xi": 0.0045}}})
    assert publish.model_params("BUNDESLIGA") == {}


def test_non_numeric_policy_values_are_ignored_not_passed_through(picks_dir):
    """A null or string in the policy must not reach LeagueModel, where it would
    fail deep inside the fit instead of here."""
    _write(picks_dir, "release_policy.json",
           {"leagues": {"PL": {"xi": None, "xg_weight": "0.75"}}})
    assert publish.model_params("PL") == {}


def test_partial_policy_entry_supplies_only_what_it_has(picks_dir):
    _write(picks_dir, "release_policy.json", {"leagues": {"PL": {"xi": 0.0045}}})
    assert publish.model_params("PL") == {"xi": 0.0045}


# --- backtested tier stats ---------------------------------------------------

def test_tier_stats_come_from_the_report(picks_dir):
    _write(picks_dir, "backtest_report.json", {"_pooled": {"tiers": [
        {"min_prob": 0.0, "n": 3958, "hit_rate_pct": 53.1,
         "league_min_pct": 51.0, "league_max_pct": 55.0},
        {"min_prob": 0.65, "n": 402, "hit_rate_pct": 76.9,
         "league_min_pct": 70.7, "league_max_pct": 85.5},
    ]}})
    s = publish._backtested_tier_stats(0.65)
    assert s["backtested_hit_rate_pct"] == 76.9
    assert s["backtested_all_picks_pct"] == 53.1
    assert s["backtested_n"] == 402
    # The spread is the point: one pooled number hides a 15-point range.
    assert s["backtested_league_range_pct"] == [70.7, 85.5]


def test_missing_report_drops_the_claim_entirely(picks_dir):
    """Better to say nothing than to publish a hardcoded number that no longer
    matches the model -- which is what the 77.4 literal did."""
    assert publish._backtested_tier_stats(0.65) == {}


def test_unknown_tier_yields_no_hit_rate(picks_dir):
    _write(picks_dir, "backtest_report.json", {"_pooled": {"tiers": [
        {"min_prob": 0.0, "n": 100, "hit_rate_pct": 53.0,
         "league_min_pct": 53.0, "league_max_pct": 53.0}]}})
    s = publish._backtested_tier_stats(0.65)
    assert "backtested_hit_rate_pct" not in s
    assert s["backtested_all_picks_pct"] == 53.0


def test_tier_present_but_empty_is_not_reported(picks_dir):
    _write(picks_dir, "backtest_report.json", {"_pooled": {"tiers": [
        {"min_prob": 0.65, "n": 0, "hit_rate_pct": None,
         "league_min_pct": None, "league_max_pct": None}]}})
    assert publish._backtested_tier_stats(0.65) == {}
