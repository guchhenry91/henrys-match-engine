"""One tap for all overs or all unders on every board that publishes both."""
import pathlib

HTML = (pathlib.Path(__file__).resolve().parents[2] / "index.html").read_text(encoding="utf-8")


def test_the_side_switch_exists_and_treats_yes_bets_as_overs():
    assert "function sideChips()" in HTML and "setFSide('under')" not in HTML  # built from a helper
    assert 'const inSide=p=>fside==="all"||(p.side||"over")===fside;' in HTML


def test_nfl_nba_and_mlb_props_all_honour_it():
    assert HTML.count(".filter(inBoard).filter(inSide)") == 3
    assert HTML.count("sideChips()") >= 4          # definition + NFL, NBA, MLB rows


def test_mlb_has_a_rail_with_every_market_and_both_sides():
    for market in ("hits", "hrr", "hr", "rbi", "strikeouts", "team_runs"):
        assert f'id:"mlb:{market}", view:"MLB", filter:"{market}"' in HTML
    assert '...props("MLB"),...sides("MLB")' in HTML
    assert '...sides("NFL")' in HTML and '...sides("NBA")' in HTML
    assert "if(t.side) fside=t.side;" in HTML
