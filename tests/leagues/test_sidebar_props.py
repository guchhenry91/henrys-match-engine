"""Every sidebar prop link must point at a market that actually exists.

The codebase's own rule: "a link that looks live and does nothing is worse than no
link" -- which is exactly what the NFL chips were when setFilter rendered a
hardcoded tab. These links are one click from the front page, so a market renamed
in the engine must fail here rather than leave a dead row on the sidebar.

Asserted against the engines' own definitions (nfl.config.MARKETS and the soccer
player markets), not against a second hand-written list.
"""
import json
import pathlib
import re

import pytest

from nfl import config as nfl_config

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")

# The soccer player markets, as publish.py names them on each pick.
SOCCER_MARKETS = {"goal", "shots", "sot"}


def _props():
    block = re.search(r"const PROPS=\[(.*?)\n\];", HTML, re.S)
    assert block, "PROPS list not found in index.html"
    out = []
    for m in re.finditer(r"\{([^{}]*)\}", block.group(1)):
        row = m.group(1)
        item = {}
        for key in ("id", "view", "filter", "t"):
            found = re.search(rf'{key}:\s*"([^"]*)"', row)
            if found:
                item[key] = found.group(1)
        out.append(item)
    return out


def _view_keys():
    block = re.search(r"const views=\[(.*?)\];", HTML, re.S)
    assert block, "views list not found"
    return {m.group(1) for m in re.finditer(r'k:"([^"]+)"', block.group(1))}


def test_there_are_prop_links():
    assert len(_props()) >= 7


def test_every_prop_link_opens_a_real_view():
    views = _view_keys()
    bad = [p for p in _props() if p["view"] not in views]
    assert not bad, f"prop links pointing at unknown views: {bad}"


def test_every_soccer_prop_names_a_real_market():
    bad = [p for p in _props()
           if p["view"] == "players" and p["filter"] not in SOCCER_MARKETS]
    assert not bad, f"unknown soccer markets: {bad}"


def test_all_three_soccer_markets_are_reachable():
    got = {p["filter"] for p in _props() if p["view"] == "players"}
    assert got == SOCCER_MARKETS


def test_every_nfl_prop_names_a_real_market():
    """Tied to nfl.config.MARKETS, so renaming a market there breaks this rather
    than silently orphaning a sidebar row."""
    allowed = set(nfl_config.MARKETS) | {"team_winner"}
    bad = [p for p in _props() if p["view"] == "NFL" and p["filter"] not in allowed]
    assert not bad, f"unknown NFL markets: {bad}"


def test_every_nfl_market_is_reachable():
    got = {p["filter"] for p in _props() if p["view"] == "NFL"}
    missing = (set(nfl_config.MARKETS) | {"team_winner"}) - got
    assert not missing, f"NFL markets with no sidebar link: {missing}"


def test_prop_ids_are_unique():
    ids = [p["id"] for p in _props()]
    assert len(ids) == len(set(ids))


def test_nba_links_exist_exactly_when_there_is_an_nba_BOARD():
    """THE RULE IS ABOUT DEAD LINKS, not about the package.

    This originally asserted no `nba/` directory existed, and it fired the moment
    the engine was built -- correctly flagging that something had changed, but
    testing the wrong thing. The engine and its fifteen-season backtest can exist
    long before anything is published; what must never happen is a sidebar row
    pointing at a board the site does not serve.

    So the invariant is the published payload, not the source tree: links exactly
    when `data/nba/board.json` exists.
    """
    has_board = (ROOT / "data" / "nba" / "board.json").exists()
    has_links = any("nba" in p["id"].lower() for p in _props())
    assert has_links == has_board, (
        "an NBA board is published but has no sidebar links"
        if has_board else
        "sidebar links point at an NBA board that is not published")


def test_the_props_group_is_rendered_by_the_sidebar():
    assert "props:true" in HTML.replace(" ", "")
    assert "b.dataset.prop=item.id" in HTML.replace(" ", "").replace("\n", "") or \
           "b.dataset.prop = item.id" in HTML


def test_go_accepts_a_market_so_a_link_can_land_on_one():
    assert re.search(r"function go\(k,\s*f\)", HTML), \
        "go() must take an optional market for the prop links to work"
