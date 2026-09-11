"""The page shell: sport first (sidebar), then that sport's markets (tab row).

Every tab must open a real board, and every sport the Results tab can grade must
have a tab that opens it already scoped -- a tab that lands on the wrong sport's
record is the pooled list this layout replaced.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _body(name):
    m = re.search(rf"function {name}\(.*?\)\{{(.*?)\n\}}", HTML, re.S)
    assert m, f"{name}() not found"
    return m.group(1)


def test_every_sport_is_in_the_sidebar():
    block = re.search(r"const SPORTS=\[(.*?)\];", HTML, re.S).group(1)
    for key in ("soccer", "nfl", "nba", "ucl"):
        assert f'"{key}"' in block


def test_every_tab_opens_a_real_view():
    views = set(re.findall(r'k:"([^"]+)"', re.search(r"const views=\[(.*?)\];", HTML, re.S).group(1)))
    tabs = set(re.findall(r'\{k:"([^"]+)"', _body("tabsFor")))
    assert tabs <= views, f"tabs pointing at unknown views: {tabs - views}"


def test_each_sport_has_a_results_tab_scoped_to_it():
    body = _body("tabsFor")
    for key in ("soccer", "nfl", "nba", "ucl"):
        assert f'gs:"{key}"' in body, f"{key} has no Results tab"


def test_nfl_and_nba_markets_come_from_props():
    """Not a second hand-written list, so a renamed market cannot orphan a tab."""
    body = _body("tabsFor")
    assert 'props("NFL")' in body and 'props("NBA")' in body


def test_render_draws_the_chrome():
    assert "chrome()" in _body("render")
