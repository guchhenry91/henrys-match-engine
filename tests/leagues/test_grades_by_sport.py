"""The Grades tab is scoped by sport first, then by market.

WHY. Every sport's record was poured into one list: NFL team winners and
Champions League picks sat in "Every match pick" beside Premier League games, and
NFL props sat among soccer's. Each sport runs its own model on its own markets,
so a pooled count describes none of them. The tab now picks a sport, then that
sport's markets -- the same two-level filter the reader uses to narrow a board.

ALSO HERE: the Champions League settled list published every pick as
"None v None" with no score, because it joined club names from the UPCOMING
fixture list, which a match leaves the moment it is played.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _code(name):
    m = re.search(rf"function {name}\(.*?\)\{{(.*?)\n\}}", HTML, re.S)
    assert m, f"{name}() not found in index.html"
    return "\n".join(re.sub(r"//.*$", "", l) for l in m.group(1).splitlines())


def test_every_sport_has_a_scope():
    block = re.search(r"const GSPORTS=\[(.*?)\];", HTML, re.S).group(1)
    for key in ("soccer", "nfl", "ucl", "nba"):
        assert f'"{key}"' in block, f"{key} has no grading scope"


def test_soccer_lists_hold_no_other_sport():
    """THE ONE THAT WAS WRONG. NFL and UCL rows used to be merged into soccer's
    match and prop lists unconditionally."""
    props, matches = _code("gradedProps"), _code("gradedMatches")
    assert 'sport==="nfl"' in props, "NFL props must only load under the NFL scope"
    assert 'sport==="ucl"' in matches and 'sport==="nfl"' in matches, (
        "UCL and NFL matches must only load under their own scope")


def test_each_sport_renders_its_own_summary():
    view = _code("viewGrades")
    for fn in ("nflGradeSummary", "uclGradeSummary", "nbaGradeSummary"):
        assert f"{fn}()" in view, f"viewGrades never renders {fn}"


def test_filters_are_labelled_by_level():
    """A sport chip and a market chip must never read as the same kind of
    choice, so each row carries its own label."""
    body = _code("gradeFilters")
    assert 'frow("Sport"' in body and 'frow("Market"' in body


def test_leaving_the_tab_resets_the_sport():
    body = _code("go")
    assert 'gsport="soccer"' in body


def test_an_empty_sport_explains_itself():
    """NBA publishes evidence and no picks. An empty panel there would look like
    a failed load; it has to say why there is nothing to grade."""
    assert "gradeEmpty(" in _code("nbaGradeSummary")


def test_ucl_settled_names_played_matches(monkeypatch):
    """A played fixture is no longer in `matches`, so its teams and score must
    come from the finished results, keyed by the same fixture id."""
    from ucl import publish
    monkeypatch.setattr(publish.data, "results_by_id", lambda *a, **k: {
        "1635609": {"home": "AEK Athens FC", "away": "Lask Linz",
                    "home_goals": 1, "away_goals": 0}})
    log = {"1635609": {"graded": "correct", "pick": "AEK Athens FC",
                       "p_pick": 0.57, "kickoff": "2026-09-08T16:45:00+00:00"},
           "_meta": {"not": "a pick"},
           "999": {"pick": "X", "kickoff": "2026-09-30T19:00:00+00:00"}}   # pending
    out = publish.settled(log, matches=[])
    assert len(out) == 1, "pending picks and metadata keys must be skipped"
    row = out[0]
    assert (row["home"], row["away"]) == ("AEK Athens FC", "Lask Linz")
    assert row["result"] == {"home_goals": 1, "away_goals": 0}
