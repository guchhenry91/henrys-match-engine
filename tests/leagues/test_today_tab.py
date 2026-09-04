"""The Today tab must show every game played today, tagged with the band the
record is actually kept by.

TWO WAYS THIS GOES WRONG QUIETLY.

  A SECOND CONFIDENCE SCHEME. index.html already had two: tierOf() bands at .75
  and .68 into Elite/Strong/Lean, while every payload's `confidence` field and
  the Grades tab's calibration use publish._confidence()'s .70/.60/.50/.40. They
  disagree on real picks -- 0.72 is "Elite" to one and "Strong" to the other, and
  0.55 is "Even" to one and "Lean" to the other. A tag whose accuracy has never
  been measured is worse than no tag, so the Today rows use the measured bands
  and this test stops the client-side copy drifting from the Python.

  A COMPETITION QUIETLY MISSING. The tab is a completeness claim: it says these
  are ALL of today's games. A sport that renders nothing looks identical to a
  sport that failed to load, which is why every competition draws a section even
  when empty -- and why a new sport must be added here too.
"""
import pathlib
import re

import pytest

from leagues import config
from leagues.publish import _confidence

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _conf_of(p: float) -> int:
    """The JS confOf(), read out of index.html rather than restated here."""
    body = re.search(r"function confOf\(p\)\{(.*?)\n\}", HTML, re.S)
    assert body, "confOf() not found in index.html"
    thresholds = [(float(a), int(b)) for a, b in
                  re.findall(r"v>=\.(\d+)\?(\d)", body.group(1).replace("v>=.", "v>=."))]
    # re-read as .70 -> 0.70
    thresholds = [(t / 100.0, band) for t, band in thresholds]
    assert thresholds, "no thresholds parsed from confOf()"
    for threshold, band in thresholds:
        if p >= threshold:
            return band
    return 1


@pytest.mark.parametrize("p", [0.0, 0.11, 0.39, 0.40, 0.41, 0.49, 0.50, 0.51,
                               0.59, 0.60, 0.61, 0.69, 0.70, 0.71, 0.95, 1.0])
def test_the_js_bands_match_the_python_the_record_uses(p):
    """THE ONE THAT KEEPS THE LABEL HONEST. If these drift, the Today tab says
    "Strong" about a pick the Grades tab counts as Elite, and the calibration
    chart silently describes different labels than the board shows."""
    assert _conf_of(p) == _confidence(p), (
        f"p={p}: index.html says band {_conf_of(p)}, "
        f"leagues.publish._confidence says {_confidence(p)}")


def test_every_league_the_engine_publishes_gets_a_section():
    """Serie A was added as a fifth league and three separate hardcoded lists had
    to be found. The Today tab must derive its football sections from FILES."""
    fixtures = re.search(r"function todayFixtures\(\)\{(.*?)\n\}", HTML, re.S)
    assert fixtures, "todayFixtures() not found"
    body = fixtures.group(1)
    assert "Object.keys(FILES)" in body, (
        "football sections must come from FILES, not a literal list")
    files = re.search(r"const FILES=\{(.*?)\};", HTML, re.S).group(1)
    for league in config.LEAGUES:
        assert league in files, f"{league} missing from FILES"


@pytest.mark.parametrize("sport", ["ucl", "nfl", "nba"])
def test_the_other_sports_are_each_read(sport):
    body = re.search(r"function todayFixtures\(\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert f"DATA.{sport}" in body, f"{sport} is never read by todayFixtures()"


def test_nba_says_it_cannot_see_its_fixtures_rather_than_no_games():
    """NBA has a model and fifteen seasons behind it but no fixture feed. "No
    games today" would be a lie: there are games, the board cannot see them."""
    body = re.search(r"function todayFixtures\(\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert "evidence_only" in body
    assert "No fixture feed" in body


def test_an_empty_competition_still_draws_a_section():
    """A section that vanishes on a quiet day is indistinguishable from one that
    broke, and the tab's whole claim is completeness."""
    view = re.search(r"function viewToday\(\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert "No games today" in view


def test_the_hero_is_only_shown_when_it_is_actually_today():
    """Heroing a fixture three days out on a tab called Today is the small
    dishonesty that makes a reader distrust the rest of the page."""
    view = re.search(r"function viewToday\(\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert re.search(r"up\.find\(\s*u\s*=>\s*isToday\(u\.date\)\s*\)", view), (
        "the hero must be selected with isToday(), not taken as upcoming[0]")
