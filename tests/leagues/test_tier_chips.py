"""The confidence-tier chips, and the thresholds they must not drift from.

WHERE THE NAMES COME FROM. Strong/Solid/Lean are `nflTier`'s, already on the NFL
and Champions League rows at 70/60, so on those tabs the chip and the rows under
it say the same word. Thin is the one addition, splitting nflTier's catch-all
"Lean" at 50.

WHAT IS DELIBERATELY NOT ALIGNED. Two other vocabularies exist and both stay:
publish.py's _confidence() bands (Elite/Strong/Even/Lean/Longshot at 70/60/50/40)
because the RECORD and the calibration chart are kept by them, and `tierOf` on
the Best rows (75/68). That was an explicit choice, so these tests pin the chip
thresholds to _confidence() and let the labels differ -- what must never drift is
which PICKS land in a chip, because that is what the reader is filtering.
"""
import pathlib
import re

import pytest

from leagues.publish import _confidence

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = (ROOT / "index.html").read_text(encoding="utf-8")


def _tiers():
    block = re.search(r"const TIERS=\[(.*?)\n\];", HTML, re.S)
    assert block, "TIERS not found in index.html"
    rows = re.findall(r'\["(\w+)",\s*"([^"]+)",\s*"([^"]+)",\s*b=>([^\]]+)\]',
                      block.group(1))
    assert len(rows) == 4, f"expected four tiers, got {rows}"
    return rows


def test_there_are_four_tiers_named_as_asked():
    assert [r[1] for r in _tiers()] == ["Strong", "Solid", "Lean", "Thin"]


def test_the_tiers_partition_every_band_exactly_once():
    """A band belonging to no chip is a pick the reader cannot reach; a band in
    two chips double-counts it."""
    tests = {
        "b>=5": lambda b: b >= 5,
        "b===4": lambda b: b == 4,
        "b===3": lambda b: b == 3,
        "b<=2": lambda b: b <= 2,
    }
    preds = []
    for _, _, _, expr in _tiers():
        expr = expr.strip()
        assert expr in tests, f"unrecognised tier predicate {expr!r}"
        preds.append(tests[expr])
    for band in (1, 2, 3, 4, 5):
        hits = sum(1 for p in preds if p(band))
        assert hits == 1, f"band {band} matched {hits} tiers, must match exactly 1"


@pytest.mark.parametrize("p,expected", [
    (0.95, "Strong"), (0.70, "Strong"),
    (0.699, "Solid"), (0.60, "Solid"),
    (0.599, "Lean"), (0.50, "Lean"),
    (0.499, "Thin"), (0.40, "Thin"), (0.10, "Thin"),
])
def test_a_probability_lands_in_the_tier_its_band_implies(p, expected):
    """Pins the chips to _confidence(), the function the record is kept by. If
    these drift, a reader filtering to "Strong" sees picks the record counts in a
    different band."""
    band = _confidence(p)
    rows = _tiers()
    tests = {"b>=5": band >= 5, "b===4": band == 4,
             "b===3": band == 3, "b<=2": band <= 2}
    got = [name for _, name, _, expr in rows if tests[expr.strip()]]
    assert got == [expected], f"p={p} (band {band}) landed in {got}"


def test_the_ranges_shown_match_the_thresholds():
    """The range is the only thing on screen that reconciles these names with the
    differently-named bands on Today and Best, so it must be true."""
    ranges = {name: rng for _, name, rng, _ in _tiers()}
    assert ranges["Strong"] == "70%+"
    assert ranges["Solid"] == "60–70%"
    assert ranges["Lean"] == "50–60%"
    assert ranges["Thin"] == "under 50%"


def test_the_range_is_on_the_chip_not_in_a_tooltip():
    """A title attribute is invisible on a phone, which is where this is read."""
    body = re.search(r"function tierChips\(items\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert 'class="rng"' in body
    assert "title=" not in body


def test_the_names_agree_with_nflTier_where_both_appear():
    """NFL and UCL rows are labelled by nflTier. If the chips said something else,
    a row tagged SOLID would sit under a chip called something different on the
    very same page."""
    body = re.search(r"function nflTier\(prob\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert "0.70" in body and '"Strong"' in body
    assert "0.60" in body and '"Solid"' in body
    assert '"Lean"' in body


def test_every_selected_sport_renders_the_chips():
    """Soccer (Best + Players), NFL, NBA and the Champions League were all asked
    for. NBA passes its picks list, which is empty until the fixture feed exists,
    so it renders nothing rather than four dead chips."""
    for view in ("viewBest", "viewPlayers", "viewNFL", "viewNBA", "viewUCL"):
        body = re.search(rf"function {view}\(\)\{{(.*?)\n\}}", HTML, re.S).group(1)
        assert "tierChips(" in body, f"{view} has no tier chips"


def test_an_empty_list_renders_no_chips_at_all():
    """The codebase's own rule: a control that looks live and does nothing is
    worse than no control."""
    body = re.search(r"function tierChips\(items\)\{(.*?)\n\}", HTML, re.S).group(1)
    assert re.search(r"if\(!items\|\|!items\.length\) return \"\"", body)


def test_tapping_the_active_chip_clears_the_filter():
    body = re.search(r"function setTier\(t\)\{(.*?)\}", HTML, re.S).group(1)
    assert 'tier===t?"all":t' in body.replace(" ", "")
