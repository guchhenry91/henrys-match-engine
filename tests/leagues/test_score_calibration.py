"""Scoreline corrections must vary the board WITHOUT touching what is graded.

The six-score board printed 1-1 on 89% of Premier League fixtures. That was not
a bug -- 1-1 genuinely is the likeliest single score across almost the whole
realistic lambda range -- but a board that says the same thing six times is one
nobody can use. The fitted grid also turned out to be reliably biased: over
1,900 PL matches it over-predicts 0-0 by 35% and 1-1 by 10%, and under-predicts
2-2 by 20%.

Correcting that does NOT buy accuracy. Exact score tops out near 13% and the
board already scores 12.84% on a holdout. It buys variety at the same hit rate,
which is the entire point and must not be oversold as anything more.

The danger is scope creep: these factors must never reach the match model, whose
own calibration is measured and working.
"""
import numpy as np

from leagues.model import calibrated_grid, top_scorelines, score_for_outcome


def _grid():
    """A favourite at home: 1-1 leads, but 2-0 and 2-1 are close behind."""
    g = np.zeros((6, 6))
    g[1, 1] = 0.121
    g[1, 0] = 0.102
    g[2, 0] = 0.099
    g[2, 1] = 0.098
    g[0, 0] = 0.073
    g[0, 1] = 0.060
    g[3, 0] = 0.050
    g[2, 2] = 0.045
    # Spread the remaining mass thinly rather than parking it in one cell, or the
    # filler itself becomes the argmax and the fixture stops representing a real
    # score grid (which is flat, not spiky).
    rest = 1.0 - g.sum()
    zero = g == 0
    g[zero] = rest / zero.sum()
    return g


def test_no_correction_leaves_the_grid_untouched():
    g = _grid()
    assert np.allclose(calibrated_grid(g, None), g)
    assert top_scorelines(g, n=1)[0]["score"] == "1-1"


def test_a_correction_renormalises_to_one():
    corr = np.ones((6, 6)); corr[1, 1] = 0.9; corr[0, 0] = 0.74
    out = calibrated_grid(_grid(), corr)
    assert abs(out.sum() - 1.0) < 1e-9


def test_deflating_the_crowded_low_corner_can_change_the_pick():
    """The whole purpose: 1-1's lead is 2.3pp, so a real bias correction moves it."""
    corr = np.ones((6, 6))
    corr[1, 1] = 0.75          # measured over-prediction, exaggerated for the test
    corr[0, 0] = 0.74
    assert top_scorelines(_grid(), n=1, corr=corr)[0]["score"] != "1-1"


def test_cells_outside_the_correction_keep_their_weight():
    """The correction is a nudge to the low-score corner, not a truncation --
    a high scoreline must stay reachable."""
    small = np.ones((2, 2)) * 0.5
    out = calibrated_grid(_grid(), small)
    assert out[5, 5] > 0, "a cell beyond the correction's range was zeroed"


def test_score_for_outcome_still_respects_the_picked_outcome():
    """Calibration must not let the card contradict its own match pick."""
    corr = np.ones((6, 6)); corr[1, 1] = 0.5
    for outcome, test in (("home", lambda h, a: h > a),
                          ("away", lambda h, a: h < a),
                          ("draw", lambda h, a: h == a)):
        s = score_for_outcome(_grid(), outcome, corr=corr)
        h, a = (int(x) for x in s.split("-"))
        assert test(h, a), f"{outcome} pick produced {s}"


def test_probabilities_reported_are_the_calibrated_ones():
    corr = np.ones((6, 6)); corr[1, 1] = 0.5
    tops = top_scorelines(_grid(), n=3, corr=corr)
    assert all(0 < t["pct"] <= 100 for t in tops)
    assert tops[0]["pct"] >= tops[1]["pct"] >= tops[2]["pct"]
