"""A pick frozen before kick-off stays on the board until graded (2026-10-18: the
whole day's NFL picks vanished from the site once their games began)."""
from nfl import forecast_board as fb


def _row(**kw):
    base = {"game_id": "2026_07_A_B", "market": "receiving_yards", "player_id": "p1",
            "player": "X", "side": "over", "line": 50.5, "probability": 0.52,
            "p_model": 0.60, "book_p": 0.50, "book_price": 1.9,
            "kickoff": "2020-01-01T17:00:00+00:00"}
    return {**base, **kw}


def test_a_frozen_pick_stays_when_its_game_starts():
    frozen = {"2026_07_A_B:receiving_yards:p1": {"side": "over", "line": 49.5, "p_pick": 0.55,
                                                 "p_model": 0.66, "pick_kind": "proving"}}
    out = fb.select([_row()], frozen)
    assert len(out) == 1 and out[0]["in_play"] and out[0]["line"] == 49.5
    assert out[0]["probability"] == 0.55


def test_a_flipped_lean_still_finds_the_frozen_side():
    frozen = {"2026_07_A_B:receiving_yards:p1:under": {"side": "under", "line": 50.5, "p_pick": 0.56}}
    out = fb.select([_row(side="over")], frozen)
    assert len(out) == 1 and out[0]["side"] == "under"


def test_an_unfrozen_pick_still_leaves_when_its_game_starts():
    assert fb.select([_row(p_model=0.70)], {}) == []


def test_a_graded_pick_leaves():
    frozen = {"2026_07_A_B:receiving_yards:p1": {"side": "over", "graded": "correct"}}
    assert fb.select([_row()], frozen) == []


def test_a_proving_pick_needs_the_model_at_60():
    """Owner, 2026-10-19: proving picks at 60% on the model's own chance."""
    assert fb._clears({"pick_kind": "proving", "p_model": 0.605, "probability": 0.52, "book_p": 0.5})
    assert not fb._clears({"pick_kind": "proving", "p_model": 0.595, "probability": 0.52, "book_p": 0.5})


def test_a_player_back_from_missed_games_is_forecast_lower():
    """Walk-forward audit 2026-10-20: returning players were overrated (receiving
    +17%, TD +29%); the forecast is scaled by the measured ratio."""
    import numpy as np
    import pandas as pd
    from nfl import forecast as fc

    class Stub:
        def predict(self, x):
            return np.full(len(x), 50.0)
    m = fc.MarketModel("receiving_yards")
    m.model = Stub()
    rows = pd.DataFrame({c: [0.0, 0.0] for c in m.cols}).assign(returning=[0.0, 1.0])
    mu = m.mean(rows)
    assert mu[0] == 50.0 and abs(mu[1] - 50.0 * fc.RETURN_ADJ["receiving_yards"]) < 1e-9
