"""How the PUBLISHED picks fare, as opposed to every row the gate scores.

A board publishes only the model's most confident picks -- 50%+, the top few per
game. Choosing the highest numbers selects exactly the rows the model most
overrates (the winner's curse), so the published subset lands below its stated
probability even when the model is calibrated on every row. Measured 2026-09-27:
NBA points 74.8% stated -> 72.6% landed; NFL receiving 57.7% -> 56.6%.

`shrink` is fitted by maximum likelihood on that subset:
    published p = 0.5 + shrink * (p - 0.5)
so 1.0 means no correction and 0.9 pulls every published number 10% of the way
toward a coin flip. The boards apply it to what they publish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRID = np.round(np.arange(0.50, 1.201, 0.01), 2)


def board_subset(pred: pd.DataFrame, group_cols, prob="prob", top=3, bar=0.5) -> pd.DataFrame:
    chosen = pred[pred[prob] >= bar].sort_values(prob, ascending=False)
    return chosen.groupby(list(group_cols), sort=False).head(top)


def fit_shrink(p: np.ndarray, o: np.ndarray) -> float:
    p, o = np.asarray(p, float), np.asarray(o, float)
    best, best_ll = 1.0, -np.inf
    for s in GRID:
        q = np.clip(0.5 + s * (p - 0.5), 1e-4, 1 - 1e-4)
        ll = float(np.sum(o * np.log(q) + (1 - o) * np.log(1 - q)))
        if ll > best_ll:
            best, best_ll = float(s), ll
    return best


def evaluate(pred: pd.DataFrame, group_cols, prob="prob", outcome="outcome") -> dict:
    sub = board_subset(pred, group_cols, prob)
    if len(sub) < 200:
        return {"n": int(len(sub)), "shrink": 1.0, "note": "too few to fit; no correction"}
    p, o = sub[prob].to_numpy(float), sub[outcome].to_numpy(float)
    s = fit_shrink(p, o)
    return {"n": int(len(sub)), "predicted": round(float(p.mean()), 4),
            "landed": round(float(o.mean()), 4), "shrink": s,
            "after_shrink": round(float((0.5 + s * (p - 0.5)).mean()), 4)}


def apply(p: float, shrink: float) -> float:
    return float(min(max(0.5 + shrink * (p - 0.5), 0.01), 0.99))


def sides(pick: dict, quote: dict, shrink: float = 1.0, allow_under: bool = True) -> list:
    """The over and (where the book prices it) the under, from one over probability.

    `pick` carries the model's OVER probability at the book's line in
    "probability". The published probability is corrected by `shrink` (fitted on
    the published subset in the gate), the raw number kept as p_model. The under
    is the same model's other side at the book's under price; only one side can
    clear 50%, so the board keeps whichever the model actually favours.
    """
    raw = float(pick["probability"])
    p = apply(raw, shrink)
    over = dict(pick, side="over", p_model=round(raw, 4), probability=round(p, 4))
    if over.get("book_p") is not None:
        over["edge"] = round(p - float(over["book_p"]), 4)
    out = [over]
    price_u = (quote or {}).get("odd_under")
    if allow_under and price_u:
        book_u = quote.get("under")
        if book_u is None and quote.get("over") is not None:
            book_u = 1.0 - float(quote["over"])
        under = dict(pick, side="under", p_model=round(1.0 - raw, 4),
                     probability=round(1.0 - p, 4), book_price=float(price_u),
                     book_p=None if book_u is None else round(float(book_u), 4))
        under["edge"] = (round(under["probability"] - under["book_p"], 4)
                         if under["book_p"] is not None else None)
        out.append(under)
    return out
