"""Phase 3: players who inherit an absent teammate's volume are underrated.

THE EVIDENCE (walk-forward, 2023-2025, games the model never saw). When 20%+ of a
team's targets or carries belonged to players absent from the game, the players
left hit their yardage overs far more often than the model said -- receiving
40.1% predicted vs 46.8% landed, rushing 42.1% vs 48.5%, passing 40.1% vs 54.7%.
Feeding the model the vacated share as a feature closed little of it: the prop
model blends 50/50 with an empirical baseline that knows nothing about who is
missing, which halves any signal.

THE FIX is a one-parameter correction AFTER the model, in logit space:

    p' = sigmoid(logit(p) + c * boost),   boost = form5 * v/(1-v) / line

where v is the share of the team's volume left by absent teammates, and form5 * v/(1-v)
is this player's slice of it if it is shared out in proportion to what the rest
already get. c is fitted per market by scripts/fit_nfl_vacancy.py on out-of-sample
walk-forward predictions only; season by season it lowered Brier overall for all
three yards markets (rushing in 3 of 3 seasons, receiving and passing in 2 of 3).
Anytime touchdowns showed no such gap (25.7% predicted vs 25.6%) and are left alone.

With nobody missing, boost is 0 and nothing changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data-raw" / "nfl" / "vacancy_adjust.json"
MARKETS = ("receiving_yards", "rushing_yards", "passing_yards")


def load() -> dict:
    try:
        raw = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {m: float(v["c"]) for m, v in (raw.get("markets") or {}).items()
            if isinstance(v, dict) and v.get("c") is not None}


def boost(form5, vacated, line):
    v = np.clip(np.asarray(vacated, dtype=float), 0.0, 0.9)
    line = np.asarray(line, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.asarray(form5, dtype=float) * (v / (1.0 - v)) / np.where(line > 0, line, np.nan)
    return np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)


def adjust(probs, form5, vacated, line, c: float | None):
    """The corrected over probability. Unchanged when c is None or nobody is out."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-4, 1 - 1e-4)
    if not c:
        return p
    z = np.log(p / (1 - p)) + float(c) * boost(form5, vacated, line)
    return 1.0 / (1.0 + np.exp(-z))
