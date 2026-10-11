"""The NFL prop board on the REBUILT model (docs/REBUILD_PLAN.md).

nfl/publish.player_projections still decides WHO is on the slate (rosters, depth
chart, injuries) and which bookmaker line each player has. This module then
replaces the old PropModel's number with the rebuilt one:

  1. forecast the player's game (nfl/forecast.py) -- the line is never an input;
  2. read the chance of beating the BOOK'S line off that forecast;
  3. combine with the book's own chance:
         final = sigmoid(a * logit(forecast) + logit(book) + c)
     The book's weight is FIXED at 1 (tracking/combine_fit.fit_offset): books set
     yardage lines at their median, near 50/50, and a free weight on a number that
     barely moves has nothing to fit. a and c per market come from the gate
     (data-raw/nfl/forecast_report.json), replaced by weights re-fitted on real
     graded lines once a market has 300 (data-raw/nfl/combine_weights.json);
  4. publish a side only at PUBLISH_BAR or above AND with the forecast on the same
     side; at most TOP_PER_MARKET a game.

What the gate found, said plainly: on a median-set yardage line the forecast moves
the book only a little (a = 0.06-0.18), so yardage picks are RARE -- only where the
forecast is very sure. Those it would have published landed 72-76% (passing,
receiving) in the 2022-25 walk-forward. Anytime TD: the forecast adds nothing over
a book that knows each player's touches (a = 0), so a TD pick is the book's own
favourite that the forecast AGREES with -- the forecast is a veto, not a booster.
Only the "yes" side is published: that is the bet a reader can place.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nfl import config
from nfl import forecast as fc

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data-raw" / "nfl" / "forecast_report.json"
PRICED_LOG = ROOT / "data-raw" / "nfl" / "priced_lines.json"
WEIGHTS = ROOT / "data-raw" / "nfl" / "combine_weights.json"
ENGINE = "forecast-v1"
PUBLISH_BAR = 0.62
TOP_PER_MARKET = 3
HISTORY_SEASONS = 6               # five to train on, plus one for entering features


def weights() -> dict:
    """{market: (a, b, c)}; adopted real-line weights win over the gate's."""
    out = {}
    try:
        markets = json.loads(REPORT.read_text(encoding="utf-8"))["markets"]
        out = {m: tuple(v["combined"]["weights_model_book_const"]) for m, v in markets.items()}
    except Exception:
        pass
    from tracking import combine_fit
    out.update(combine_fit.adopted(WEIGHTS))
    return out


def combine(p_model: float, p_book: float, w) -> float:
    a, b, c = w or (0.0, 1.0, 0.0)

    def lg(p):
        p = min(max(float(p), 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-(a * lg(p_model) + b * lg(p_book) + c)))


def frame(projections: list, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Built features: every player-week of history plus ONE upcoming row per
    (player, game) on the slate. `history` (fc.load output) may be passed in."""
    if history is None:
        first = config.CURRENT_SEASON - HISTORY_SEASONS
        history = fc.load(range(first, config.CURRENT_SEASON + 1))
    last = history.sort_values(["season", "week"]).groupby("player_id").tail(1).set_index("player_id")
    synth, seen = [], set()
    for p in projections:
        pid, gid = str(p.get("player_id")), str(p.get("game_id"))
        if (pid, gid) in seen or pid not in last.index:
            continue
        seen.add((pid, gid))
        try:
            season, week = int(gid.split("_")[0]), int(gid.split("_")[1])
        except (ValueError, IndexError):
            continue
        synth.append({"player_id": pid, "player_display_name": p.get("player"),
                      "position": last.at[pid, "position"], "season": season, "week": week,
                      "season_type": "REG", "game_id": gid, "team": p.get("team"),
                      "opponent_team": p.get("opponent"), "is_next": True})
    rows = history.assign(is_next=False)
    if synth:
        rows = pd.concat([rows, fc._context(pd.DataFrame(synth))], ignore_index=True)
    return fc.build(rows)


def _train(built: pd.DataFrame) -> pd.DataFrame:
    return built[~built["is_next"] & (built["season"] >= config.CURRENT_SEASON - fc.TRAIN_SEASONS)]


def select(priced: list) -> list:
    keep = sorted([p for p in priced if p["probability"] >= PUBLISH_BAR and p["p_model"] >= 0.5
                   and (p["market"] != "anytime_touchdown" or p["side"] == "over")],
                  key=lambda p: -p["probability"])
    by_game = {}
    for p in keep:
        by_game.setdefault(str(p["game_id"]), []).append(p)
    return sorted([p for ps in by_game.values() for p in ps[:TOP_PER_MARKET]],
                  key=lambda p: -p["probability"])


def price(market: str, projections: list, built: pd.DataFrame, w=None) -> list:
    """Re-price the old board's projections on the rebuilt model. Only players with a
    book's chance are priced; each gets side, probability, p_model, book_p, mu."""
    w = w if w is not None else weights().get(market)
    ask = built[built["is_next"]].copy()
    ask["_key"] = ask["player_id"].astype(str) + "|" + ask["game_id"].astype(str)
    ask = ask.drop_duplicates("_key").set_index("_key")
    todo = [p for p in projections if p.get("book_p") is not None
            and f"{p.get('player_id')}|{p.get('game_id')}" in ask.index
            and (market == "anytime_touchdown" or p.get("line") is not None)]
    if not todo:
        return []
    model = fc.MarketModel(market).fit(_train(built))
    rows = ask.loc[[f"{p['player_id']}|{p['game_id']}" for p in todo]]
    if market == "anytime_touchdown":
        p_over = model.p_over(rows, 0.5)
        mu = [None] * len(todo)
    else:
        mu = model.mean(rows)
        p_over = model.spread.p_over(mu, np.array([float(p["line"]) for p in todo]))
    out = []
    for p, pm, m in zip(todo, p_over, mu):
        pb = float(p["book_p"])
        pc = combine(float(pm), pb, w)
        side = "over" if pc >= 0.5 else "under"
        q = dict(p)
        q.pop("ladder", None)
        q.update({"side": side, "engine": ENGINE,
                  "probability": round(pc if side == "over" else 1 - pc, 4),
                  "p_model": round(float(pm) if side == "over" else 1 - float(pm), 4),
                  "book_p": round(pb if side == "over" else 1 - pb, 4),
                  "mu": None if m is None else round(float(m), 1)})
        if side == "under":
            q["book_price"] = p.get("book_price_under")
        q["edge"] = round(q["probability"] - q["book_p"], 4)
        out.append(q)
    out.sort(key=lambda p: -p["probability"])
    return out
