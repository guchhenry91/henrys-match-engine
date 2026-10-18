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
# BALANCED LINES (owner, 2026-10-18): a book line set near 50/50 can never reach 62%
# once the forecast is pulled toward the book (combine_fit.shrink) -- the board went
# empty. On a line whose book chance is under BALANCED_BOOK_MAX the bar is 55%: about
# the +3% value point at typical prices, and the pick must still be value or proving.
BALANCED_BAR = 0.55
BALANCED_BOOK_MAX = 0.56


def bar_for(p: dict) -> float:
    try:
        balanced = float(p.get("book_p")) < BALANCED_BOOK_MAX
    except (TypeError, ValueError):
        balanced = False
    return BALANCED_BAR if balanced else PUBLISH_BAR
TOP_PER_MARKET = 3
# ALTERNATE LINES: the book's lower rungs, where overs are likelier. A rung is a pick
# only at ALT_BAR or above AND with the forecast at least as sure as the book; the
# highest such rung (the best price) is the one taken, at most ALT_PER_GAME a game.
# PAUSED (four-reviewer audit, 2026-10-12): validated only against the weakest
# simulation, and a one-sided rung carries 8-12% margin, not the main line's ~5%.
# Back on once real alternate-line prices have been logged and graded.
ALT_ON = False
ALT_BAR = 0.70
ALT_PER_GAME = 2
DEFAULT_OVERROUND = 1.10            # a rung's own margin is wider than the main line's
HISTORY_SEASONS = 6               # five to train on, plus one for entering features


def weights() -> dict:
    """{market: (a, b, c)}; adopted real-line weights win over the gate's."""
    out = {}
    try:
        markets = json.loads(REPORT.read_text(encoding="utf-8"))["markets"]
        from tracking import combine_fit
        out = {m: combine_fit.prior(v["combined"]["weights_model_book_const"])
               for m, v in markets.items()}
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


def live_outs(ruled_out: dict, history: pd.DataFrame, projections: list) -> pd.DataFrame:
    """The live report's OUT players as out_reports rows for the upcoming game of
    their team. `ruled_out` is {name: team or None}; names join to player ids within
    the team where the report names one, and a name shared by two players is skipped."""
    from nfl import odds as odds_mod
    last = history.sort_values(["season", "week"]).groupby("player_id").tail(1)
    by_name = {}
    for r in last.itertuples():
        by_name.setdefault(odds_mod.norm_name(str(r.player_display_name)), []).append((r.player_id, r.team))
    games = {}
    for p in projections:
        try:
            gid = str(p["game_id"])
            games[p["team"]] = (int(gid.split("_")[0]), int(gid.split("_")[1]))
        except (KeyError, ValueError, IndexError):
            continue
    rows = []
    for name, team in (ruled_out or {}).items():
        cands = by_name.get(odds_mod.norm_name(str(name))) or []
        if team:
            cands = [c for c in cands if c[1] == team] or cands
        if len(cands) != 1:
            continue
        pid, last_team = cands[0]
        team = team or last_team
        if team in games:
            rows.append({"player_id": pid, "team": team, "season": games[team][0], "week": games[team][1]})
    return pd.DataFrame(rows, columns=["player_id", "team", "season", "week"])


def frame(projections: list, history: pd.DataFrame | None = None, ruled_out: dict | None = None) -> pd.DataFrame:
    """Built features: every player-week of history plus ONE upcoming row per
    (player, game) on the slate. `history` (fc.load output) may be passed in.
    `ruled_out` ({name: team}) are this week's OUT players: their teammates' rows
    carry what they leave behind (nfl/forecast._vacated)."""
    first = config.CURRENT_SEASON - HISTORY_SEASONS
    seasons = range(first, config.CURRENT_SEASON + 1)
    if history is None:
        history = fc.load(seasons)
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
    outs = pd.concat([fc.out_reports(seasons), live_outs(ruled_out, history, projections)],
                     ignore_index=True).drop_duplicates()
    return fc.build(rows, outs)


def _train(built: pd.DataFrame) -> pd.DataFrame:
    return built[~built["is_next"] & (built["season"] >= config.CURRENT_SEASON - fc.TRAIN_SEASONS)]


# TRACKING ONLY (audit, 2026-10-12): anytime TD has a = 0 -- its pick is the book's own
# favourite, and its gate overstated (said 85.5%, landed 80.9%). Priced and logged.
HELD = {"anytime_touchdown"}


def _clears(p: dict) -> bool:
    """The chance a pick must show. A VALUE pick is judged on the published (book-led)
    chance. A PROVING pick -- never a bet, frozen and graded so the market can earn
    trust -- is judged on the MODEL'S own chance (62%+); model_backs already holds it
    within 15 points of the book and 2+ above it. Judging proving picks on the
    shrunk chance made them impossible on any line the book sets near 50/50: the
    board published nothing at all (five-sport audit, 2026-10-18)."""
    if p.get("pick_kind") == "proving":
        return float(p.get("p_model") or 0) >= PUBLISH_BAR
    return p["probability"] >= bar_for(p)


def select(priced: list) -> list:
    # VALUE OR PROVING (tracking/pricecheck.qualifies, owner 2026-10-12): +3% at bet365,
    # or a real contender the model backs (as sure as the book, within 15 points of it).
    from tracking import pricecheck
    allp = priced
    # A game already under way is not a pick any more: it can never be frozen.
    now = pd.Timestamp.now(tz="UTC")

    def _started(p):
        try:
            k = pd.Timestamp(p.get("kickoff"))
            return (k.tz_localize("UTC") if k.tzinfo is None else k) <= now
        except Exception:                                   # noqa: BLE001
            return False
    priced = [p for p in priced if not _started(p)]
    priced = [p for p in priced if pricecheck.qualifies(p)]
    keep = sorted([p for p in priced if _clears(p) and p["p_model"] >= 0.5
                   and p["market"] not in HELD
                   and (p["market"] != "anytime_touchdown" or p.get("side") == "over")],
                  key=lambda p: -p["probability"])
    by_game = {}
    for p in keep:
        by_game.setdefault(str(p["game_id"]), []).append(p)
    out = sorted([p for ps in by_game.values() for p in ps[:TOP_PER_MARKET]],
                  key=lambda p: -p["probability"])
    pricecheck.untag_unpublished(allp, out)
    return out


def _alt_rung(p: dict, model, mu: float, w):
    """The highest rung BELOW the main line clearing ALT_BAR with the forecast at least
    as sure as the book, or None. The book's rung price is one-sided, so its chance
    is de-vigged by the main line's own overround."""
    ladder = p.get("ladder") or {}
    if not ladder.get("rungs") or p.get("line") is None:
        return None
    over_price, over_p = p.get("book_price"), p.get("book_p")
    overround = DEFAULT_OVERROUND
    if over_price and over_p:
        overround = max((1.0 / float(over_price)) / float(over_p), DEFAULT_OVERROUND)
    best = None
    for r in sorted(ladder["rungs"], key=lambda r: r["line"]):
        line, price = float(r["line"]), float(r.get("price") or 0)
        if line >= float(p["line"]) or price <= 1.0:
            continue
        pb = min((1.0 / price) / overround, 0.99)
        pm = float(model.spread.p_over(np.array([mu]), np.array([line]))[0])
        pc = combine(pm, pb, w)
        if pc >= ALT_BAR and pm >= pb:
            best = {"line": line, "price": price, "book": ladder.get("book"),
                    "probability": round(pc, 4), "p_model": round(pm, 4), "book_p": round(pb, 4)}
    return best


def select_alt(priced: list, taken: list) -> list:
    """Alternate-line picks for players with no main-line pick, best first."""
    have = {(str(p["game_id"]), p.get("player_id")) for p in taken}
    cands = []
    for p in priced:
        a = p.get("alt_candidate")
        if not a or (str(p["game_id"]), p.get("player_id")) in have:
            continue
        q = {k: v for k, v in p.items() if k != "alt_candidate"}
        q.update({"side": "over", "alt": True, "line": a["line"], "book_price": a["price"],
                  "book": a["book"], "line_source": "alternate", "probability": a["probability"],
                  "p_model": a["p_model"], "book_p": a["book_p"],
                  "edge": round(a["probability"] - a["book_p"], 4)})
        cands.append(q)
    by_game = {}
    for q in sorted(cands, key=lambda q: -q["probability"]):
        by_game.setdefault(str(q["game_id"]), []).append(q)
    return [q for qs in by_game.values() for q in qs[:ALT_PER_GAME]]


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
    model_ = model
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
        if ALT_ON and m is not None:
            q["alt_candidate"] = _alt_rung(p, model_, float(m), w)
        out.append(q)
    out.sort(key=lambda p: -p["probability"])
    return out
