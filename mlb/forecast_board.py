"""The MLB prop and team-total board on the REBUILT model (mlb/forecast.py).

Replaces the pricing in mlb/board.props_and_totals (who is quoted, who is available,
and the book's line still come from the same sources). For each quoted player:

  1. forecast the full distribution of his stat in this game -- his recent rates,
     role, the opposing PROBABLE starter, the park, the two teams' run form; the
     bookmaker's line is never an input;
  2. read the chance of beating the BOOK'S line off it;
  3. combine with the book's de-vigged chance, the book's weight held at 1
     (tracking/combine_fit.fit_offset):  final = sigmoid(a*logit(model) + logit(book) + c),
     a and c per market from the gate (data-raw/mlb/forecast_report.json), replaced by
     weights re-fitted on real graded lines once a market has 300
     (data-raw/mlb/combine_weights.json);
  4. publish a side at PUBLISH_BAR or above with the forecast on the same side, at most
     TOP_PER_MARKET a game; home runs on the "yes" side only.

RULE per market, from the gate (scripts/mlb_forecast_gate.py, 2021-25):
  * VETO (hits, RBI): the stated chance is the MODEL'S own -- calibrated at every line
    over five seasons -- and a pick needs the book on the same side. The simulated
    book could not stand in for a real one here (it does not know the opposing
    starter or the park), so no blend weight from it is trusted. Gate: hits said
    66.8% landed 66.3%, RBI 71.3/71.5.
  * COMBINED (strikeouts, team runs): the blend above. Gate: Ks said 67.8% landed
    67.5%; team runs 65.4/63.2.
  * HELD (H+R+RBI, home runs): priced and logged, never published. H+R+RBI landed 45%
    of 22 real lines stated at 66%; HR "yes" never reaches the bar.
Once a market has 300 graded real lines and re-fitted weights are ADOPTED
(tracking/combine_fit), it is priced on those weights with the COMBINED rule.
Picks carry engine=ENGINE.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mlb import current
from mlb import forecast as fc
from nfl import odds as odds_mod
from tracking import combine_fit, recent

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data-raw" / "mlb" / "forecast_report.json"
PRICED_LOG = ROOT / "data-raw" / "mlb" / "priced_lines.json"
WEIGHTS = ROOT / "data-raw" / "mlb" / "combine_weights.json"
ENGINE = "forecast-v1"
PUBLISH_BAR = 0.62
TOP_PER_MARKET = 3
HISTORY_SEASONS = 6
# RBI unders are the juiced side (about 1.36): the live record went 43-29 (60%) where the
# book said 68% (props-trader review, 2026-10-11), so only overs are published there.
OVER_ONLY = {"hr", "rbi"}
# Audit 2026-10-12: every market trusts the book by default (combine_fit.prior), and
# strikeouts are held (27 real lines: model said 54% over, 30% landed).
RULE = {"hits": "combined", "rbi": "combined", "strikeouts": "held", "team_runs": "combined",
        "hrr": "held", "hr": "held"}
OFFSET_MARKETS = {"strikeouts", "team_runs"}       # median-set lines: book weight held at 1


def weights() -> dict:
    out = {}
    try:
        markets = json.loads(REPORT.read_text(encoding="utf-8"))["markets"]
        out = {m: combine_fit.prior(v["combined"]["weights_model_book_const"])
               for m, v in markets.items()}
    except Exception:
        pass
    out.update(combine_fit.adopted(WEIGHTS))
    return out


def combine(p_model: float, p_book: float, w) -> float:
    a, b, c = w or (0.0, 1.0, 0.0)

    def lg(p):
        p = min(max(float(p), 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-(a * lg(p_model) + b * lg(p_book) + c)))


def select(priced: list) -> list:
    # VALUE OR PROVING (tracking/pricecheck.qualifies, owner 2026-10-12): +3% at bet365,
    # or a real contender the model backs (as sure as the book, within 15 points of it).
    from tracking import pricecheck
    priced = [p for p in priced if pricecheck.qualifies(p)]
    keep = sorted([p for p in priced if p["probability"] >= PUBLISH_BAR and p["p_model"] >= 0.5
                   and (p["market"] not in OVER_ONLY or p["side"] == "over")],
                  key=lambda p: -p["probability"])
    by_game = {}
    for p in keep:
        by_game.setdefault(str(p["game_id"]), []).append(p)
    return sorted([p for ps in by_game.values() for p in ps[:TOP_PER_MARKET]],
                  key=lambda p: -p["probability"])


def _probable(g, side):
    v = getattr(g, f"{side}_pitcher", None)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return current.retro_id(int(v))
    except (TypeError, ValueError):
        return None


def props(games_ahead: pd.DataFrame, odds_store: dict, avail=None, collect=None,
          unavailable=None) -> dict:
    """{market: [published picks]}; every priced pick goes to `collect` per game."""
    odds_games = odds_store.get("games") or {}
    frames = fc.load(range(current.SEASON - HISTORY_SEASONS, current.SEASON))   # + this season live
    live = {k: current.load(n) for k, n in (("batter", "batting"), ("pitcher", "pitching"))}
    names, team_of = {}, {}
    for k, f in live.items():
        if f.empty or "NAME" not in f:
            continue
        latest = f.sort_values("game_date").groupby("PLAYER_ID").tail(1)
        for r in latest.itertuples():
            names.setdefault((k, odds_mod.norm_name(r.NAME)), set()).add(r.PLAYER_ID)
            team_of[r.PLAYER_ID] = r.TEAM
    synth = {"batter": [], "pitcher": [], "team": []}
    meta, taken = {}, set()
    for g in games_ahead.sort_values("start").itertuples():
        og = odds_games.get(str(g.game_pk)) or {}
        gid = f"NEXT{g.game_pk}"
        base = {"GAME_ID": gid, "game_date": pd.Timestamp(g.game_date), "season": current.SEASON,
                "is_next": True}
        for side in ("home", "away"):
            q = (og.get("team_totals") or {}).get(side)
            team = getattr(g, f"{side}_team")
            opp = g.away_team if side == "home" else g.home_team
            opp_side = "away" if side == "home" else "home"
            row = {**base, "PLAYER_ID": team, "TEAM": team, "opponent": opp,
                   "is_home": float(side == "home"), "OPP_STARTER": _probable(g, opp_side)}
            synth["team"].append(row)
            if q and ("team_runs", team) not in taken:
                taken.add(("team_runs", team))
                meta[("team_runs", team, gid)] = (g, team, opp, side == "home", q, getattr(g, f"{side}_name", team))
        for market in ("hits", "hrr", "hr", "rbi", "strikeouts"):
            kind = fc.MARKETS[market][0]
            for name, q in ((og.get("props") or {}).get(market) or {}).items():
                ids = names.get((kind, odds_mod.norm_name(name))) or set()
                if len(ids) != 1:
                    continue
                pid = next(iter(ids))
                team = team_of.get(pid)
                if team not in (g.home_team, g.away_team) or (market, pid) in taken:
                    continue                   # unknown side, or his NEXT game only
                if unavailable is not None and unavailable(kind, pid, team, g, avail):
                    continue
                taken.add((market, pid))
                home = team == g.home_team
                opp = g.away_team if home else g.home_team
                row = {**base, "PLAYER_ID": pid, "TEAM": team, "opponent": opp, "is_home": float(home),
                       "OPP_STARTER": _probable(g, "away" if home else "home")}
                if not any(r["PLAYER_ID"] == pid and r["GAME_ID"] == gid for r in synth[kind]):
                    synth[kind].append(row)
                meta[(market, pid, gid)] = (g, team, opp, home, q, name)
    # Pitcher rows: a starter's own upcoming row (for K) -- his probable status is
    # checked by `unavailable`; the team frame needs the starters too.
    for kind, rows in synth.items():
        if rows:
            frames[kind] = pd.concat([frames[kind], pd.DataFrame(rows)], ignore_index=True)
    if not meta:
        return {m: [] for m in fc.MARKETS}
    built = fc.build(frames)
    w = weights()
    adopted = combine_fit.adopted(WEIGHTS)
    out = {}
    for market, (kind, stat, _) in fc.MARKETS.items():
        f = built[kind]
        keys = [(k[1], k[2]) for k in meta if k[0] == market]
        if not keys:
            out[market] = []
            continue
        train = f[~f["is_next"] & (f["season"] >= current.SEASON - fc.TRAIN_SEASONS)]
        model = fc.CountModel(market).fit(train)
        ask = f[f["is_next"]].drop_duplicates(["PLAYER_ID", "GAME_ID"]).set_index(["PLAYER_ID", "GAME_ID"])
        keys = [k for k in keys if k in ask.index]
        if not keys:
            out[market] = []
            continue
        rows = ask.loc[keys].reset_index()
        lines = np.array([float(meta[(market, k[0], k[1])][4]["line"]) for k in keys])
        pm_all = model.p_over(rows, lines)
        mu_all = model.mean(rows)
        last5 = recent.last_n(frames[kind][~frames[kind]["is_next"].fillna(False).astype(bool)],
                              rows["PLAYER_ID"], stat)
        priced, rule = [], RULE[market]          # adoption changes weights, never the rule
        for (pid, gid), pm, mu, line in zip(keys, pm_all, mu_all, lines):
            g, team, opp, home, q, name = meta[(market, pid, gid)]
            if q.get("over") is None:
                continue
            pb = float(q["over"])
            rule = RULE[market]
            pc = float(pm) if rule in ("veto", "held") else combine(float(pm), pb, w.get(market))
            side = "over" if pc >= 0.5 else "under"
            if side == "under" and not q.get("odd_under"):
                continue
            pick = {"market": market, "player": name, "player_id": pid, "team": team,
                    "opponent": opp, "home": bool(home), "game_id": str(g.game_pk),
                    "tipoff": g.start, "kickoff": g.start, "line": float(line),
                    "side": side, "probability": round(pc if side == "over" else 1 - pc, 4),
                    "p_model": round(float(pm) if side == "over" else 1 - float(pm), 4),
                    "book": q.get("book"), "line_source": q.get("source"),
                    "book_price": q.get("odd_over") if side == "over" else q.get("odd_under"),
                    "book_p": round(pb if side == "over" else float(q.get("under", 1 - pb)), 4),
                    "book_p_fair": q.get("fair", True), "mu": round(float(mu), 2),
                    "last_five": last5.get(pid, []), "engine": ENGINE, "rule": rule,
                    # the book's chance of the OVER, kept so the priced-line log can learn
                    "book_p_over": round(pb, 4), "p_model_over": round(float(pm), 4)}
            pick["edge"] = round(pick["probability"] - pick["book_p"], 4)
            priced.append(pick)
            if collect is not None:
                collect.setdefault(str(g.game_pk), {}).setdefault(market, []).append(pick)
        if rule == "held":
            out[market] = []
        else:
            if rule == "veto":
                # the book must be on the model's side
                priced_ok = [p for p in priced
                             if (p["book_p_over"] >= 0.5) == (p["side"] == "over")]
            else:
                priced_ok = priced
            out[market] = select(priced_ok)
        for p in out[market]:
            p["on_board"] = True
        print(f"  {market} [{ENGINE}]: {len(priced)} priced, {len(out[market])} published "
              f"at {PUBLISH_BAR:.0%}+ and worth the price", flush=True)
    from tracking import pricecheck
    return pricecheck.one_per_player(out)        # one pick per player per game


def learn(payload: dict, now) -> dict:
    """Log every priced line, grade finished ones from the box scores, re-fit at 300."""
    priced = [p for markets in (payload.get("game_props") or {}).values()
              for ps in (markets or {}).values() for p in ps or [] if p.get("engine")]
    combine_fit.record(PRICED_LOG, priced, now)
    rows = {k: current.load(n) for k, n in (("batter", "batting"), ("pitcher", "pitching"), ("team", "team"))}
    played = set(rows["team"]["GAME_ID"]) if not rows["team"].empty else set()
    idx = {k: {(str(r.GAME_ID), str(r.PLAYER_ID)): r for r in f.itertuples()} if not f.empty else {}
           for k, f in rows.items()}

    def _actual(e):
        gid = f"MLB{e['game_id']}"
        if gid not in played:
            return None
        kind, stat, _ = fc.MARKETS[e["market"]]
        r = idx[kind].get((gid, str(e["player_id"])))
        if r is None:
            return "void"                       # did not start / did not pitch
        return float(getattr(r, stat))
    combine_fit.grade(PRICED_LOG, _actual)
    return combine_fit.refit(PRICED_LOG, WEIGHTS, weights(), now, offset=OFFSET_MARKETS)

