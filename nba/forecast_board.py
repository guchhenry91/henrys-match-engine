"""The NBA prop board on the REBUILT model (docs/REBUILD_PLAN.md).

Replaces nba/board.prop_picks. For every player a bookmaker quotes on the slate:

  1. forecast his stat's full distribution (nba/forecast.py) from his and his
     team's history -- the bookmaker's line is never an input;
  2. read the chance of beating the BOOK'S line off that distribution;
  3. combine it with the book's own de-vigged chance:
         final = sigmoid(a * logit(model) + b * logit(book) + c)
     with a, b, c per stat from the gate (data-raw/nba/forecast_report.json; to be
     re-fitted on the real-line archive as it fills);
  4. publish the side the combined chance favours, only at PUBLISH_BAR or above.

The gate (scripts/nba_forecast_gate.py) showed the picks this rule publishes land at
the rate stated (2021-2026 walk-forward against an informed book); the model alone,
the way the old board picked, overstated by 10-15 points.

Each pick carries engine=ENGINE, so its record is kept apart from the old model's
and none of the old layers (trust re-blend, shrink, the old record's gate) touch it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import dist
from nba import config, current
from nba import forecast as fc
from nfl import odds as odds_mod
from tracking import recent

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data-raw" / "nba" / "forecast_report.json"
ENGINE = "forecast-v1"
PUBLISH_BAR = 0.62
TOP_PER_MARKET = 3              # per game, per market
HISTORY_SEASONS = 6             # five to train on, plus one for entering features


def weights() -> dict:
    """{stat: (a, b, c)} from the gate; missing -> lean on the book (0.3, 0.7, 0)."""
    try:
        stats = json.loads(REPORT.read_text(encoding="utf-8"))["stats"]
        return {s: tuple(v["combined"]["weights_model_book_const"]) for s, v in stats.items()}
    except Exception:
        return {}


def combine(p_model: float, p_book: float, w) -> float:
    a, b, c = w or (0.3, 0.7, 0.0)

    def lg(p):
        p = min(max(float(p), 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-(a * lg(p_model) + b * lg(p_book) + c)))


def history_rows() -> pd.DataFrame:
    first = config.CURRENT_SEASON - HISTORY_SEASONS
    rows = fc.load_history(seasons=set(range(first, config.CURRENT_SEASON)))
    live = current.player_games()            # this season's REGULAR-season box scores
    if not live.empty:
        live = live.copy()
        live["season"] = config.CURRENT_SEASON
        rows = pd.concat([rows, live], ignore_index=True)
    rows["GAME_ID"] = rows["GAME_ID"].astype(str)
    return rows


def name_index(rows: pd.DataFrame) -> dict:
    """norm name -> PLAYER_ID from each player's newest row; shared names dropped."""
    latest = rows.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
    seen, dupes = {}, set()
    for r in latest.itertuples():
        key = odds_mod.norm_name(r.PLAYER_NAME)
        if key in seen and seen[key] != int(r.PLAYER_ID):
            dupes.add(key)
        seen[key] = int(r.PLAYER_ID)
    return {k: v for k, v in seen.items() if k not in dupes}


def upcoming_rows(games_ahead, odds_games, names, teams_now, last_team, ruled_out):
    """One placeholder row per quoted player, for his NEXT game only (a second
    upcoming game would see the first's empty row as his previous game)."""
    synth, meta, taken = [], {}, set()
    for g in games_ahead.sort_values("tipoff").itertuples():
        quotes = (odds_games.get(g.game_id) or {}).get("props") or {}
        for name in sorted({n for q in quotes.values() for n in q}):
            pid = names.get(odds_mod.norm_name(name))
            if pid is None or pid in taken or odds_mod.norm_name(name) in ruled_out:
                continue
            team = teams_now.get(pid) or last_team.get(pid)
            if team not in (g.home_team, g.away_team):
                continue                 # we cannot say which side he is on
            taken.add(pid)
            home = team == g.home_team
            opp = g.away_team if home else g.home_team
            gid = f"NEXT{g.game_id}"
            synth.append({"PLAYER_ID": pid, "PLAYER_NAME": name, "TEAM_ABBREVIATION": team,
                          "GAME_ID": gid, "GAME_DATE": str(g.game_date)[:10],
                          "MATCHUP": f"{team} vs. {opp}" if home else f"{team} @ {opp}",
                          "season": config.CURRENT_SEASON, "is_next": True, "MIN": np.nan})
            meta[(pid, gid)] = {"game": g, "team": team, "opponent": opp, "home": home,
                                "name": name, "club_source": "this season" if pid in teams_now
                                else "last season"}
    return synth, meta


def select(priced: list) -> list:
    """The published shortlist: combined chance at PUBLISH_BAR or more, AND the
    forecast itself on the same side (a pick the model argues against is the book's
    pick, not ours -- the gate showed requiring agreement costs ~2% of picks and no
    accuracy), at most TOP_PER_MARKET per game, best first."""
    keep = sorted([p for p in priced if p["probability"] >= PUBLISH_BAR and p["p_model"] >= 0.5],
                  key=lambda p: -p["probability"])
    by_game = {}
    for p in keep:
        by_game.setdefault(p["game_id"], []).append(p)
    return sorted([p for ps in by_game.values() for p in ps[:TOP_PER_MARKET]],
                  key=lambda p: -p["probability"])


def picks(games_ahead, odds_store, teams_now=None, ruled_out=(), collect=None) -> tuple:
    """({market: [picks]}, {market: unpriced count}) on the rebuilt model."""
    odds_games = odds_store.get("games") or {}
    rows = history_rows()
    names = name_index(rows)
    if teams_now is None:
        teams_now = current.current_teams(names)
    last_team = (rows.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ABBREVIATION"]
                 .last().to_dict())
    ruled = {odds_mod.norm_name(n) for n in ruled_out or ()}
    synth, meta = upcoming_rows(games_ahead, odds_games, names, teams_now, last_team, ruled)
    out, held = {m: [] for m in config.MARKETS}, {}
    if not synth:
        return out, held
    frame = fc.build(pd.concat([rows, pd.DataFrame(synth)], ignore_index=True))
    train = frame[~frame["is_next"] & (frame["season"] >= config.CURRENT_SEASON - fc.TRAIN_SEASONS)]
    ask = frame[frame["is_next"]]
    w = weights()
    real = rows.copy()
    real["game_date"] = pd.to_datetime(real["GAME_DATE"].astype(str).str[:10])
    for market, stat in fc.MARKET_STAT.items():
        model = fc.StatModel(stat).fit(train)
        mu = model.mean(ask)
        alpha = model.disp(mu)
        last5 = recent.last_n(real, ask["PLAYER_ID"], stat, played="MIN")
        priced, unpriced = [], 0
        for (_, r), m_, a_ in zip(ask.iterrows(), mu, alpha):
            info = meta.get((int(r["PLAYER_ID"]), r["GAME_ID"]))
            g = info["game"]
            quote = odds_mod.match_player(
                ((odds_games.get(g.game_id) or {}).get("props") or {}).get(market) or {},
                info["name"])
            if not quote or quote.get("over") is None:
                unpriced += 1
                continue
            line = float(quote["line"])
            p_model = float(dist.p_over(m_, a_, line))
            p_over = combine(p_model, float(quote["over"]), w.get(stat))
            side = "over" if p_over >= 0.5 else "under"
            prob = p_over if side == "over" else 1 - p_over
            book_p = float(quote["over"]) if side == "over" else float(quote["under"])
            pick = {"market": market, "player": info["name"], "player_id": int(r["PLAYER_ID"]),
                    "team": info["team"], "opponent": info["opponent"], "home": info["home"],
                    "game_id": g.game_id, "tipoff": g.tipoff, "kickoff": g.tipoff,
                    "line": line, "line_source": quote.get("source"), "book": quote.get("book"),
                    "side": side, "probability": round(prob, 4),
                    "p_model": round(p_model if side == "over" else 1 - p_model, 4),
                    "book_p": round(book_p, 4),
                    "book_price": quote.get("odd_over") if side == "over" else quote.get("odd_under"),
                    "edge": round(prob - book_p, 4), "mu": round(float(m_), 2),
                    "last_five": last5.get(r["PLAYER_ID"], []),
                    "games_before": int(r["games_before"]), "club_source": info["club_source"],
                    "engine": ENGINE}
            priced.append(pick)
            if collect is not None:
                collect.setdefault(str(g.game_id), {}).setdefault(market, []).append(pick)
        out[market] = select(priced)
        for p in out[market]:
            p["on_board"] = True
        held[market] = 0            # (players simply not quoted in this market are not "held")
        print(f"  {market} [{ENGINE}]: {len(priced)} priced, {len(out[market])} published "
              f"at {PUBLISH_BAR:.0%}+", flush=True)
    return out, held
