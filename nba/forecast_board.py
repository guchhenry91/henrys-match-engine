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
from tracking import combine_fit, recent

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data-raw" / "nba" / "forecast_report.json"
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
TOP_PER_MARKET = 3              # per game, per market
# TEAMMATES OUT, trained on the official pre-game injury report (nba/injury_report).
# Gate 2026-10-12 (scored 2025+2026, trained 2022 on, with vs without):
#   PTS MAE 4.908 vs 4.939, ECE 0.022 vs 0.027; REB 1.976 vs 1.981, ECE 0.0061 vs 0.0073;
#   AST 1.463 vs 1.465, combined ECE 0.0087 vs 0.0113, @70 landed 0.743 vs 0.734;
#   FG3M 0.985 vs 0.984 -- no gain, so threes stay without it.
# Re-run after the report-name parser fix (third review): PTS 4.907, REB 1.976, AST 1.462,
# FG3M 0.984 (no gain) -- same decision. Both arms are kept for re-checking:
# data-raw/nba/forecast_report_reported.json and forecast_report_noreported.json.
VACATED_STATS = frozenset({"PTS", "REB", "AST"})
VACATED = bool(VACATED_STATS)
HISTORY_SEASONS = 6             # five to train on, plus one for entering features


PRICED_LOG = ROOT / "data-raw" / "nba" / "priced_lines.json"
WEIGHTS = ROOT / "data-raw" / "nba" / "combine_weights.json"


def weights() -> dict:
    """{stat: (a, b, c)}: weights re-fitted on REAL graded lines where adopted
    (tracking/combine_fit.py, data-raw/nba/combine_weights.json), otherwise the gate's
    from its simulated book CAPPED to trust the book (combine_fit.prior: book weight 1,
    forecast pull <= 0.2) until real lines earn more."""
    out = {}
    try:
        stats = json.loads(REPORT.read_text(encoding="utf-8"))["stats"]
        out = {s: combine_fit.prior(v["combined"]["weights_model_book_const"])
               for s, v in stats.items()}
    except Exception:
        pass
    for market, w in combine_fit.adopted(WEIGHTS).items():
        if market in fc.MARKET_STAT:
            out[fc.MARKET_STAT[market]] = w
    return out


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
    # VALUE OR PROVING (tracking/pricecheck.qualifies, owner 2026-10-12): +3% at bet365,
    # or a real contender the model backs (as sure as the book, within 15 points of it).
    from tracking import pricecheck
    priced = [p for p in priced if pricecheck.qualifies(p)]
    keep = sorted([p for p in priced if p["probability"] >= bar_for(p) and p["p_model"] >= 0.5],
                  key=lambda p: -p["probability"])
    by_game = {}
    for p in keep:
        by_game.setdefault(p["game_id"], []).append(p)
    return sorted([p for ps in by_game.values() for p in ps[:TOP_PER_MARKET]],
                  key=lambda p: -p["probability"])


def picks(games_ahead, odds_store, teams_now=None, ruled_out=(), collect=None,
          vacated_out=None) -> tuple:
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
    # TEAMMATES OUT: the OFFICIAL REPORT's OUT players (vacated_out), on the team they
    # are on now, are the absences the forecast redistributes usage for -- the same
    # source it was trained on. None = no report today: teammates-out is off this run
    # rather than reading "nobody is out" (third review, 2026-10-12).
    use_vacated = VACATED and vacated_out is not None
    out_players = {}
    for name in ({odds_mod.norm_name(n) for n in vacated_out} if use_vacated else ()):
        pid = names.get(name)
        team = (teams_now.get(pid) or last_team.get(pid)) if pid is not None else None
        if team:
            out_players.setdefault(team, set()).add(pid)
    # Played games' absences come from the official pre-game report, as in training
    # (nba/injury_report.py); without VACATED the feature is unused, so skip the work.
    reported = fc.report_outs(rows) if use_vacated else None
    frame = fc.build(pd.concat([rows, pd.DataFrame(synth)], ignore_index=True), out_players, reported)
    train = frame[~frame["is_next"] & (frame["season"] >= config.CURRENT_SEASON - fc.TRAIN_SEASONS)]
    ask = frame[frame["is_next"]]
    w = weights()
    real = rows.copy()
    real["game_date"] = pd.to_datetime(real["GAME_DATE"].astype(str).str[:10])
    for market, stat in fc.MARKET_STAT.items():
        # Teammates-out: trained and fed on the same official report (see VACATED_STATS).
        model = fc.StatModel(stat, vacated=use_vacated and stat in VACATED_STATS).fit(train)
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
    from tracking import pricecheck
    out = pricecheck.one_per_player(out)        # one pick per player per game
    return out, held
