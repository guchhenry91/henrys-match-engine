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
# GATED 2026-10-20 (live blend vs an informed book, 2021-26): said/landed PTS 58.8/62.3,
# REB 58.6/66.0, AST 57.9/65.7, FG3M 57.6/63.9 -- it understates, never overstates.
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


# PROVING PICKS ON THE PUBLISHED CHANCE (review, 2026-10-20). They were judged on the
# model's own chance at 62%+, and against an informed book that stated 67-70% and
# landed 57-65% (scripts/nba_forecast_gate.py "live"). On the published (book-led)
# chance at 55%+, with the model 2+ points above the book, they landed at or above
# what they state in rebounds, assists and threes, and within 3 points in points.
PROVING_BAR = 0.55
# POINTS AT 54% (owner, 2026-10-20). Points lines are set near 50/50, so at 55% the
# board showed one points pick on opening night. Gate bands on the live blend
# (forecast_report.json PTS live.proving_on_blend): picks at 54-55% landed 54.4%, above
# bet365's ~53.5% break-even; picks at 52-54% landed 51.0%, so the bar stops at 54%.
PROVING_BAR_BY_MARKET = {"points": 0.54}


def _clears(p: dict) -> bool:
    """The chance a pick must show -- always the PUBLISHED (book-led) chance. VALUE:
    bar_for (62%, or 55% on a near-even line -- both gated). PROVING: PROVING_BAR,
    with model_backs already holding the model 2+ points above the book."""
    if p.get("pick_kind") == "proving":
        return p["probability"] >= PROVING_BAR_BY_MARKET.get(p.get("market"), PROVING_BAR)
    return p["probability"] >= bar_for(p)


# EARLY-SEASON MINUTES DIP (opening-night audit, 2026-10-19). A player whose minutes
# fell late last season (injury return, rest, tanking) is forecast on that dip, as if
# the summer never happened. Walk-forward over four openers: those players were
# UNDER-forecast by 1.7 points, 0.5 rebounds, 0.4 assists (every other opener came out
# close or slightly high). The forecast now resets a season's first game to the longer
# run (nba/forecast.SUMMER_RESET): the gap fell to -0.67 points, and games 2-5 were
# already unbiased. What remains: no UNDER on such a player in his first game.
DIP_MIN = 5.0
DIP_GAMES = 1


def _dip_under(p: dict) -> bool:
    return (p.get("side") == "under" and (p.get("min_dip") or 0) >= DIP_MIN
            and (p.get("season_games_before") or 0) < DIP_GAMES)


def select(priced: list) -> list:
    """The published shortlist: combined chance at PUBLISH_BAR or more, AND the
    forecast itself on the same side (a pick the model argues against is the book's
    pick, not ours -- the gate showed requiring agreement costs ~2% of picks and no
    accuracy), at most TOP_PER_MARKET per game, best first."""
    # VALUE OR PROVING (tracking/pricecheck.qualifies, owner 2026-10-12): +3% at bet365,
    # or a real contender the model backs (as sure as the book, within 15 points of it).
    from tracking import pricecheck
    allp = priced
    priced = [p for p in priced if pricecheck.qualifies(p)]
    for p in priced:
        if _dip_under(p):
            p["dropped"] = "minutes dipped late last season; the forecast underrates him early on"
    keep = sorted([p for p in priced if _clears(p) and p["p_model"] >= 0.5 and not _dip_under(p)],
                  key=lambda p: -p["probability"])
    by_game = {}
    for p in keep:
        by_game.setdefault(p["game_id"], []).append(p)
    out = sorted([p for ps in by_game.values() for p in ps[:TOP_PER_MARKET]],
                  key=lambda p: -p["probability"])
    pricecheck.untag_unpublished(allp, out)
    return out


# CUSTOM PLAYS and MONSTER NIGHTS (owner, 2026-10-20). Walk-forward over 2024-25 and
# 2025-26, overs only, a team's focal players: milestone ("k+") chances were the
# model's best-calibrated product -- at 70%+ said/landed PTS 20+ 78/80, REB 8+ 80/82,
# AST 6+ 81/81, FG3M 3+ 75/74 -- and its top monster-night pick each night hit 3-5x
# the base rate (PTS 30+: 21% against 5.5%). Priced on the books' ALTERNATE ladders.
FOCAL_TOP = 3              # a team's top three by expected usage this game...
FOCAL_MIN = 26.0           # ...who play real minutes
CUSTOM_BAR = 0.70          # the model's own chance a milestone needs
CUSTOM_PER_GAME = 4
CUSTOM_MIN_PRICE = 1.40    # a milestone worth betting: "2+ assists at 1.10" lands and pays nothing
ALT_OVERROUND = 1.06       # a one-sided ladder rung's margin, removed for its fair chance
MONSTER = {"points": (30, 1.5), "rebounds": (12, 1.5), "assists": (10, 1.5), "threes": (5, 2.0)}
MONSTER_MIN_P = 0.12
MONSTER_PER_NIGHT = 3


# PLAYOFFS (walk-forward over five playoffs 2022-26, 8,923 player-games). The forecast
# carried regular-season usage into the playoffs: a team's focal players kept their
# points (1.00) and out-rebounded it (1.08) on bigger minutes, while everyone else fell
# well short (points 0.87, assists 0.81) as rotations shorten. Scaled by group; each
# season's factors fitted on the other four left its bias within about 5%.
PLAYOFF_ADJ = {True: {"PTS": 1.000, "REB": 1.081, "AST": 0.942, "FG3M": 0.949},
               False: {"PTS": 0.874, "REB": 0.947, "AST": 0.807, "FG3M": 0.862}}


def playoff_factor(stage: str, focal: bool, stat: str) -> float:
    return PLAYOFF_ADJ[bool(focal)][stat] if stage == current.PLAYOFFS else 1.0


def _focal(ask: pd.DataFrame) -> set:
    """(PLAYER_ID, GAME_ID) of each team's focal players in its next game."""
    a = ask.assign(_use=ask["usage_pm10"] * ask["min_ewm10"])
    a["_rank"] = a.groupby(["GAME_ID", "TEAM_ABBREVIATION"])["_use"].rank(ascending=False, method="first")
    a = a[(a["_rank"] <= FOCAL_TOP) & (a["min_ewm10"] >= FOCAL_MIN)]
    return {(int(r.PLAYER_ID), r.GAME_ID) for r in a.itertuples()}


def _rungs(market, r, m_, a_, info, alt, w, stat):
    """Every ladder rung for one focal player, priced: model chance, the book's fair
    chance, the combined chance, and the price check."""
    from tracking import pricecheck
    g = info["game"]
    avg = float(r[f"{stat}_pm10"] * r["min_ewm10"]) if pd.notna(r[f"{stat}_pm10"]) else None
    out = []
    for line, price in alt.get("ladder") or []:
        p_model = float(dist.p_over(m_, a_, float(line)))
        book_p = min(0.99, (1.0 / float(price)) / ALT_OVERROUND)
        prob = combine(p_model, book_p, w.get(stat))
        c = {"market": market, "player": info["name"], "player_id": int(r["PLAYER_ID"]),
             "team": info["team"], "opponent": info["opponent"], "home": info["home"],
             "game_id": g.game_id, "tipoff": g.tipoff, "kickoff": g.tipoff,
             "line": float(line), "milestone": int(float(line) + 0.5), "side": "over",
             "book": alt.get("book"), "line_source": alt.get("source"), "fair": False,
             "book_price": float(price), "book_p": round(book_p, 4),
             "probability": round(prob, 4), "p_model": round(p_model, 4),
             "mu": round(float(m_), 2), "avg": None if avg is None else round(avg, 1),
             "engine": ENGINE}
        pricecheck.qualifies(c)
        out.append(c)
    return out


def custom_plays(cands: list) -> dict:
    """{"custom": [...], "monster": [...]} from every focal player's priced rungs.
    CUSTOM: the model at CUSTOM_BAR+, a price of CUSTOM_MIN_PRICE+, and the price check
    passed (value or proving);
    per player and market the HIGHEST such milestone (the best price); CUSTOM_PER_GAME
    a game. MONSTER: a rung at the monster threshold, the model at MONSTER_MIN_P+, and
    VALUE at bet365's estimated price -- a long shot is a pick only at a price that
    pays for its misses; MONSTER_PER_NIGHT, best expected return first."""
    best = {}
    for c in cands:
        if (c["p_model"] >= CUSTOM_BAR and c.get("pick_kind") in ("value", "proving")
                and float(c.get("book_price") or 0) >= CUSTOM_MIN_PRICE):
            k = (c["game_id"], c["player_id"], c["market"])
            if k not in best or c["line"] > best[k]["line"]:
                best[k] = c
    by_game = {}
    for c in sorted(best.values(), key=lambda c: -c["p_model"]):
        by_game.setdefault(c["game_id"], []).append(c)
    custom = [dict(c, play="custom") for cs in by_game.values() for c in cs[:CUSTOM_PER_GAME]]
    mons = []
    for c in cands:
        floor, mult = MONSTER[c["market"]]
        thr = max(floor, mult * (c.get("avg") or 0))
        if (c["milestone"] >= thr and c["p_model"] >= MONSTER_MIN_P
                and c.get("pick_kind") == "value"):
            mons.append(dict(c, play="monster"))
    nights, keep, seen = {}, [], set()
    for c in sorted(mons, key=lambda c: -((c.get("price_check") or {}).get("ev") or 0)):
        night = str(c["tipoff"])[:10]
        if (c["game_id"], c["player_id"], c["market"]) in seen or nights.get(night, 0) >= MONSTER_PER_NIGHT:
            continue
        seen.add((c["game_id"], c["player_id"], c["market"]))
        nights[night] = nights.get(night, 0) + 1
        keep.append(c)
    return {"custom": sorted(custom, key=lambda c: -c["probability"]), "monster": keep}


def picks(games_ahead, odds_store, teams_now=None, ruled_out=(), collect=None,
          vacated_out=None, plays=None) -> tuple:
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
    focal = _focal(ask)
    rungs = []
    for market, stat in fc.MARKET_STAT.items():
        # Teammates-out: trained and fed on the same official report (see VACATED_STATS).
        model = fc.StatModel(stat, vacated=use_vacated and stat in VACATED_STATS).fit(train)
        mu = model.mean(ask)
        # Playoff games: the measured playoff correction, by focal player or not.
        mu = mu * np.array([playoff_factor(
            str(getattr((meta.get((int(pid), gid)) or {}).get("game"), "stage", "")),
            (int(pid), gid) in focal, stat) for pid, gid in zip(ask["PLAYER_ID"], ask["GAME_ID"])])
        alpha = model.disp(mu)
        last5 = recent.last_n(real, ask["PLAYER_ID"], stat, played="MIN")
        priced, unpriced = [], 0
        for (_, r), m_, a_ in zip(ask.iterrows(), mu, alpha):
            info = meta.get((int(r["PLAYER_ID"]), r["GAME_ID"]))
            g = info["game"]
            if plays is not None and (int(r["PLAYER_ID"]), r["GAME_ID"]) in focal:
                alt = odds_mod.match_player(
                    ((odds_games.get(g.game_id) or {}).get("alt") or {}).get(market) or {},
                    info["name"])
                if alt:
                    rungs += _rungs(market, r, m_, a_, info, alt, w, stat)
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
                    "season_games_before": int(r["season_games_before"]),
                    "min_dip": round(float(r["min_dip"]), 1) if pd.notna(r.get("min_dip")) else None,
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
    if plays is not None:
        plays.update(custom_plays(rungs))
        print(f"  custom plays: {len(rungs)} focal rungs priced, {len(plays['custom'])} custom, "
              f"{len(plays['monster'])} monster", flush=True)
    return out, held
