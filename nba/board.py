"""The live NBA slate: team-winner picks from Elo, player props at BOOKMAKER lines.

Called by nba.publish. Everything here reads data already on disk or passed in:
the committed history (nba.data), this season's CDN box scores (nba.current),
the Odds API lines and match odds (data-raw/nba/odds_api.json, written by
scripts/sync_nba_odds.py), and the Elo settings (data-raw/nba/elo_params.json).

PROPS STAND ON A BOOKMAKER'S LINE, NEVER THE PLAYER'S OWN. A player is priced
only if a book quotes him, and the model is asked about THAT line -- it is
trained across a spread of lines for exactly this (features.augment_lines). A
line below the lowest the model was trained on is dropped, not extrapolated.

WHO IS ON WHICH TEAM. Summer trades make last season's team a guess, so a
player's team is his newest box score this season (preseason included). The
book's quote already places him in the game; if his known team is neither side
of it, he is skipped rather than guessed onto one.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nba import config, current, data, features
from nfl import odds as odds_mod
from nfl.games_model import run_elo
from nfl.model import PropModel
from oddsapi import props as shared

ROOT = Path(__file__).resolve().parent.parent
ODDS = ROOT / "data-raw" / "nba" / "odds_api.json"
ELO = ROOT / "data-raw" / "nba" / "elo_params.json"

BOARD_HOURS = 36.0           # games listed this far ahead
LOOKAHEAD_DAYS = 7           # before the season: show the next slate within a week
TOP_PER_MARKET = 3           # per game, per market -- a shortlist, not a database
MIN_PROBABILITY = 0.50       # never publish a leg the model itself makes a dog
HISTORY_SEASONS = config.TRAIN_SEASONS + 1


def _read(path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def slate(sched: pd.DataFrame, now) -> pd.DataFrame:
    """Regular-season games still to tip, inside the board window -- or, before
    the season starts, the first game date within LOOKAHEAD_DAYS."""
    ahead = sched[(sched["stage"] == current.REGULAR) & (sched["status"] == 1)].copy()
    ahead["t"] = pd.to_datetime(ahead["tipoff"], utc=True)
    now = pd.Timestamp(shared.utc(now))
    ahead = ahead[ahead["t"] > now]
    window = ahead[ahead["t"] <= now + pd.Timedelta(hours=BOARD_HOURS)]
    if window.empty:
        soon = ahead[ahead["t"] <= now + pd.Timedelta(days=LOOKAHEAD_DAYS)]
        if not soon.empty:
            first = soon["game_date"].min()
            window = soon[soon["game_date"] == first]
    return window.sort_values("t").drop(columns=["t"])


def current_games(sched: pd.DataFrame) -> pd.DataFrame:
    """This season's regular-season games as rows nfl.games_model understands."""
    reg = sched[sched["stage"] == current.REGULAR].copy()
    out = pd.DataFrame({
        "game_id": reg["game_id"], "season": config.CURRENT_SEASON,
        "game_date": pd.to_datetime(reg["game_date"]),
        "home_team": reg["home_team"], "away_team": reg["away_team"],
        "home_score": pd.to_numeric(reg["home_score"], errors="coerce"),
        "away_score": pd.to_numeric(reg["away_score"], errors="coerce"),
        "neutral": reg["neutral"].astype(bool),
        "played": reg["status"] == current.FINAL,
    })
    out["winner"] = None
    out.loc[out["played"] & (out["home_score"] > out["away_score"]), "winner"] = "home"
    out.loc[out["played"] & (out["home_score"] < out["away_score"]), "winner"] = "away"
    return out


def team_picks(history_games, sched, games_ahead, odds_store) -> list:
    params = _read(ELO) or {"k": 20.0, "home_edge": 55.0, "regression": 0.33}
    this = current_games(sched)
    frame = pd.concat([history_games, this], ignore_index=True)
    walked = run_elo(frame, params["k"], params["home_edge"], params["regression"])
    # run_elo records each game's PRE-game probability and never updates on an
    # unplayed one, so an upcoming game's row is priced on today's ratings.
    walked = walked[walked["season"] == config.CURRENT_SEASON]
    by_id = {r.game_id: r for r in walked.itertuples()}
    out = []
    for g in games_ahead.itertuples():
        row = by_id.get(g.game_id)
        if row is None:
            continue
        p_home = float(row.prob_home)
        pick_home = p_home >= 0.5
        entry = {"game_id": g.game_id, "tipoff": g.tipoff, "date": g.tipoff,
                 "kickoff": g.tipoff, "neutral": bool(g.neutral),
                 "home": g.home_team, "away": g.away_team,
                 "pick": g.home_team if pick_home else g.away_team,
                 "p_home": round(p_home, 4),
                 "p_pick": round(p_home if pick_home else 1 - p_home, 4),
                 "rating_home": round(float(row.rating_home), 1),
                 "rating_away": round(float(row.rating_away), 1)}
        line = ((odds_store.get("games") or {}).get(g.game_id) or {}).get("moneyline")
        if line:
            book_p = line["home"] if pick_home else line["away"]
            entry.update({"book": line["book"], "book_p_pick": book_p,
                          "edge": round(entry["p_pick"] - book_p, 4)})
        out.append(entry)
    return out


def _name_index(rows: pd.DataFrame) -> dict:
    """norm name -> PLAYER_ID, newest season first; a name two players share is
    dropped rather than resolved by guessing."""
    latest = rows.sort_values("game_date").groupby("PLAYER_ID").tail(1)
    seen, dupes = {}, set()
    for r in latest.itertuples():
        key = odds_mod.norm_name(r.PLAYER_NAME)
        if key in seen and seen[key] != int(r.PLAYER_ID):
            dupes.add(key)
        seen[key] = int(r.PLAYER_ID)
    return {k: v for k, v in seen.items() if k not in dupes}


def player_rows(history_rows: pd.DataFrame) -> pd.DataFrame:
    """History plus this season's box scores, normalised the same way."""
    live = current.player_games()
    frames = [history_rows]
    if not live.empty:
        frames.append(data._normalise(live, config.CURRENT_SEASON))
    rows = pd.concat(frames, ignore_index=True)
    rows["GAME_ID"] = pd.to_numeric(rows["GAME_ID"], errors="coerce")
    return rows.dropna(subset=["GAME_ID", "PLAYER_ID", "game_date"])


def prop_picks(rows, games_ahead, odds_store, teams_now, released) -> tuple:
    """({market: [picks]}, {market: held_back_count}) at bookmaker lines only."""
    names = _name_index(rows)
    last_team = (rows.sort_values("game_date").groupby("PLAYER_ID")["TEAM_ABBREVIATION"]
                 .last().to_dict())
    odds_games = odds_store.get("games") or {}
    # One synthetic row per quoted player per upcoming game: dummy stats, since
    # every feature is ENTERING (shift(1)) and never reads its own row's values.
    synth, meta = [], {}
    for g in games_ahead.itertuples():
        quotes = (odds_games.get(g.game_id) or {}).get("props") or {}
        players = {name for q in quotes.values() for name in q}
        for name in players:
            pid = names.get(odds_mod.norm_name(name))
            if pid is None:
                continue
            team = teams_now.get(pid) or last_team.get(pid)
            if team not in (g.home_team, g.away_team):
                continue            # the book puts him in this game; we can't say which side
            is_home = team == g.home_team
            opp = g.away_team if is_home else g.home_team
            fake_gid = -int(str(g.game_id)[-7:])      # never collides with a real id
            synth.append({"PLAYER_ID": pid, "PLAYER_NAME": name, "TEAM_ABBREVIATION": team,
                          "GAME_ID": fake_gid, "season": config.CURRENT_SEASON,
                          "game_date": pd.Timestamp(g.game_date), "is_home": float(is_home),
                          "opponent": opp, "MIN": 30.0, "PTS": 0.0, "REB": 0.0,
                          "AST": 0.0, "FG3M": 0.0, "won": 0.0})
            meta[(pid, fake_gid)] = {"game": g, "team": team, "opponent": opp,
                                     "home": is_home, "name": name,
                                     "club_source": "this season" if pid in teams_now
                                     else "last season"}
    if not synth:
        return {m: [] for m in config.MARKETS}, {}
    allrows = pd.concat([rows, pd.DataFrame(synth)], ignore_index=True)
    out, held = {}, {}
    for market in config.MARKETS:
        if market not in released:
            out[market] = []
            continue
        built = features.build(allrows, market)
        is_synth = built["GAME_ID"] < 0
        train = built[~is_synth & (built["season"] >= config.CURRENT_SEASON - config.TRAIN_SEASONS)]
        model = PropModel(market).fit(features.augment_lines(train, market))
        ask = built[is_synth].copy()
        picks, below = [], 0
        for idx, r in ask.iterrows():
            m = meta.get((int(r["PLAYER_ID"]), int(r["GAME_ID"])))
            quote = odds_mod.match_player(
                ((odds_games.get(m["game"].game_id) or {}).get("props") or {}).get(market)
                or {}, m["name"])
            if quote is None:
                continue
            if float(quote["line"]) < config.MIN_LINE[market]:
                below += 1
                continue
            row = features.at_line(ask.loc[[idx]], float(quote["line"]))
            prob = float(model.predict(row)[0])
            picks.append({
                "market": market, "player": m["name"], "player_id": int(r["PLAYER_ID"]),
                "team": m["team"], "opponent": m["opponent"], "home": m["home"],
                "game_id": m["game"].game_id, "tipoff": m["game"].tipoff,
                "kickoff": m["game"].tipoff,
                "line": float(quote["line"]), "line_source": quote.get("source"),
                "book": quote.get("book"), "probability": round(prob, 4),
                "book_price": quote.get("odd_over"), "book_p": quote.get("over"),
                "edge": round(prob - float(quote["over"]), 4) if quote.get("over") else None,
                "form5": round(float(r["form5"]), 1), "min5": round(float(r["min5"]), 1),
                "games_before": int(r["games_before"]), "club_source": m["club_source"],
            })
        held[market] = below
        shortlist = [p for p in picks if p["probability"] >= MIN_PROBABILITY]
        by_game = {}
        for p in sorted(shortlist, key=lambda p: -p["probability"]):
            by_game.setdefault(p["game_id"], []).append(p)
        trimmed = [p for ps in by_game.values() for p in ps[:TOP_PER_MARKET]]
        out[market] = sorted(trimmed, key=lambda p: -p["probability"])
        print(f"  {market}: {len(picks)} priced at a book line, {len(out[market])} published"
              f"{f', {below} below the trained floor' if below else ''}")
    return out, held


def build(sched: pd.DataFrame, now, released: list) -> dict:
    """{"games", "props", "held_back", "odds_checked_at"} for nba.publish."""
    games_ahead = slate(sched, now)
    odds_store = _read(ODDS)
    history_games = data.games()
    teams = team_picks(history_games, sched, games_ahead, odds_store) if len(games_ahead) else []
    hist_rows = data.player_games(
        seasons=config.SEASONS[-HISTORY_SEASONS:]) if len(games_ahead) else pd.DataFrame()
    if len(games_ahead) and (odds_store.get("games") or {}):
        props, held = prop_picks(player_rows(hist_rows), games_ahead, odds_store,
                                 current.current_teams(), released)
    else:
        props, held = {m: [] for m in config.MARKETS}, {}
    return {"games": teams, "props": props, "held_back": held,
            "odds_checked_at": odds_store.get("updated")}
