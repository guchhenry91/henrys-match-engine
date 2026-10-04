"""The live MLB slate: team winners from Elo; team totals and player props at
BOOKMAKER lines only.

Players are matched from the book's name to this season's box scores (which carry
the name and the Retrosheet id, mlb.current), so their full history joins up. A
player's team is his newest box score this season; a player the book lists in a
game his known team is not in is skipped, never guessed onto a side. A book line
outside the range the model was trained and gated on is dropped, not
extrapolated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlb import config, current, data, features
from mlb.backtest import predict as model_predict
from nfl import odds as odds_mod
from nfl import selection
from nfl.games_model import run_elo
from oddsapi import props as shared
from oddsapi import props as shared_props
from nfl.model import PropModel
from tracking import recent

ROOT = Path(__file__).resolve().parent.parent
ODDS = ROOT / "data-raw" / "mlb" / "odds_api.json"
REPORT = ROOT / "data-raw" / "mlb" / "backtest_report.json"
BOARD_HOURS = 36.0
TOP_PER_MARKET = 3
MIN_PROBABILITY = 0.50
UPCOMING = ("Scheduled", "Pre-Game", "Warmup")


def _read(path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def slate(sched: pd.DataFrame, now) -> pd.DataFrame:
    s = sched[sched["status"].isin(UPCOMING)].copy()
    s["t"] = pd.to_datetime(s["start"], utc=True)
    now = pd.Timestamp(shared.utc(now))
    s = s[(s["t"] > now) & (s["t"] <= now + pd.Timedelta(hours=BOARD_HOURS))]
    return s.sort_values("t").drop(columns=["t"])


def team_picks(sched: pd.DataFrame, games_ahead: pd.DataFrame, odds_store: dict) -> list:
    params = ((_read(REPORT).get("markets") or {}).get("team_winner") or {}).get(
        "elo_params") or {"k": 2.0, "home_edge": 40.0, "regression": 0.2}
    hist = data.games()
    # The WHOLE season's results, not just the window around today: the Elo has
    # to walk every 2026 game to know where each team stands. One free request.
    try:
        this = current.schedule(f"{current.SEASON}-03-01",
                                (pd.Timestamp(games_ahead["game_date"].max()) + pd.Timedelta(days=1)).date().isoformat())
    except Exception:
        this = sched.copy()
    this = pd.DataFrame({
        "game_id": this["game_pk"].map(lambda k: f"MLB{k}"), "season": current.SEASON,
        "game_date": pd.to_datetime(this["game_date"]),
        "home_team": this["home_team"], "away_team": this["away_team"],
        "home_score": pd.to_numeric(this["home_score"], errors="coerce"),
        "away_score": pd.to_numeric(this["away_score"], errors="coerce"),
        "neutral": False, "played": this["status"].isin(current.FINISHED)})
    this["winner"] = None
    this.loc[this["played"] & (this["home_score"] > this["away_score"]), "winner"] = "home"
    this.loc[this["played"] & (this["home_score"] < this["away_score"]), "winner"] = "away"
    walked = run_elo(pd.concat([hist, this], ignore_index=True),
                     params["k"], params["home_edge"], params["regression"])
    by_id = {r.game_id: r for r in walked[walked["season"] == current.SEASON].itertuples()}
    out = []
    for g in games_ahead.itertuples():
        row = by_id.get(f"MLB{g.game_pk}")
        if row is None:
            continue
        p_home = float(row.prob_home)
        home_pick = p_home >= 0.5
        entry = {"game_id": str(g.game_pk), "tipoff": g.start, "kickoff": g.start,
                 "date": g.start, "home": g.home_team, "away": g.away_team,
                 "home_name": g.home_name, "away_name": g.away_name,
                 "home_pitcher": g.home_pitcher_name, "away_pitcher": g.away_pitcher_name,
                 "pick": g.home_team if home_pick else g.away_team,
                 "p_home": round(p_home, 4), "p_pick": round(p_home if home_pick else 1 - p_home, 4),
                 "rating_home": round(float(row.rating_home), 1),
                 "rating_away": round(float(row.rating_away), 1), "game_type": g.game_type}
        ml = ((odds_store.get("games") or {}).get(str(g.game_pk)) or {}).get("moneyline")
        if ml:
            bp = ml["home"] if home_pick else ml["away"]
            entry.update({"book": ml["book"], "book_p_pick": bp, "edge": round(entry["p_pick"] - bp, 4)})
        out.append(entry)
    return out


def _rows(kind: str) -> pd.DataFrame:
    seasons = config.SEASONS[-config.TRAIN_SEASONS:]
    loader = {"batter": data.batter_games, "pitcher": data.pitcher_games, "team": data.team_games}[kind]
    hist = loader(seasons)
    live = current.load({"batter": "batting", "pitcher": "pitching", "team": "team"}[kind])
    if not live.empty:
        live = live[[c for c in hist.columns if c in live.columns] + [c for c in ("NAME",) if c in live.columns]]
    return pd.concat([hist, live], ignore_index=True) if not live.empty else hist


def _price(prob, quote):
    return {"book": quote.get("book"), "line_source": quote.get("source"),
            "book_price": quote.get("odd_over"), "book_p": quote.get("over"),
            "book_p_fair": quote.get("fair", True),
            "edge": round(prob - float(quote["over"]), 4) if quote.get("over") else None}


def unavailable(kind: str, pid, team: str, g, avail: dict | None) -> str | None:
    """Why this player should NOT be priced for game `g`, or None if he can play.

    The books' listings alone used to decide who appears, so a player on the
    injured list, sent down or resting could still carry a prop. Checked against
    MLB's own feed: on the team's active roster; in the posted lineup once there is
    one (batters); the announced probable starter (pitchers).
    """
    if not avail:
        return None
    active = (avail.get("active") or {}).get(team)
    if active is not None and pid not in active:
        return "not on the active roster"
    if kind == "batter":
        lineup = (avail.get("lineups") or {}).get((int(g.game_pk), team))
        if lineup and pid not in lineup:
            return "not in the posted lineup"
    if kind == "pitcher":
        side = "home" if team == g.home_team else "away"
        probable = getattr(g, f"{side}_pitcher", None)
        if probable is not None and not pd.isna(probable) \
                and current.retro_id(int(probable)) != pid:
            return "not the probable starter"
    return None


def props_and_totals(games_ahead: pd.DataFrame, odds_store: dict, released: list,
                     avail: dict | None = None, collect: dict | None = None) -> dict:
    """Board shortlist per market. `collect`, when given, receives EVERY priced
    pick per game ({game_pk: {market: [picks]}}) for the game pages."""
    odds_games = odds_store.get("games") or {}
    out = {m: [] for m in config.MARKETS}
    # The gate's fitted correction for publishing only the top picks (nfl/selection.py).
    shrink = {m: float(((v or {}).get("selection") or {}).get("shrink") or 1.0)
              for m, v in (_read(REPORT).get("markets") or {}).items()}
    for market, spec in config.MARKETS.items():
        if market not in released:
            continue
        kind = spec["kind"]
        rows = _rows(kind)
        live = current.load({"batter": "batting", "pitcher": "pitching", "team": "team"}[kind])
        latest = live.sort_values("game_date").groupby("PLAYER_ID").tail(1) if not live.empty else pd.DataFrame()
        names = {}
        if not latest.empty and "NAME" in latest:
            for r in latest.itertuples():
                names.setdefault(odds_mod.norm_name(r.NAME), set()).add(r.PLAYER_ID)
        team_of = dict(zip(latest["PLAYER_ID"], latest["TEAM"])) if not latest.empty else {}
        synth, meta, dropped = [], {}, {}
        for g in games_ahead.itertuples():
            og = odds_games.get(str(g.game_pk)) or {}
            if kind == "team":
                for side in ("home", "away"):
                    q = (og.get("team_totals") or {}).get(side)
                    if not q:
                        continue
                    team = getattr(g, f"{side}_team")
                    opp = g.away_team if side == "home" else g.home_team
                    synth.append(_synth(team, team, opp, side == "home", g, spec))
                    meta[(team, synth[-1]["GAME_ID"])] = (g, team, opp, side == "home", q,
                                                          getattr(g, f"{side}_name"))
                continue
            for name, q in ((og.get("props") or {}).get(market) or {}).items():
                ids = names.get(odds_mod.norm_name(name)) or set()
                if len(ids) != 1:
                    continue
                pid = next(iter(ids))
                team = team_of.get(pid)
                if team not in (g.home_team, g.away_team):
                    continue
                reason = unavailable(kind, pid, team, g, avail)
                if reason:
                    dropped[reason] = dropped.get(reason, 0) + 1
                    continue
                opp = g.away_team if team == g.home_team else g.home_team
                synth.append(_synth(pid, team, opp, team == g.home_team, g, spec))
                meta[(pid, synth[-1]["GAME_ID"])] = (g, team, opp, team == g.home_team, q, name)
        if dropped:
            print(f"  {market}: dropped " + ", ".join(f"{n} {r}" for r, n in dropped.items()))
        if not synth:
            continue
        # ONE UPCOMING GAME PER PLAYER AT A TIME. Features are ENTERING (shift(1)):
        # priced together, a player's second game (tomorrow, or a doubleheader's
        # nightcap) would see the first game's zero-stat placeholder as his last
        # game. Each player's 1st, 2nd... upcoming game is built in its own pass.
        synth_df = pd.DataFrame(synth)
        nth = synth_df.sort_values(["game_date", "GAME_ID"]).groupby("PLAYER_ID").cumcount()
        parts = [features.build(pd.concat([rows, synth_df[nth.reindex(synth_df.index) == k]],
                                          ignore_index=True), market)
                 for k in sorted(nth.unique())]
        is_real = [~b["GAME_ID"].astype(str).str.startswith("NEXT") for b in parts]
        built = pd.concat([parts[0][is_real[0]]] + [b[~r] for b, r in zip(parts, is_real)],
                          ignore_index=True)
        is_synth = built["GAME_ID"].astype(str).str.startswith("NEXT")
        train = features.augment_lines(built[~is_synth], market)
        ask = built[is_synth]
        # FIT ONCE, reuse for the main line and every ladder rung: one model per
        # market, or one per line step where config.PER_STEP_MODELS says.
        per_step = market in config.PER_STEP_MODELS
        models = ({int(st): PropModel(market).fit(train[train["line_step"] == st])
                   for st in config.LINE_STEPS[market]} if per_step
                  else {None: PropModel(market).fit(train)})
        steps = config.LINE_STEPS[market]

        def prob_at(row_frame, base_line, line):
            """The model's over probability at `line`, or None where the line is
            outside the range it was trained and gated on."""
            if not config.MIN_LINE[market] <= line <= config.MAX_LINE[market]:
                return None
            # Step 0 is trained at the base line FLOORED at the book minimum
            # (features.augment_lines), so a batter whose HR/RBI median is 0
            # (base -0.5) is asked about 0.5 at step 0 -- not at step 1, which
            # HR and RBI do not carry and which dropped every one of them.
            step = (0 if abs(line - max(base_line, config.MIN_LINE[market])) < 1e-9
                    else int(round(line - base_line)))
            if not min(steps) <= step <= max(steps):
                return None
            model = models.get(step if per_step else None)
            if model is None:
                return None
            asked = features.at_line(row_frame, line)
            asked["line_step"] = step
            value = float(model.predict(asked)[0])
            return value if np.isfinite(value) else None

        alts = {str(k): (v or {}).get("alt") or {} for k, v in odds_games.items()}
        # Each player's last five games of THIS stat, for the card's strip.
        past = allrows[~allrows["GAME_ID"].astype(str).str.startswith("NEXT")]
        last5 = recent.last_n(past, ask["PLAYER_ID"], spec["stat"])
        picks = []
        for idx, r in ask.iterrows():
            g, team, opp, home, q, name = meta[(r["PLAYER_ID"], r["GAME_ID"])]
            line = float(q["line"])
            row_frame = ask.loc[[idx]]
            prob = prob_at(row_frame, float(r["base_line"]), line)
            if prob is None:
                continue                      # outside what the model was gated on
            sides = selection.sides(
                {"market": market, "player": name, "player_id": r["PLAYER_ID"],
                 "team": team, "opponent": opp, "home": bool(home),
                 "game_id": str(g.game_pk), "tipoff": g.start, "kickoff": g.start,
                 "line": line, "probability": round(prob, 4),
                 "form5": round(float(r["form5"]), 2),
                 "last_five": last5.get(r["PLAYER_ID"], []),
                 "games_before": int(r["games_before"]), **_price(prob, q)},
                q, shrink.get(market, 1.0), allow_under=(market != "hr"))
            # THE LADDER UP: the book's higher lines for this player, each with its
            # chance of hitting (model where gated, the book's price elsewhere).
            alt = odds_mod.match_player(alts.get(str(g.game_pk), {}).get(market) or {}, name)
            for pick in sides:
                pick["ladder_up"] = shared_props.ladder_up(
                    alt, pick, lambda l, rf=row_frame, bl=float(r["base_line"]): prob_at(rf, bl, l),
                    shrink=shrink.get(market, 1.0))
            picks.extend(sides)
            if collect is not None:
                for s_ in sides:
                    collect.setdefault(str(s_["game_id"]), {}).setdefault(market, []).append(s_)
        # Home runs: a ~12% event, so the bar is the PRICE, not 50% -- publish
        # where the model rates him above the book's own view.
        keep = [p for p in picks if (p["edge"] or 0) > 0] if market == "hr" else \
            [p for p in picks if p["probability"] >= MIN_PROBABILITY
             # The MODEL'S side, not the book's: at least as sure as the book is.
             and (p.get("edge") is None or p["edge"] >= 0)]
        by_game = {}
        for p in sorted(keep, key=lambda p: -p["probability"]):
            by_game.setdefault(p["game_id"], []).append(p)
        out[market] = sorted([p for ps in by_game.values() for p in ps[:TOP_PER_MARKET]],
                             key=lambda p: -p["probability"])
        for p in out[market]:
            p["on_board"] = True
        print(f"  {market}: {len(picks)} priced at a book line, {len(out[market])} published")
    return out


def _synth(pid, team, opp, home, g, spec) -> dict:
    """A future row: dummy outcome values, since every feature is ENTERING."""
    row = {"PLAYER_ID": pid, "TEAM": team, "GAME_ID": f"NEXT{g.game_pk}",
           "game_date": pd.Timestamp(g.game_date), "season": current.SEASON,
           "opponent": opp, "is_home": float(home), spec["stat"]: 0.0}
    row[spec["opp"]] = 4.0 if spec["kind"] == "batter" else 25.0 if spec["kind"] == "pitcher" else 38.0
    return row


def build(sched: pd.DataFrame, now, released: list) -> dict:
    games_ahead = slate(sched, now)
    odds_store = _read(ODDS)
    game_props = {}
    if games_ahead.empty:
        return {"games": [], "props": {m: [] for m in config.MARKETS},
                "odds_checked_at": odds_store.get("updated")}
    return {"games": team_picks(sched, games_ahead, odds_store),
            "props": props_and_totals(games_ahead, odds_store, released,
                                      current.availability(games_ahead), collect=game_props),
            "game_props": game_props,
            "odds_checked_at": odds_store.get("updated")}
