"""Build the published NFL board: data/nfl/board.json.

WHAT THE BOARD CAN SAY BEFORE A SEASON STARTS. Every feature is built from games
already played, so week 1 of a new season is projected from the previous one -- a
receiver's last five games are his last five, whichever September they fall
either side of. Elo carries across seasons with a regression toward the mean,
which is the same claim in the team model: a rating in September is a weaker
claim than the same number in December.

What it CANNOT know is a player who changed teams in the offseason and has not
yet played for the new one. His stats follow him; his listed club comes from his
last appearance. That is stated on the card rather than hidden, because a
projection attached to the wrong club is worse than no projection at all.
"""
import json

from scripts import json_safe
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl import (config, data, depth, features, games_model,
                 odds as odds_mod, rosters)
from nfl import picks
from nfl import book_lines
from nfl import market_blend
from nfl import selection
from nfl import news as player_news
from nfl.model import PropModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nfl"
REPORT = ROOT / "data-raw" / "nfl" / "backtest_report.json"

TOP_PER_MARKET = 3          # a shortlist, not a database
MIN_PROBABILITY = 0.50      # never publish a leg the model itself makes a dog
# BOOKMAKER LINES ONLY (owner's instruction, 2026-09-27). A yardage prop is
# published only where a bookmaker quotes that player's line -- bet365, or the
# Pinnacle / DraftKings / FanDuel gap-fill. The player's own median is still how
# the model is TRAINED, but it is never the line a published pick stands on: a
# line no one offers is not a bet anyone can make. Anytime TD has no line.
REQUIRE_BOOK_LINE = True


def _report() -> dict:
    try:
        return json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _released() -> set:
    return set(_report().get("released_markets") or [])


def _evidence() -> dict:
    """The measured numbers, published WITH the picks so they travel together."""
    out = {}
    for market, result in (_report().get("markets") or {}).items():
        overall = result.get("overall") or {}
        out[market] = {
            "released": result.get("released"),
            "n": overall.get("n"),
            "brier": overall.get("brier"),
            "baseline_brier": overall.get("baseline_brier"),
            "ece": overall.get("ece"),
            "accuracy": overall.get("accuracy"),
            "seasons_scored": sorted((result.get("per_season") or {}).keys()),
        }
    return out


def availability() -> dict:
    """player name -> {"status": "out"|"doubt", ...} from the API-NFL report.

    ABSENCE MEANS NOT REPORTED, never "confirmed fit". The distinction matters: a
    quiet file because the sync failed looks identical to a quiet file because
    everyone is healthy, and treating the first as the second would publish a
    ruled-out player with full confidence. So this only ever REMOVES or FLAGS
    players it has positive information about.
    """
    path = ROOT / "data-raw" / "nfl" / "injuries.json"
    try:
        api = json.loads(path.read_text(encoding="utf-8")).get("players") or {}
    except Exception:
        api = {}
    # Manual news from the cloud team-news routine (nfl/news.py): late scratches
    # and inactives the API report has not caught. Only ever more cautious.
    return player_news.merge(api, player_news.load_player_news(
        ROOT / "data-raw" / "nfl" / "news.json"))


def book_prices() -> dict:
    """Fair (de-vigged) book probabilities per fixture, or {} when none exist.

    Empty is the normal state until books post, and it must read as "no price",
    never as "the book thinks this is unlikely" -- the second manufactures an
    enormous false edge out of a missing file.
    """
    path = ROOT / "data-raw" / "nfl" / "odds.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("games") or {}
    except Exception:
        return {}


def book_props() -> dict:
    """{"HOME|AWAY": {market: {player: quote}}} -- the bookmaker's prop lines.

    Empty means the book quoted nothing, and the board then falls back to each
    player's own line and says so on the card; it never invents a book line.
    """
    path = ROOT / "data-raw" / "nfl" / "odds.json"
    try:
        bet365 = json.loads(path.read_text(encoding="utf-8")).get("props") or {}
    except Exception:
        bet365 = {}
    # GAP-FILL from The Odds API (Pinnacle, DraftKings, FanDuel) for players
    # bet365 does not quote -- receiving yards above all, which API-NFL never
    # carries. bet365 is never overwritten (nfl/book_lines.py).
    try:
        extra = json.loads((ROOT / "data-raw" / "nfl" / "odds_api_props.json")
                           .read_text(encoding="utf-8")).get("games") or {}
    except Exception:
        extra = {}
    return book_lines.merge(bet365, {k: (g or {}).get("props") or {}
                                     for k, g in extra.items()})


def _released_report() -> dict:
    try:
        return json.loads((ROOT / "data-raw" / "nfl" / "backtest_report.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def book_alts() -> dict:
    """{"HOME|AWAY": {market: {player: ladder}}} -- bookmakers' ALTERNATE lines
    (DraftKings / FanDuel via The Odds API), for the "70%+ line" on each card."""
    try:
        games = json.loads((ROOT / "data-raw" / "nfl" / "odds_api_props.json")
                           .read_text(encoding="utf-8")).get("games") or {}
    except Exception:
        return {}
    return {k: (g or {}).get("alt") or {} for k, g in games.items() if (g or {}).get("alt")}


def depth_population(roster_index: dict, rosters_complete: bool, known_ids: list) -> list:
    """The players a depth chart must recognise before it is trusted.

    PLAYERS CURRENTLY ON A ROSTER, not everyone who played last season. The old
    population included retired, cut and unsigned players who will never appear
    on a current depth chart, so a complete, same-day chart (32 teams, 1,962
    players) "recognised" only 73.5% and was refused -- and the board published
    21 backup quarterbacks, Marcus Mariota and Shedeur Sanders among them. Against
    current rosters the same chart recognises 98.8%, and 99.6% of the board's own
    candidates. The corroboration idea is unchanged: a chart that does not know
    the players actually on teams is still refused. Only when the roster file is
    itself untrusted does this fall back to last season's players.
    """
    if rosters_complete and roster_index:
        return sorted(str(i) for i in roster_index)
    return known_ids


def upcoming_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """The next slate: the earliest unplayed week."""
    future = schedule[~schedule["played"]].copy()
    if future.empty:
        return future
    first = future.sort_values(["season", "week"]).iloc[0]
    future = future[(future["season"] == first["season"])
                    & (future["week"] == first["week"])]
    return future.sort_values("gameday")


def next_game_frame(player_weeks, games, upcoming, fixtures, market,
                    roster_index, rosters_complete):
    """One built row per active player whose CURRENT team plays in `upcoming`,
    with that game's week, opponent, venue and rest -- see player_projections."""
    last = player_weeks.sort_values(["season", "week"]).groupby("player_id").tail(1)
    newest = int(player_weeks["season"].max())
    last = last[last["season"] >= newest - (config.ACTIVE_WITHIN_SEASONS - 1)]
    synth = []
    numeric = [c for c in player_weeks.columns
               if c not in ("player_id", "player_display_name", "position", "team",
                            "season", "week", "season_type", "opponent_team")
               and pd.api.types.is_numeric_dtype(player_weeks[c])]
    for _, row in last.iterrows():
        team, why = rosters.reconcile(row["player_id"], row["team"],
                                      roster_index, rosters_complete)
        if team not in fixtures:
            continue
        game, opponent, _ = fixtures[team]
        new = row.copy()
        new["team"], new["opponent_team"] = team, opponent
        new["season"], new["week"] = int(game["season"]), int(game["week"])
        new["season_type"] = "REG"
        new[numeric] = 0.0
        new["_next"] = True
        new["_why"] = why
        synth.append(new)
    if not synth:
        return None
    rows = pd.concat([player_weeks.assign(_next=False), pd.DataFrame(synth)],
                     ignore_index=True)
    context = None
    if games is not None:
        context = pd.concat([games, upcoming[[c for c in games.columns if c in upcoming.columns]]],
                            ignore_index=True).drop_duplicates(
            subset=["season", "week", "home_team", "away_team"], keep="first")
    built = features.build(rows, market, games=context)
    return built[built["_next"].fillna(False).astype(bool)].copy()


def player_projections(player_weeks, games, market, upcoming, injuries=None,
                       roster_index=None, rosters_complete=False,
                       depth_index=None, depth_trusted=False,
                       book_quotes=None, alt_quotes=None) -> list:
    """Project every eligible player in the upcoming slate for one market.

    `book_quotes` is {"HOME|AWAY": {player: quote}} for this market. Where the
    bookmaker quotes a player, the model is asked about THE BOOK'S LINE; only
    where it does not is the player's own median used, and the card says which.
    """
    injuries = injuries or {}
    roster_index = roster_index or {}
    depth_index = depth_index or {}
    frame = features.build(player_weeks, market, games=games)
    if frame.empty:
        return []
    # Fitted across a spread of lines, so it can be asked about the book's number
    # and mean it -- see features.augment_lines and config.LINE_MULTIPLIERS.
    fit_on = frame
    if config.TRAIN_SEASONS:
        fit_on = frame[frame["season"] > int(frame["season"].max()) - config.TRAIN_SEASONS]
    model = PropModel(market).fit(features.augment_lines(fit_on, market))

    fixtures = {}
    for _, game in upcoming.iterrows():
        fixtures[game["home_team"]] = (game, game["away_team"], True)
        fixtures[game["away_team"]] = (game, game["home_team"], False)

    # THE NEXT GAME'S OWN ROW. The board used to price each player from the row
    # of his LAST game -- inputs describing the moment before that game: form
    # missing his most recent game, and LAST WEEK'S opponent, venue and rest.
    # Measured on the 2025 walk-forward, that cost the published picks 2-4.5pt
    # of hit rate against the same model given the right inputs (receiving 56.9%
    # -> 54.8%, rushing 59.7% -> 56.5%, passing 59.3% -> 54.9%). The gate scores
    # each game with that game's own entering inputs; the board now does too: a
    # placeholder row for the upcoming game (outcome columns zero -- every input
    # is ENTERING, shift(1), so a row never reads its own values) goes through the
    # same build, and the model is still trained only on real games.
    latest = next_game_frame(player_weeks, games, upcoming, fixtures, market,
                             roster_index, rosters_complete)
    if latest is None or latest.empty:
        return []

    # RECONCILE THE CLUB BEFORE choosing who is playing. nflverse says where a man
    # last PLAYED; the roster snapshot says where he IS, and through an offseason
    # those differ. Doing this after the fixture lookup would project a moved
    # player onto his OLD team's game.
    resolved, reasons = [], []
    for _, row in latest.iterrows():
        team, why = rosters.reconcile(row["player_id"], row["team"],
                                      roster_index, rosters_complete)
        resolved.append(team)
        reasons.append(why)
    latest = latest.assign(_team=resolved, _why=reasons)
    latest = latest[latest["_team"].notna()].copy()
    if latest.empty:
        return []
    latest["team"] = latest["_team"]

    playing = latest[latest["team"].isin(fixtures)]
    if playing.empty:
        return []

    # THE BOOKMAKER'S LINE, WHERE IT QUOTES ONE. Joined by name within the
    # player's own game only; an ambiguous name is refused, never guessed.
    book_quotes = book_quotes or {}
    quoted, found = playing.copy(), {}
    for idx, player in playing.iterrows():
        game = fixtures[player["team"]][0]
        quote = odds_mod.match_player(
            book_quotes.get(f"{game['home_team']}|{game['away_team']}") or {},
            player["player_display_name"])
        if quote is None:
            continue
        found[idx] = quote
        if market != "anytime_touchdown" and quote.get("line") is not None:
            quoted.at[idx, "line"] = float(quote["line"])

    rows, dropped_depth, below_floor = [], [], []
    for (idx, player), prob in zip(quoted.iterrows(), model.predict(quoted)):
        game, opponent, is_home = fixtures[player["team"]]
        name = player["player_display_name"]
        quote = found.get(idx)
        from_book = (quote is not None and market != "anytime_touchdown"
                     and quote.get("line") is not None)
        # A book line below the lines the model was trained on is outside what it
        # has been measured on, so the pick is dropped rather than extrapolated.
        if from_book and float(quote["line"]) < config.MIN_LINE[market]:
            below_floor.append(f"{name} {quote['line']}")
            continue
        book_p = book_price = None
        if quote is not None:
            if market == "anytime_touchdown":
                book_p, book_price = quote.get("raw_yes"), quote.get("odd")
            elif from_book:
                book_p, book_price = quote.get("over"), quote.get("odd_over")

        # RULED OUT MEANS OFF THE BOARD. His last five games look exactly as good
        # as anyone's right up until he is inactive, which is precisely why a
        # projection for a player who will not dress is the most misleading thing
        # this board could print.
        report = injuries.get(name) or {}
        if report.get("status") == "out":
            continue

        # TOO FAR DOWN THE DEPTH CHART IS THE SAME KIND OF FACT AS RULED OUT: the
        # model rates him on the snaps he took in relief, and cannot see that he
        # will not take any. Only applied when the chart corroborates the board
        # (see nfl/depth.py) -- otherwise a thin or broken chart would delete
        # real players, which is exactly what happened in August.
        entry = depth_index.get(str(player["player_id"]))
        if depth_trusted:
            ok, why = depth.verdict(market, entry)
            if not ok:
                dropped_depth.append(f"{name} ({why})")
                continue

        # THE BOOKMAKER'S ALTERNATE LADDER, each rung asked of the model. Only
        # rungs inside the line range the model was trained on get a model
        # number (config.LINE_MULTIPLIERS x his median); others carry None and
        # are judged on the book's price alone. See nfl/market_blend.ladder_up.
        ladder = None
        alt = odds_mod.match_player(
            (alt_quotes or {}).get(f"{game['home_team']}|{game['away_team']}") or {}, name)
        if alt and market != "anytime_touchdown":
            try:
                median = float(player.get("median") or 0.0)
                lo = median * min(config.LINE_MULTIPLIERS)
                hi = median * max(config.LINE_MULTIPLIERS) + 1.0
                rungs = [(float(l), float(pr)) for l, pr in alt["ladder"]]
                inside = [l for l, _ in rungs if lo <= l <= hi and l >= config.MIN_LINE[market]]
                raw = {}
                if inside:
                    asked = pd.concat([features.at_line(quoted.loc[[idx]], l) for l in inside])
                    raw = dict(zip(inside, (float(x) for x in model.predict(asked))))
                ladder = {"book": alt["book"], "source": alt["source"],
                          "rungs": [{"line": l, "price": pr, "raw": raw.get(l)} for l, pr in rungs]}
            except Exception as exc:      # never let a ladder take the board down
                print(f"  {market}: ladder for {name} skipped ({type(exc).__name__}: {exc})")
                ladder = None

        last_five = [float(v) for v in (player["last_five"] or [])]
        rows.append({
            "ladder": ladder,
            "market": market,
            "player": name,
            "player_id": player["player_id"],
            "team": player["team"],
            "opponent": opponent,
            "home": bool(is_home),
            # The REAL kickoff (nfl.data._kickoff_utc), not the bare date this
            # used to carry. A pick frozen against midnight is frozen hours
            # early, before the inactives report, or is marked late and voided.
            "game_id": game["game_id"],
            "kickoff": pd.Timestamp(game["kickoff"]).isoformat(),
            "line": None if pd.isna(player["line"]) else float(player["line"]),
            # Whose line this is: the bookmaker that quoted it ("bet365", or a
            # gap-fill book -- "pinnacle", "draftkings", "fanduel"), or "model",
            # the player's own median, used only where no book quotes him.
            "line_source": ((quote.get("source") or "bet365") if from_book else
                            None if market == "anytime_touchdown" else "model"),
            "probability": round(float(prob), 4),
            "book": (quote or {}).get("book") if book_p is not None else None,
            "book_price": book_price,
            "book_price_under": (quote or {}).get("odd_under") if book_p is not None else None,
            # De-vigged for yards (both sides quoted); RAW for anytime TD, which
            # is one-sided, so its edge is understated rather than flattered.
            "book_p": book_p,
            "book_p_fair": market != "anytime_touchdown",
            "edge": (round(float(prob) - float(book_p), 4) if book_p is not None else None),
            # THE LAST FIVE, as asked: the individual games, not an average, and
            # the same five the projection was computed from. A board that shows
            # one form window while the model used another is explaining itself
            # with numbers it never saw.
            "last_five": last_five,
            "last_five_average": (round(sum(last_five) / len(last_five), 1)
                                  if last_five else None),
            # "doubt" survives onto the board rather than being dropped: a
            # questionable player who plays is a real pick, and hiding the doubt
            # is what would mislead. Absent from the report means NOT REPORTED,
            # which is why the field says so rather than saying "fit".
            "availability": report.get("status") or "not reported",
            # How the club on this card was decided, so a reader can tell a
            # confirmed roster spot from an inference off last season.
            "club_source": player.get("_why", "unknown"),
            "injury_note": report.get("detail") or None,
            "depth_pos": (entry or {}).get("pos"),
            "depth_rank": (entry or {}).get("rank"),
            "depth_label": (f"{entry['pos']}{entry['rank']}" if entry else None),
            "games_played": int(player["games_before"]),
            "as_of_season": int(player["season"]),
            "as_of_week": int(player["week"]),
        })
    if dropped_depth:
        print(f"  {market}: depth chart removed {len(dropped_depth)} -> "
              f"{dropped_depth[:6]}")
    if below_floor:
        print(f"  {market}: book line below the trained floor, dropped "
              f"{len(below_floor)} -> {below_floor[:6]}")
    sourced = sum(1 for r in rows if r.get("line_source") not in (None, "model"))
    print(f"  {market}: {sourced} of {len(rows)} on the bookmaker's line")
    rows.sort(key=lambda r: -r["probability"])
    return rows


def _odds_checked_at():
    """When the odds sync last ran, from the file it writes. None if it never has.

    Published so an empty odds block is legible: "asked an hour ago and there was
    nothing" and "never asked" look identical otherwise, and only one of them is a
    reason to doubt the board.
    """
    path = ROOT / "data-raw" / "nfl" / "odds.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("updated")
    except Exception:
        return None


def build() -> dict:
    player_weeks = data.player_weeks()
    history = data.games()
    current = data.games(seasons=(config.CURRENT_SEASON,))
    schedule = pd.concat([history, current], ignore_index=True)

    params = games_model.fit_parameters(history[history["played"]])
    walked = games_model.run_elo(schedule, params["k"], params["home_edge"],
                                 params["regression"])
    upcoming = upcoming_games(schedule)
    released = _released()
    injuries = availability()
    prices = book_prices()
    quotes = book_props()
    alts = book_alts()
    roster = data.rosters()
    roster_index = rosters.build_index(roster)

    # THE CURRENT DEPTH CHART. Answers two things the roster cannot: who is
    # actually being carried (four days after the cut to 53 the roster file still
    # listed 90 active a team) and who STARTS. See nfl/depth.py.
    chart = data.depth_charts()
    depth_index = depth.build_index(chart)
    # The roster may only overrule the box scores if it demonstrably describes the
    # same league. Measured BY ID against the players we independently know were
    # active last season -- see rosters.corroborates for what happened without it.
    latest_season = player_weeks[player_weeks["season"] == player_weeks["season"].max()]
    known_ids = sorted({str(i) for i in latest_season["player_id"] if str(i) != "nan"})
    rosters_complete, agreement = rosters.corroborates(roster_index, known_ids)
    # Same corroboration discipline for the depth chart: it may only overrule the
    # board if it recognises the board's own players. A chart that does not is far
    # more likely to be broken than to be evidence the board is wrong.
    depth_ids = depth_population(roster_index, rosters_complete, known_ids)
    depth_trusted, depth_coverage, depth_size = depth.usable(depth_index, depth_ids)
    print(f"depth chart: {depth_size} players, recognises "
          f"{depth_coverage:.1%} of {len(depth_ids)} "
          f"{'current roster players' if depth_ids is not known_ids else 'last-season players'} -> "
          f"{'APPLIED' if depth_trusted else 'IGNORED (below the bar)'}")
    if not rosters_complete:
        print(f"WARNING: roster file recognises only {agreement:.0%} of known active "
              f"players; NOT trusting it to drop or move anyone")
    else:
        print(f"roster file agrees with {agreement:.0%} of known active players; "
              f"{len(roster_index)} placed")

    games_out = []
    for _, game in upcoming.iterrows():
        priced = walked[(walked["season"] == game["season"])
                        & (walked["week"] == game["week"])
                        & (walked["home_team"] == game["home_team"])]
        if priced.empty:
            continue
        prob_home = float(priced.iloc[0]["prob_home"])
        pick_home = prob_home >= 0.5
        games_out.append({
            "game_id": game["game_id"],
            "season": int(game["season"]), "week": int(game["week"]),
            "kickoff": pd.Timestamp(game["kickoff"]).isoformat(),
            "home": game["home_team"], "away": game["away_team"],
            "neutral": str(game.get("location", "Home")).lower() == "neutral",
            "p_home": round(prob_home, 4),
            "pick": game["home_team"] if pick_home else game["away_team"],
            "p_pick": round(prob_home if pick_home else 1 - prob_home, 4),
            "rating_home": round(float(priced.iloc[0]["rating_home"]), 1),
            "rating_away": round(float(priced.iloc[0]["rating_away"]), 1),
            "gradeable": "team_winner" in released,
        })
        # THE EDGE, where a price exists. Everything else on this board is
        # calibrated against history; only this compares the model with what a
        # bookmaker will actually pay. Absent a price the fields stay null rather
        # than defaulting to something that looks like agreement.
        line = prices.get(f"{game['home_team']}|{game['away_team']}")
        entry = games_out[-1]
        entry["book"] = None
        entry["book_p_pick"] = None
        entry["edge"] = None
        entry["verdict"] = None
        if line:
            book_p = line["home"] if pick_home else line["away"]
            verdict, gap = odds_mod.value_verdict(entry["p_pick"], book_p)
            entry["book"] = line.get("book")
            entry["book_p_pick"] = book_p
            entry["book_price_overround"] = line.get("overround")
            entry["edge"] = gap
            entry["verdict"] = verdict

    props, awaiting_book = {}, {}
    blend_w = market_blend.weights()
    for market in config.MARKETS:
        if market not in released:
            props[market] = {"released": False, "picks": []}
            continue
        projections = player_projections(player_weeks, history, market, upcoming,
                                         injuries=injuries,
                                         roster_index=roster_index,
                                         rosters_complete=rosters_complete,
                                         depth_index=depth_index,
                                         depth_trusted=depth_trusted,
                                         book_quotes={k: (v or {}).get(market) or {}
                                                      for k, v in quotes.items()},
                                         alt_quotes={k: (v or {}).get(market) or {}
                                                     for k, v in alts.items()})
        if REQUIRE_BOOK_LINE and market != "anytime_touchdown":
            waiting = sum(1 for p in projections if p.get("line_source") == "model")
            projections = [p for p in projections if p.get("line_source") != "model"]
            awaiting_book[market] = waiting
            print(f"  {market}: {waiting} projection(s) held back -- no bookmaker line yet")
        # PULLED TOWARD THE BOOK (nfl/market_blend.py). The raw model is
        # overconfident on yardage in the live record, so only the fitted share w
        # of its disagreement with the book's de-vigged price is kept. The raw
        # number is published alongside as p_model, never hidden.
        if market == "anytime_touchdown":
            # The gate's correction for publishing only the top picks
            # (nfl/selection.py), from backtest_report.json. Yardage needs none
            # here: its market blend is fitted on the published picks themselves.
            s_td = float(((((_released_report().get("markets") or {})
                            .get(market) or {}).get("selection") or {}).get("shrink") or 1.0))
            for p in projections:
                p["p_model"] = p["probability"]
                p["probability"] = round(selection.apply(p["probability"], s_td), 4)
        if market in market_blend.YARDAGE:
            w = blend_w.get(market, 0.0)
            for p in projections:
                if p.get("book_p") is None:
                    continue
                p["p_model"] = p["probability"]
                p["probability"] = round(market_blend.blend(p["probability"], p["book_p"], w), 4)
                p["edge"] = round(p["probability"] - p["book_p"], 4)
                p["blend_w"] = w
            for p in projections:
                p["ladder_up"] = market_blend.ladder_up(p.get("ladder"), p, w)
                p.pop("ladder", None)
            # A PICK IS THE MODEL'S VIEW, NOT THE BOOK'S. Where the raw model rates
            # the over BELOW the book's own price it leans UNDER -- so the bet it
            # actually supports is the under, priced at the book's under. Books
            # tend to shade overs (the public backs them), so a board of overs
            # only sat in the worst part of the market.
            overs, unders = [], []
            for p in projections:
                if p.get("p_model") is None or p["p_model"] >= p["book_p"]:
                    overs.append(p)
                    continue
                if p["probability"] > 0.5 or not p.get("book_price_under"):
                    continue          # leans under, but not enough to make it the bet
                u = dict(p)
                u["side"] = "under"
                u["probability"] = round(1.0 - p["probability"], 4)
                u["p_model"] = round(1.0 - p["p_model"], 4)
                u["book_p"] = round(1.0 - p["book_p"], 4)
                u["book_price"] = p["book_price_under"]
                u["edge"] = round(u["probability"] - u["book_p"], 4)
                u["ladder_up"] = []
                unders.append(u)
            print(f"  {market}: {len(overs)} over(s), {len(unders)} under(s) where the model "
                  f"leans under against the book")
            projections = sorted(overs + unders, key=lambda p: -p["probability"])
        shortlist = [p for p in projections if p["probability"] >= MIN_PROBABILITY]
        by_game = {}
        for pick in shortlist:
            game_key = pick["team"] + "|" + pick["opponent"]
            by_game.setdefault(game_key, []).append(pick)
        trimmed = []
        for picks_for_game in by_game.values():
            trimmed.extend(picks_for_game[:TOP_PER_MARKET])
        trimmed.sort(key=lambda p: -p["probability"])
        props[market] = {"released": True, "picks": trimmed,
                         "awaiting_book_line": awaiting_book.get(market, 0)}

    last_season = int(player_weeks["season"].max())
    last_week = int(player_weeks[player_weeks["season"] == last_season]["week"].max())
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "season": int(upcoming.iloc[0]["season"]) if not upcoming.empty else None,
        "week": int(upcoming.iloc[0]["week"]) if not upcoming.empty else None,
        "elo": {"k": params["k"], "home_edge": params["home_edge"],
                "regression": params["regression"]},
        # Stated plainly, because it bounds everything on the board: props are
        # projected from the last week of player data that exists upstream, which
        # before a season starts is the previous December.
        "player_data_through": {"season": last_season, "week": last_week},
        "games": games_out,
        "props": props,
        "evidence": _evidence(),
        # STATED, NOT HIDDEN. Each of these is a real hole a reader could
        # otherwise mistake for a signal, and the first one is visible on the
        # board right now: a backup quarterback carries a low line because he has
        # only played in relief, which makes "over" look easy until you notice he
        # may not take a snap.
        # Stated beside the roster check, because a reader deserves to know
        # whether the starter filter actually ran on the board they are seeing.
        "depth_check": {
            "source": "nflverse depth charts",
            "players": depth_size,
            "coverage_of_known_actives": round(depth_coverage, 3),
            "applied": depth_trusted,
            "caps": config.MAX_DEPTH_RANK,
            "note": ("Backups are removed, not flagged: a backup's line is his own "
                     "entering median, set in relief, so 'over' looks easy right "
                     "up until he takes no snap. Passing is capped at the starter "
                     "because one quarterback takes essentially every drop-back."
                     if depth_trusted else
                     "NOT APPLIED -- the depth chart did not recognise enough of "
                     "the board's own players to be trusted, so no player was "
                     "removed on its say-so."),
        },
        "roster_check": {
            "source": "nflverse rosters",
            "players_placed": len(roster_index),
            "teams": int(roster["team"].nunique()) if not roster.empty else 0,
            "agreement_with_known_players": round(agreement, 3),
            "trusted": rosters_complete,
        },
        "odds": {
            "fixtures_priced": len(prices),
            "source": "API-NFL, bet365 preferred" if prices else None,
            # Said plainly: without prices the board is calibrated, not proven
            # profitable, and those are different claims.
            "note": ("No book prices available yet -- the board states calibrated "
                     "probabilities and makes NO claim to beat a bookmaker."
                     if not prices else
                     "Edges are model probability minus the de-vigged book price."),
            # WHEN THIS WAS LAST ASKED, so "0 priced" cannot be confused with
            # "never checked" -- the same distinction the injury report draws
            # between "not reported" and "confirmed fit".
            "checked_at": _odds_checked_at(),
            "player_props": {
                "available": bool(quotes),
                "markets_quoted": sorted({m for g in quotes.values() for m in (g or {})}),
                "bet_type_ids": odds_mod.PLAYER_PROP_BETS,
                # Established 2026-08-30 by scripts/probe_nfl_odds.py: bet365 IS a
                # visible bookmaker (id 4) and all four markets exist as bet types,
                # but asking for the week-1 opener's odds BY GAME ID returned zero
                # records from bet365 and from every book. The previous "no odds"
                # conclusion had been reached with a broken query -- it filtered on
                # a `date` parameter the endpoint does not have -- so this is the
                # first properly established answer.
                # Established 2026-09-13 from the live response: bet365 quotes
                # rushing yards, passing yards and anytime TD through API-NFL, and
                # no book there quotes receiving yards.
                "note": ("Rushing, passing and anytime-TD picks use bet365's own "
                         "line and price where bet365 quotes the player. No book "
                         "quotes receiving yards through API-NFL, so that market "
                         "and any unquoted player use the model's own line."
                         if quotes else
                         "No bookmaker prop lines were captured on the last sync, "
                         "so every prop uses the model's own line."),
            },
        },
        "injury_report": {
            "players_listed": len(injuries),
            "source": "API-NFL" if injuries else None,
        },
        "caveats": [
            "No depth charts. The model does not know who starts, so a backup "
            "with a low line can top a market he may not play in.",
            "Injury data covers players API-NFL reports on. Absence from that "
            "report means not reported, which is not the same as confirmed fit.",
            "Clubs are reconciled against the current API-NFL rosters where those "
            "are complete; where a roster came back thin the club falls back to "
            "the player's last appearance and the card says so.",
            "Yardage probabilities are pulled toward the book's de-vigged price: "
            "only a fitted share of the model's disagreement with the book is kept "
            "(nfl/market_blend.py, re-fitted each run on the graded record -- 10% as "
            "of 27 Sep 2026), because the live record showed the raw model "
            "overconfident. Each card still shows the raw model's number.",
            "Every yardage pick stands on a bookmaker's line: bet365 first, then "
            "Pinnacle, DraftKings or FanDuel for players bet365 does not quote. A "
            "player no book quotes yet is held back, not given a made-up line. "
            "The edge is against that book's de-vigged price.",
        ],
        "markets": {
            "anytime_touchdown": "Anytime touchdown",
            "receiving_yards": "Receiving yards",
            "rushing_yards": "Rushing yards",
            "passing_yards": "Passing yards",
        },
    }


def main():
    payload = build()
    # FREEZE AND GRADE BEFORE WRITING. Done here rather than inside build() so the
    # locked pick is what reaches disk: the board must display the pick the record
    # grades, not a fresher one computed moments earlier.
    payload["record"] = picks.freeze_and_grade(payload)
    # The picks BEHIND the record, so the Grades tab can show which hit and
    # which missed rather than only a total.
    payload["settled"] = picks.settled(picks.core.load_log(picks.PICKS_LOG))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "board.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json_safe.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    counts = {m: len(v["picks"]) for m, v in payload["props"].items()}
    print(f"season {payload['season']} week {payload['week']}: "
          f"{len(payload['games'])} games, props {counts}")
    rec = payload["record"]
    print(f"  team winner: {rec['team_winner']['correct']}-"
          f"{rec['team_winner']['wrong']} settled, "
          f"{rec['team_winner']['pending']} pending, "
          f"{rec['team_winner']['void']} void")
    print(f"  props:       {rec['props']['correct']}-{rec['props']['wrong']} "
          f"settled, {rec['props']['pending']} pending, "
          f"{rec['props']['void']} void")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
