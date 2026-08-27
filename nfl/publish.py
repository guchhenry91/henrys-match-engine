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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl import config, data, features, games_model, rosters
from nfl.model import PropModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nfl"
REPORT = ROOT / "data-raw" / "nfl" / "backtest_report.json"

TOP_PER_MARKET = 3          # a shortlist, not a database
MIN_PROBABILITY = 0.50      # never publish a leg the model itself makes a dog


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
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw.get("players") or {}


def upcoming_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """The next slate: the earliest unplayed week."""
    future = schedule[~schedule["played"]].copy()
    if future.empty:
        return future
    first = future.sort_values(["season", "week"]).iloc[0]
    future = future[(future["season"] == first["season"])
                    & (future["week"] == first["week"])]
    return future.sort_values("gameday")


def player_projections(player_weeks, games, market, upcoming, injuries=None,
                       roster_index=None, rosters_complete=False) -> list:
    """Project every eligible player in the upcoming slate for one market."""
    injuries = injuries or {}
    roster_index = roster_index or {}
    frame = features.build(player_weeks, market, games=games)
    if frame.empty:
        return []
    model = PropModel(market).fit(frame)

    latest = frame.sort_values(["season", "week"]).groupby("player_id").tail(1).copy()

    # STILL PLAYING. A player's club is taken from his last appearance, so without
    # a recency test the week 1 board fills with men who retired years ago --
    # Alfred Blue projected onto Houston, C.J. Anderson onto Detroit, Colt McCoy
    # onto Arizona, each carried forward from a final season half a decade back.
    # Appearing in the most recent completed season is the weakest test that
    # excludes them, and it is deliberately weak: it will still wrongly attribute
    # a player who moved this offseason, which the card says out loud rather than
    # pretending otherwise.
    newest = int(frame["season"].max())
    latest = latest[latest["season"] >= newest - (config.ACTIVE_WITHIN_SEASONS - 1)]
    if latest.empty:
        return []

    fixtures = {}
    for _, game in upcoming.iterrows():
        fixtures[game["home_team"]] = (game, game["away_team"], True)
        fixtures[game["away_team"]] = (game, game["home_team"], False)

    # RECONCILE THE CLUB BEFORE choosing who is playing. nflverse says where a man
    # last PLAYED; the roster snapshot says where he IS, and through an offseason
    # those differ. Doing this after the fixture lookup would project a moved
    # player onto his OLD team's game.
    if roster_index:
        resolved, reasons = [], []
        for _, row in latest.iterrows():
            team, why = rosters.reconcile(row["player_display_name"], row["team"],
                                          roster_index, rosters_complete)
            resolved.append(team)
            reasons.append(why)
        latest = latest.assign(_team=resolved, _why=reasons)
        latest = latest[latest["_team"].notna()].copy()
        latest["team"] = latest["_team"]
    else:
        latest = latest.assign(_why="no roster snapshot; using last appearance")

    playing = latest[latest["team"].isin(fixtures)]
    if playing.empty:
        return []

    rows = []
    for (_, player), prob in zip(playing.iterrows(), model.predict(playing)):
        game, opponent, is_home = fixtures[player["team"]]
        name = player["player_display_name"]

        # RULED OUT MEANS OFF THE BOARD. His last five games look exactly as good
        # as anyone's right up until he is inactive, which is precisely why a
        # projection for a player who will not dress is the most misleading thing
        # this board could print.
        report = injuries.get(name) or {}
        if report.get("status") == "out":
            continue

        last_five = [float(v) for v in (player["last_five"] or [])]
        rows.append({
            "market": market,
            "player": name,
            "player_id": player["player_id"],
            "team": player["team"],
            "opponent": opponent,
            "home": bool(is_home),
            "kickoff": pd.Timestamp(game["gameday"]).isoformat(),
            "line": None if pd.isna(player["line"]) else float(player["line"]),
            "probability": round(float(prob), 4),
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
            "games_played": int(player["games_before"]),
            "as_of_season": int(player["season"]),
            "as_of_week": int(player["week"]),
        })
    rows.sort(key=lambda r: -r["probability"])
    return rows


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
    snapshot = rosters.load()
    roster_index, rosters_complete = rosters.index(snapshot)
    # A snapshot may only overrule nflverse if it demonstrably describes the same
    # league. Measured against the players we independently know are active --
    # see rosters.corroborates for what happened without this.
    known_active = sorted(set(
        player_weeks[player_weeks["season"] == player_weeks["season"].max()]
        ["player_display_name"]))
    trusted, agreement = rosters.corroborates(roster_index, known_active)
    if not trusted:
        print(f"WARNING: roster snapshot recognises only {agreement:.0%} of known "
              f"active players; NOT trusting it to drop or move anyone")
        roster_index, rosters_complete = {}, False

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
            "season": int(game["season"]), "week": int(game["week"]),
            "kickoff": pd.Timestamp(game["gameday"]).isoformat(),
            "home": game["home_team"], "away": game["away_team"],
            "neutral": str(game.get("location", "Home")).lower() == "neutral",
            "p_home": round(prob_home, 4),
            "pick": game["home_team"] if pick_home else game["away_team"],
            "p_pick": round(prob_home if pick_home else 1 - prob_home, 4),
            "rating_home": round(float(priced.iloc[0]["rating_home"]), 1),
            "rating_away": round(float(priced.iloc[0]["rating_away"]), 1),
            "gradeable": "team_winner" in released,
        })

    props = {}
    for market in config.MARKETS:
        if market not in released:
            props[market] = {"released": False, "picks": []}
            continue
        projections = player_projections(player_weeks, history, market, upcoming,
                                         injuries=injuries,
                                         roster_index=roster_index,
                                         rosters_complete=rosters_complete)
        shortlist = [p for p in projections if p["probability"] >= MIN_PROBABILITY]
        by_game = {}
        for pick in shortlist:
            game_key = pick["team"] + "|" + pick["opponent"]
            by_game.setdefault(game_key, []).append(pick)
        trimmed = []
        for picks_for_game in by_game.values():
            trimmed.extend(picks_for_game[:TOP_PER_MARKET])
        trimmed.sort(key=lambda p: -p["probability"])
        props[market] = {"released": True, "picks": trimmed}

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
        "roster_check": {
            "teams": len((snapshot.get("teams") or {})),
            "complete": sum(1 for e in (snapshot.get("teams") or {}).values()
                            if e.get("complete")),
            "all_complete": rosters_complete,
            "agreement_with_known_players": round(agreement, 3),
            "trusted": trusted,
            "updated": snapshot.get("updated"),
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
            "Lines are each player's own entering median, not a sportsbook price. "
            "'Over' means a better day than his typical one.",
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
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "board.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    counts = {m: len(v["picks"]) for m, v in payload["props"].items()}
    print(f"season {payload['season']} week {payload['week']}: "
          f"{len(payload['games'])} games, props {counts}")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
