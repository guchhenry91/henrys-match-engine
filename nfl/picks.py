"""Freeze and grade the NFL board's picks.

The NFL board published team-winner picks and four prop markets with **no picks
log, no freeze and no record** -- the last board in the repo without one. Nothing
stopped a displayed pick moving after kickoff, and no number on the page could be
checked against what actually happened.

It reuses `leagues.picks` for the freezing itself, exactly as the Champions League
board does. Rules 1-3 (lock before kickoff, grade the FROZEN pick, void a pick
first locked after kickoff) are not sport-specific, and a third copy of them would
be a third place for the record to drift.

GRADING IS NOT SHARED, because the markets are not. `leagues.picks.grade_prop`
settles soccer thresholds (1+ goal, 2+ shots); NFL settles a touchdown event and
three over/under lines. Those definitions are mirrored EXACTLY from
`nfl/features.py` -- `touchdowns > 0` and `yards > line` -- because the release
gate measured the board against that definition, and a record grading anything
else would be reporting on a product the gate never validated.
"""
from pathlib import Path
from urllib.error import HTTPError, URLError

import pandas as pd

from leagues import picks as core
# Same window the soccer and UCL boards use, set from the MEASURED worst gap
# between locker runs (see leagues/config.py). A second constant here would be a
# second thing to forget to update.
from leagues import lockwindow
from leagues.config import LOCK_WINDOW_HOURS
from nfl import config, data

ROOT = Path(__file__).resolve().parent.parent
PICKS_LOG = ROOT / "data-raw" / "nfl" / "picks_log.json"

GAMES_KEY, PROPS_KEY = "games", "props"


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _confidence(p: float) -> int:
    if p >= 0.70:
        return 5
    if p >= 0.60:
        return 4
    if p >= 0.50:
        return 3
    if p >= 0.40:
        return 2
    return 1


def prop_key(pick: dict) -> str:
    """One prop per player per market per game."""
    return f"{pick.get('game_id')}:{pick['market']}:{pick.get('player_id')}"


def season_week(game_id):
    """`2026_01_NE_SEA` -> (2026, 1). The id encodes both, so no second lookup."""
    try:
        parts = str(game_id).split("_")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError, TypeError):
        return None, None


def grade_game(entry: dict, game: dict | None) -> dict:
    """Settle a frozen team-winner pick.

    A TIE IS VOID, not a loss. The gate scores a tie as 0.5 -- half a win to each
    side, which is what it is -- and a win/loss record has no half. Calling it a
    loss for whichever team was named would understate the model against its own
    measured performance; calling it a win would flatter it. A moneyline pushes on
    a tie for the same reason.
    """
    out = dict(entry)
    if entry.get("tainted"):
        out.update(void=True, graded="void", void_reason="locked after kickoff")
        return out
    if not game:
        return out                        # unplayed, or not yet in the feed
    winner = game.get("winner")
    if winner is None or (isinstance(winner, float) and pd.isna(winner)):
        return out
    if winner == "tie":
        out.update(void=True, graded="void", void_reason="tie")
        return out
    won = game["home"] if winner == "home" else game["away"]
    out["void"] = False
    out["graded"] = "correct" if entry["pick"] == won else "wrong"
    out["result"] = (f"{game['home']} {game['home_score']:.0f}-"
                     f"{game['away_score']:.0f} {game['away']}")
    return out


def grade_prop(entry: dict, actual=None) -> dict:
    """Settle a frozen prop against the player's actual line for that game.

    `actual` is his stats_player_week row, or None when the feed holds the game
    but no row for him.

    None is graded WRONG, not void, and for the NFL both readings agree: a man who
    did not play scored no touchdown and gained no yards, so the harsh reading and
    the literal one settle him under alike. What matters is that this is reached
    ONLY for a game the feed actually covers -- see `covered_games`.
    """
    out = dict(entry)
    if entry.get("tainted"):
        out.update(void=True, graded="void", void_reason="locked after kickoff")
        return out

    stat = config.MARKETS[entry["market"]]
    if actual is None:
        value = 0.0
    else:
        raw = actual.get(stat) if isinstance(actual, dict) else actual[stat]
        value = 0.0 if raw is None or pd.isna(raw) else float(raw)
    line = entry.get("line")
    # Mirrored from nfl/features.py. Lines are quoted on the half yard, so no
    # result can land exactly on one and there is no push to handle.
    hit = value > 0 if line is None else value > float(line)
    out["void"] = False
    out["actual"] = value
    out["graded"] = "correct" if hit else "wrong"
    return out


def covered_games(stats) -> set:
    """(season, week, team) triples the player feed has actually filed.

    THE GUARD THAT MATTERS. On 2026-08-23 the soccer board graded 18 player picks
    0-18 against a feed that had published nothing for the new season: the frame
    was nowhere near empty, so an "is the feed up" check passed, every lookup
    missed, and every miss became a fabricated loss in an append-only record.

    A prop settles only where the feed holds that player's OWN TEAM in that week.
    Where it is silent the pick stays PENDING. Absence of the game is not evidence
    about the player.
    """
    if stats is None or len(stats) == 0:
        return set()
    return set(zip(stats["season"].astype(int), stats["week"].astype(int),
                   stats["team"].astype(str)))


def record(log: dict) -> dict:
    """The published record: games and props separately, never pooled.

    Pooled, a 52% team-winner pick and a 90% touchdown prop average into a number
    describing neither -- the same reason the soccer board grades its markets
    apart.
    """
    def tally(entries):
        counts = {"correct": 0, "wrong": 0, "void": 0, "pending": 0}
        for key, entry in entries.items():
            if key.startswith("_"):
                continue                  # _released archive, never a pick
            counts[entry.get("graded") or "pending"] += 1
        settled = counts["correct"] + counts["wrong"]
        return {**counts, "total": sum(counts.values()), "settled": settled,
                "hit_rate": round(counts["correct"] / settled, 4) if settled else None}

    props = log.get(PROPS_KEY, {})
    by_market = {}
    for market in config.MARKETS:
        by_market[market] = tally({k: v for k, v in props.items()
                                   if not k.startswith("_")
                                   and v.get("market") == market})
    return {"team_winner": tally(log.get(GAMES_KEY, {})),
            "props": tally(props),
            "props_by_market": by_market}


def settled(log: dict) -> list:
    """Every graded pick, newest first, flattened for display.

    WHY THIS EXISTS AS A LIST and not just the counts in `record()`. A record that
    reports 12-7 asks to be trusted; a list that names each pick and what actually
    happened can be checked. The soccer boards have published their settled picks
    from the start and the NFL published only totals, so the moment week 1 is
    played the Grades tab would have shown NFL aggregates with nothing behind
    them.

    Read-only: it derives from the log and never grades, locks or writes anything.
    """
    out = []
    for section, kind in ((GAMES_KEY, "game"), (PROPS_KEY, "prop")):
        for key, entry in (log.get(section) or {}).items():
            if key.startswith("_") or not entry.get("graded"):
                continue
            row = {"kind": kind, "key": key,
                   "graded": entry.get("graded"),
                   "void": bool(entry.get("void")),
                   "p_pick": entry.get("p_pick"),
                   "confidence": entry.get("confidence"),
                   "kickoff": entry.get("kickoff"),
                   "date": entry.get("kickoff"),
                   "home": entry.get("home"), "away": entry.get("away"),
                   "tainted": bool(entry.get("tainted"))}
            if kind == "game":
                row.update(pick=entry.get("pick"), result=entry.get("result"))
            else:
                row.update(market=entry.get("market"), player=entry.get("player"),
                           team=entry.get("team"), line=entry.get("line"),
                           actual=entry.get("actual"))
            out.append(row)
    out.sort(key=lambda r: str(r.get("kickoff") or ""), reverse=True)
    return out


def grading_stats(season):
    """The player feed for a season, or an EMPTY frame if it does not exist yet.

    nflverse publishes `stats_player_week_{season}.csv` only once a season has
    started, so asking for it before week 1 is a 404. That is not an error worth
    failing the publish over -- there is simply nothing to grade -- but it must
    degrade to an EMPTY frame rather than to something that looks like coverage.
    An empty frame means `covered_games` is empty, which means every prop stays
    PENDING. That is the honest outcome, and it is the same shape as the guard
    that stops a silent feed manufacturing losses.
    """
    try:
        return data.player_weeks([season])
    except (HTTPError, URLError) as exc:
        print(f"  no player feed for {season} yet ({exc}); props stay pending")
        return pd.DataFrame(columns=["player_id", "season", "week", "team"])


def _stats_lookup(stats) -> dict:
    if stats is None or len(stats) == 0:
        return {}
    return {(str(row["player_id"]), int(row["season"]), int(row["week"])): row
            for _, row in stats.iterrows()}


def _lock_games(payload, log, now):
    for game in payload.get("games", []):
        key, kickoff = game.get("game_id"), game.get("kickoff")
        if not key or not kickoff or not game.get("gradeable"):
            # No id, no real kickoff, or a market the gate withheld. Freezing a
            # pick against a guessed kickoff is what froze Alaves 10.5 hours early
            # and voided Sevilla out of the record on 2026-08-15.
            game["lockable"] = False
            continue
        game["lockable"] = True
        kickoff = _utc(kickoff)
        core.release_moved_lock(log, key, kickoff, now=now)
        hours_out = (kickoff - now).total_seconds() / 3600.0
        if key not in log and hours_out <= lockwindow.window(now):
            entry = core.lock_pick(log, key, game["pick"],
                                   _confidence(game["p_pick"]), kickoff, now=now,
                                   p_pick=game["p_pick"], board=True)
            # Stored so the settled record can name the fixture later. The log is
            # keyed by game_id and grading writes only a score STRING, so without
            # these a graded entry cannot say who played whom.
            entry["home"], entry["away"] = game.get("home"), game.get("away")
        entry = log.get(key)
        if entry:
            # Show what was frozen, not what the model would say now.
            game["pick"] = entry["pick"]
            game["p_pick"] = entry.get("p_pick", game["p_pick"])
            game["locked"] = True
            game["tainted"] = bool(entry.get("tainted"))


def _lock_props(payload, log, now):
    for market, block in (payload.get("props") or {}).items():
        if not block.get("released"):
            continue                      # withheld market: nothing to freeze
        for pick in block.get("picks", []):
            key, kickoff = prop_key(pick), pick.get("kickoff")
            if not pick.get("game_id") or not kickoff or not pick.get("player_id"):
                pick["lockable"] = False
                continue
            pick["lockable"] = True
            kickoff = _utc(kickoff)
            core.release_moved_lock(log, key, kickoff, now=now)
            hours_out = (kickoff - now).total_seconds() / 3600.0
            if key not in log and hours_out <= lockwindow.window(now):
                entry = core.lock_prop(
                    log, key, market, pick["player"], pick["team"],
                    pick["probability"], _confidence(pick["probability"]), kickoff,
                    now=now, doubt=(pick.get("availability") == "doubt"),
                    unavailable=(pick.get("availability") == "out"),
                    team_attribution=pick.get("club_source"))
                # NFL-specific, needed to settle it later and to prove what the
                # pick was made against. THE LINE ESPECIALLY: it is his own
                # entering median and moves week to week, so grading against a
                # later line would settle the bet nobody made.
                entry["line"] = pick.get("line")
                entry["player_id"] = pick.get("player_id")
                entry["game_id"] = pick.get("game_id")
                # Same reason as the games: the settled row has to name the
                # fixture, and the prop key does not carry it.
                entry["home"] = pick.get("home") or pick.get("team")
                entry["away"] = pick.get("opponent")
            entry = log.get(key)
            if entry:
                pick["probability"] = entry.get("p_pick", pick["probability"])
                pick["locked"] = True
                pick["tainted"] = bool(entry.get("tainted"))


def freeze_and_grade(payload: dict, now=None, stats=None, results=None) -> dict:
    """Freeze what is due, grade what has been played, return the record.

    Mutates `payload` so the board DISPLAYS the frozen pick. A board showing a
    freshly computed pick beside a record grading a different one has quietly
    stopped describing itself.
    """
    now = _utc(now if now is not None else pd.Timestamp.now("UTC"))
    season = int(payload.get("season") or config.SEASONS[-1])
    log = core.load_log(PICKS_LOG)
    log.setdefault(GAMES_KEY, {})
    log.setdefault(PROPS_KEY, {})

    if results is None:
        frame = data.games([season])
        results = {row["game_id"]: {"home": row["home_team"], "away": row["away_team"],
                                    "home_score": row["home_score"],
                                    "away_score": row["away_score"],
                                    "winner": row["winner"]}
                   for _, row in frame.iterrows() if row["played"]}
    if stats is None:
        stats = grading_stats(season)
    covered = covered_games(stats)

    _lock_games(payload, log[GAMES_KEY], now)
    _lock_props(payload, log[PROPS_KEY], now)

    # GRADING SWEEPS THE LOG, NOT THE BOARD. The board publishes only UPCOMING
    # games, so a fixture leaves it the moment it is played. Grading driven off the
    # board would skip every pick on the one run that could have settled it, and
    # the record would read 0-0 forever.
    for key, entry in list(log[GAMES_KEY].items()):
        if key.startswith("_") or entry.get("graded") is not None:
            continue
        log[GAMES_KEY][key] = grade_game(entry, results.get(key))

    lookup = _stats_lookup(stats)
    for key, entry in list(log[PROPS_KEY].items()):
        if key.startswith("_") or entry.get("graded") is not None:
            continue
        if not results.get(entry.get("game_id")):
            continue                      # not played, or not yet in the feed
        season_, week = season_week(entry.get("game_id"))
        if (season_, week, entry.get("team")) not in covered:
            continue                      # feed silent for his side: stay PENDING
        log[PROPS_KEY][key] = grade_prop(
            entry, lookup.get((str(entry.get("player_id")), season_, week)))

    core.save_log(log, PICKS_LOG)
    return record(log)
