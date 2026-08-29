"""Freeze picks that are about to kick off, fast, from the LAST PUBLISHED board.

WHY THIS EXISTS. Locking used to happen only inside `leagues.publish`, which
refits four league models and takes about ten minutes to reach the lock step. Add
GitHub's own scheduling delay -- a median of 8 minutes after the cron slot,
measured over 109 runs -- and a nominal 18:45 run freezes a pick at roughly
19:05. For a 19:00 kickoff that is a late lock, which taints the pick, which
voids it. Four La Liga fixtures, eight player picks and thirty-seven parlays went
that way in a fortnight, and every one of them was a perfectly good pick.

THE INSIGHT IS THAT LOCKING NEEDS NO MODEL. The probability was already computed
and published; freezing it is a file read, a comparison and a file write. This
runs in seconds, so it can be scheduled often enough that something lands inside
the window even when a third of scheduled runs fail.

IT ONLY EVER FREEZES WHAT WAS ALREADY PUBLISHED. It computes nothing, so it can
never introduce a number the board did not show. A pick frozen here is the pick a
reader could see at that moment, which is the whole claim the record makes.

Idempotent by construction: picks.lock_pick and picks.lock_prop no-op on a second
call for the same key, so running this every five minutes costs nothing and can
never overwrite an earlier, better-timed lock.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from leagues import config, parlays, picks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "leagues"
PICKS_DIR = ROOT / "data-raw" / "leagues"
LOCK_WINDOW_HOURS = config.LOCK_WINDOW_HOURS

FILES = {"PL": "pl", "LALIGA": "laliga", "BUNDESLIGA": "bundesliga", "LIGUE1": "ligue1"}


def _season_tag(league: str) -> str:
    """The same namespace publish uses. fixturedownload restarts MatchNumbers at 1
    every season while the log persists, so without this next season's fixture #1
    inherits -- and is graded against -- this season's pick."""
    return config.get(league).fixture_slug.rsplit("-", 1)[-1]


def _hours_out(date_text, now) -> float:
    return (picks._utc(pd.Timestamp(date_text)) - now).total_seconds() / 3600.0


def lock_matches(now) -> list:
    """Freeze the match-winner pick for anything inside the window."""
    frozen = []
    for league, stem in FILES.items():
        payload_path = OUT / f"{stem}.json"
        log_path = PICKS_DIR / league.lower() / "picks_log.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        log = picks.load_log(log_path)
        tag = _season_tag(league)
        changed = False
        for match in payload.get("matches", []):
            prediction = match.get("prediction") or {}
            if not prediction.get("pick") or prediction.get("p_pick") is None:
                continue
            # The same guard publish applies: never freeze against a kickoff time
            # the feed does not believe.
            if match.get("time_suspect"):
                continue
            hours = _hours_out(match["date"], now)
            if hours <= 0 or hours > LOCK_WINDOW_HOURS:
                continue
            key = f"{tag}:{match['id']}"
            if key in log:
                continue
            picks.lock_pick(log, key, pick=prediction["pick"],
                            confidence=int(prediction.get("confidence") or 0),
                            kickoff=match["date"], now=now,
                            p_pick=prediction["p_pick"],
                            board=bool(prediction.get("best_pick")))
            frozen.append(f"{league} {match['home']} v {match['away']} "
                          f"({hours * 60:.0f}m out)")
            changed = True
        if changed:
            picks.save_log(log, log_path)
    return frozen


def lock_players(now) -> list:
    """Freeze player picks for anything inside the window."""
    path = OUT / "player_picks.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen, logs = [], {}
    for pick in payload.get("upcoming", []):
        league = pick.get("league_key")
        if not league or pick.get("p_pick") is None:
            continue
        hours = _hours_out(pick["date"], now)
        if hours <= 0 or hours > LOCK_WINDOW_HOURS:
            continue
        log_path = PICKS_DIR / league.lower() / "player_picks_log.json"
        if league not in logs:
            logs[league] = (log_path, picks.load_log(log_path))
        _, log = logs[league]
        key = f"{_season_tag(league)}:{pick['id']}:{pick['market']}:{pick['player']}"
        if key in log:
            continue
        picks.lock_prop(log, key, market=pick["market"], player=pick["player"],
                        team=pick["team"], p_pick=pick["p_pick"],
                        confidence=int(pick.get("confidence") or 0),
                        kickoff=pick["date"], now=now,
                        bar=pick.get("bar"),
                        lineup_confirmed=pick.get("lineup_confirmed"),
                        appearance_pct=pick.get("appearance_pct"),
                        expected_minutes=pick.get("expected_minutes"),
                        news_checked_hours_ago=pick.get("news_checked_hours_ago"),
                        doubt=pick.get("doubt"),
                        unavailable=pick.get("unavailable"),
                        team_attribution=pick.get("team_attribution"))
        frozen.append(f"{pick['player']} {pick['market']} ({hours * 60:.0f}m out)")
    for log_path, log in logs.values():
        picks.save_log(log, log_path)
    return frozen


def lock_parlays(now) -> list:
    """Freeze a parlay before its EARLIEST leg kicks off.

    Rebuilt from the published boards rather than recomputed from the model --
    build_parlays is pure arithmetic over probabilities that already exist, so it
    costs nothing and cannot invent a number the board did not show.
    """
    best_path, players_path = OUT / "best.json", OUT / "player_picks.json"
    if not (best_path.exists() and players_path.exists()):
        return []
    before = picks.load_log(PICKS_DIR / "parlays_log.json")
    count_before = len([k for k in before if not str(k).startswith("_")])
    parlays.build_parlays(json.loads(best_path.read_text(encoding="utf-8")),
                          json.loads(players_path.read_text(encoding="utf-8")),
                          PICKS_DIR / "parlays_log.json", now=now)
    after = picks.load_log(PICKS_DIR / "parlays_log.json")
    count_after = len([k for k in after if not str(k).startswith("_")])
    gained = count_after - count_before
    return [f"{gained} parlay(s)"] if gained > 0 else []


def lock_nfl(now) -> list:
    """Freeze the NFL board's picks from the board it already published.

    THE NFL BOARD CANNOT LOCK ITSELF IN TIME. Its own workflow publishes at 09:00
    and 16:00 UTC; the Sunday slate kicks off at 17:00, 20:05 and 00:20 UTC. Only
    the first of those ever falls inside the lock window from a publish run, so
    the late games -- Sunday night football included -- would reach kickoff
    unfrozen and then be frozen late by the next morning's run, which taints them,
    which VOIDS them. Every late-window game would be lost from the record, which
    is precisely the La Liga failure that created this script.

    Uses nfl.picks in LOCK-ONLY mode: no results, no player feed, no network. It
    freezes what the board already showed and grades nothing.
    """
    board_path = ROOT / "data" / "nfl" / "board.json"
    if not board_path.exists():
        return []
    payload = json.loads(board_path.read_text(encoding="utf-8"))
    before = _nfl_locked_count()
    # Import here, not at module scope: this script installs pandas only, and a
    # missing NFL dependency must never take the soccer locking down with it.
    from nfl import picks as nfl_picks
    log = picks.load_log(nfl_picks.PICKS_LOG)
    log.setdefault("games", {})
    log.setdefault("props", {})
    nfl_picks._lock_games(payload, log["games"], now)
    nfl_picks._lock_props(payload, log["props"], now)
    picks.save_log(log, nfl_picks.PICKS_LOG)
    after = _nfl_locked_count()
    return [f"NFL {after - before} pick(s)"] if after > before else []


def _nfl_locked_count() -> int:
    from nfl import picks as nfl_picks
    log = picks.load_log(nfl_picks.PICKS_LOG)
    return sum(len([k for k in log.get(section, {}) if not k.startswith("_")])
               for section in ("games", "props"))


def main():
    now = picks._utc(pd.Timestamp.now("UTC"))
    matches = lock_matches(now)
    players = lock_players(now)
    accas = lock_parlays(now)
    try:
        nfl = lock_nfl(now)
    except Exception as exc:                 # never let NFL sink soccer locking
        print(f"  NFL locking skipped: {exc}")
        nfl = []
    if not (matches or players or accas or nfl):
        print(f"nothing inside the {LOCK_WINDOW_HOURS:g}h lock window; no changes")
        return 0
    for line in matches:
        print(f"  froze match  {line}")
    for line in players:
        print(f"  froze player {line}")
    for line in accas:
        print(f"  froze {line}")
    for line in nfl:
        print(f"  froze {line}")
    print(f"locked {len(matches)} match, {len(players)} player pick(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
