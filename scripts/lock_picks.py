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
from tracking import manifest, release
from pathlib import Path

import pandas as pd

from leagues import config, lockstamp, lockwindow, odds, parlays, picks, prop_odds, shot_odds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "leagues"
PICKS_DIR = ROOT / "data-raw" / "leagues"
# The FLOOR. The window actually applied is lockwindow.window(), which widens to
# cover the gap since the last locking run so a fixture cannot fall between two
# runs and vanish from the record. See leagues/lockwindow.py.
LOCK_WINDOW_HOURS = config.LOCK_WINDOW_HOURS

FILES = {"PL": "pl", "LALIGA": "laliga", "BUNDESLIGA": "bundesliga",
         "LIGUE1": "ligue1", "SERIEA": "seriea"}


def _season_tag(league: str) -> str:
    """The same namespace publish uses. fixturedownload restarts MatchNumbers at 1
    every season while the log persists, so without this next season's fixture #1
    inherits -- and is graded against -- this season's pick."""
    return config.get(league).fixture_slug.rsplit("-", 1)[-1]


def _hours_out(date_text, now) -> float:
    return (picks._utc(pd.Timestamp(date_text)) - now).total_seconds() / 3600.0


def lock_matches(now, window=None) -> list:
    """Freeze the match-winner pick for anything inside the window."""
    frozen = []
    market_odds = {}             # fetched only for a league that actually locks something
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
            if hours <= 0 or hours > (window or LOCK_WINDOW_HOURS):
                continue
            key = f"{tag}:{match['id']}"
            if key in log:
                continue
            entry = picks.lock_pick(log, key, pick=prediction["pick"],
                                    confidence=int(prediction.get("confidence") or 0),
                                    kickoff=match["date"], now=now,
                                    p_pick=prediction["p_pick"],
                                    board=bool(prediction.get("best_pick")))
            # bet365's price and the value verdict, frozen now -- before release.mark
            # decides official/tracked on them (leagues/lockstamp.py).
            if league not in market_odds:
                market_odds[league] = odds.fetch_fixture_odds(league)
            lockstamp.stamp_match(entry, market_odds[league], match["home"], match["away"], now)
            frozen.append(f"{league} {match['home']} v {match['away']} "
                          f"({hours * 60:.0f}m out)")
            changed = True
        if changed:
            manifest.stamp(log.values(), "soccer")      # which model made each new pick
            release.mark(log.values(), "soccer", lambda e: "match_winner", now=now)
            picks.save_log(log, log_path)
    return frozen


XI_WAIT_HOURS = 0.5       # leagues/forecast_board.XI_WAIT_HOURS (kept light: no model import)


def _xi_status(pick: dict, news_cache: dict) -> str | None:
    """"starts", "benched", "absent" from the CONFIRMED XI in news.json (refreshed by
    this job's own lineup step), or None while no XI is confirmed. The board's
    lineup_confirmed flag lags by a full soccer refresh; this reads the XI directly so
    a benched player is never frozen (third review, 2026-10-12)."""
    league = pick.get("league_key")
    if league not in news_cache:
        try:
            from leagues import players
            news_cache[league] = players.load_news(league)
        except Exception:                                  # noqa: BLE001
            news_cache[league] = {}
    entry = news_cache[league].get(pick.get("team")) or {}
    starters = entry.get("starters") or []
    if entry.get("lineup_confirmed") is not True or len(starters) != 11:
        return None
    from nfl.odds import norm_name
    name = norm_name(str(pick.get("player") or ""))
    if name in {norm_name(str(n)) for n in starters}:
        return "starts"
    if name in {norm_name(str(n)) for n in entry.get("bench") or []}:
        return "benched"
    return "absent"


def lock_players(now, window=None) -> list:
    """Freeze player picks for anything inside the window."""
    path = OUT / "player_picks.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen, logs, news_cache = [], {}, {}
    shots_store, prop_store = shot_odds.load(), prop_odds.load()
    for pick in payload.get("upcoming", []):
        league = pick.get("league_key")
        if not league or pick.get("p_pick") is None:
            continue
        if pick.get("time_suspect"):
            continue             # a kickoff we do not trust is never locked against
        if pick.get("awaiting_price") and pick.get("pick_kind") != "proving":
            continue             # no price yet and not a proving pick (tracking/pricecheck.qualifies)
        hours = _hours_out(pick["date"], now)
        if hours <= 0 or hours > (window or LOCK_WINDOW_HOURS):
            continue
        xi = _xi_status(pick, news_cache) if pick.get("engine") else None
        if xi in ("benched", "absent"):
            continue             # not in the confirmed XI: never frozen (a void at best)
        confirmed = bool(pick.get("lineup_confirmed")) or xi == "starts"
        if (pick.get("engine") and not confirmed
                and hours > XI_WAIT_HOURS and (window or LOCK_WINDOW_HOURS) <= LOCK_WINDOW_HOURS):
            continue             # wait for the confirmed XI while runs are frequent
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
                        lineup_confirmed=confirmed or pick.get("lineup_confirmed"),
                        appearance_pct=pick.get("appearance_pct"),
                        expected_minutes=pick.get("expected_minutes"),
                        news_checked_hours_ago=pick.get("news_checked_hours_ago"),
                        doubt=pick.get("doubt"),
                        unavailable=pick.get("unavailable"),
                        team_attribution=pick.get("team_attribution"),
                        p_model=pick.get("p_model"))
        if pick.get("engine"):
            log[key]["engine"] = pick["engine"]       # which model made it
        if pick.get("price_check"):
            log[key]["price_check"] = pick["price_check"]   # the price it was approved at
        if pick.get("pick_kind"):
            log[key]["pick_kind"] = pick["pick_kind"]       # value or proving
        # The price at lock and the value verdict (leagues/lockstamp.py).
        lockstamp.stamp_prop(log[key], league, pick["id"], now,
                             prop_store=prop_store, shots_store=shots_store)
        frozen.append(f"{pick['player']} {pick['market']} ({hours * 60:.0f}m out)")
    for log_path, log in logs.values():
        manifest.stamp(log.values(), "soccer")
        release.mark(log.values(), "soccer", lambda e: e.get("market"), now=now)
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


def lock_ucl(now, window=None) -> list:
    """Freeze the Champions League board's picks from the board it last published.

    The board used to freeze only inside the soccer refresh (ucl.publish), so a slow
    or failed refresh near kickoff left a played match with no pick: seven of the
    eighteen matchday-1 games never entered the record. Same rule as everything
    here: freeze exactly what the board shows, inside the lock window, never after
    kickoff. pandas-only -- ucl.publish's model imports are not needed to freeze."""
    board_path = ROOT / "data" / "ucl" / "board.json"
    if not board_path.exists():
        return []
    payload = json.loads(board_path.read_text(encoding="utf-8"))
    from ucl import publish as ucl_publish
    log = picks.load_log(ucl_publish.PICKS_LOG)
    frozen = []
    for match in payload.get("matches", []):
        if not match.get("lockable", True) or match.get("id") is None or not match.get("kickoff"):
            continue
        if not match.get("pick") or match.get("p_pick") is None or str(match["id"]) in log:
            continue
        hours = _hours_out(match["kickoff"], now)
        if hours <= 0 or hours > (window or LOCK_WINDOW_HOURS):
            continue
        ucl_publish.lock_one(log, match, now)
        frozen.append(f"UCL {match['home']} v {match['away']} ({hours * 60:.0f}m out)")
    if frozen:
        picks.save_log(log, ucl_publish.PICKS_LOG)
    # Champions League player props (ucl/props.py), the same way.
    from ucl import props as ucl_props
    plog, pfrozen = picks.load_log(ucl_props.LOG), []
    for pick in ((payload.get("props") or {}).get("upcoming") or []):
        if ucl_props.key_of(pick) in plog or not pick.get("kickoff"):
            continue
        hours = _hours_out(pick["kickoff"], now)
        if hours <= 0 or hours > (window or LOCK_WINDOW_HOURS):
            continue
        ucl_props.lock_one(plog, pick, now)
        pfrozen.append(f"UCL {pick['player']} {pick['market']} ({hours * 60:.0f}m out)")
    if pfrozen:
        picks.save_log(plog, ucl_props.LOG)
    return frozen + pfrozen


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
    for section in ("games", "props"):
        manifest.stamp([e for k, e in log[section].items() if not k.startswith("_")], "nfl")
        release.mark([e for k, e in log[section].items() if not k.startswith("_")], "nfl",
                     (lambda e: "team_winner") if section == "games" else (lambda e: e.get("market")),
                     now=now)
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
    # Widened to cover the gap since the last locking run. With a healthy
    # scheduler this is just LOCK_WINDOW_HOURS; after a six-hour drought it is
    # what stops four played fixtures leaving no pick at all.
    window = lockwindow.window(now)
    if window > LOCK_WINDOW_HOURS:
        print(f"lock window widened to {window:.1f}h "
              f"(last locking run was that long ago)")
    matches = lock_matches(now, window)
    players = lock_players(now, window)
    accas = lock_parlays(now)
    try:
        matches += lock_ucl(now, window)
    except Exception as exc:                 # never let the UCL sink soccer locking
        print(f"  Champions League locking skipped: {exc}")
    try:
        nfl = lock_nfl(now)
    except Exception as exc:                 # never let NFL sink soccer locking
        print(f"  NFL locking skipped: {exc}")
        nfl = []
    lockwindow.beat(now)          # always, even when nothing froze: the NEXT run
                                  # widens from the last time locking was ATTEMPTED
    if not (matches or players or accas or nfl):
        print(f"nothing inside the {window:.1f}h lock window; no changes")
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
