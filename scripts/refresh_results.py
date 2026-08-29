"""Update RESULTS on the published boards without refitting anything.

WHY THIS EXISTS. Grading a played fixture and redrawing the table needs no model:
the pick was frozen days ago and the score is a fact. But both only happened
inside `leagues.publish`, which refits four league models and takes about eighteen
minutes on a matchday. So the visible results could only ever be as fresh as the
last full refresh, and GitHub is currently starting 4-8 scheduled runs a day
against the 56 the crons ask for. On 2026-08-29 that left the boards four hours
stale with roughly fifteen finished games missing -- reported, correctly, as
"games that have been played still missing".

Worse, 21% of scheduled runs fail outright. When the heavy job dies, or stands
down because a fresher run landed, NOTHING updates. Results should not be hostage
to a model refit that has no bearing on them.

So this is the fast path. It reads the boards that were already published, applies
the scores, and writes them back:

  * grades frozen picks against the results feed,
  * redraws the real standings,
  * recomputes `unrecorded` (played fixtures that froze no pick),
  * drops finished fixtures from the upcoming list,

and it touches NOTHING ELSE. Predictions, props, parlays and the projected table
are left exactly as the model last computed them, because a finished match does
not change what the model thinks about the next one. It runs in seconds, on every
trigger, ahead of and independently of the refit.

IT CANNOT INVENT A PICK. Grading reads the frozen log; a fixture with no entry
stays out of the record and is reported under `unrecorded`. Nothing here writes a
pick, so the "grade what was frozen, never a hindsight re-pick" rule is intact.
"""
import json
import sys
from pathlib import Path

import pandas as pd

# NOT leagues.publish: that would drag penaltyblog, scipy, sklearn and
# soccerdata into a job that installs pandas only. standings.py holds the
# two pure functions this needs.
from leagues import config, fixtures, picks, standings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "leagues"
PICKS_DIR = ROOT / "data-raw" / "leagues"

FILES = {"PL": "pl", "LALIGA": "laliga", "BUNDESLIGA": "bundesliga",
         "LIGUE1": "ligue1"}


def _season_tag(league: str) -> str:
    """The same namespace publish and the locker use.

    fixturedownload restarts MatchNumbers at 1 every season while the log
    persists, so without this next season's fixture #1 inherits -- and is graded
    against -- this season's pick. Derived the same way in all three places
    rather than hardcoded, so they cannot drift apart.
    """
    return config.get(league).fixture_slug.rsplit("-", 1)[-1]


def refresh_league(league: str, now=None) -> dict:
    """Apply results to one published board. Returns a summary of what changed."""
    slug = FILES[league]
    path = OUT / f"{slug}.json"
    if not path.exists():
        return {"league": league, "skipped": "no published board"}
    board = json.loads(path.read_text(encoding="utf-8"))

    fx = fixtures.fetch_fixtures(league)
    played = fx[fx["played"]].copy()
    if played.empty:
        return {"league": league, "skipped": "nothing played"}

    log_path = PICKS_DIR / slug / "picks_log.json"
    log = picks.load_log(log_path)
    tag = _season_tag(league)
    log_key = lambda mid: f"{tag}:{mid}"          # noqa: E731 - matches publish

    # --- grade the frozen picks ---------------------------------------------
    graded, newly = [], 0
    for _, m in played.iterrows():
        key = log_key(m["match_id"])
        entry = log.get(key)
        if not entry:
            continue                      # never frozen -> `unrecorded`, not a loss
        was = entry.get("graded")
        g = picks.grade(entry, {"home": m["home"], "away": m["away"],
                                "home_goals": m["home_goals"],
                                "away_goals": m["away_goals"]})
        log[key].update({"graded": g["graded"], "void": g["void"]})
        if was != g["graded"]:
            newly += 1
        graded.append(log[key])
    picks.save_log(log, log_path)

    # Every frozen pick, not only the played ones, so pending stays visible.
    everything = [v for k, v in log.items() if not k.startswith("_")]

    # --- redraw what the scores changed --------------------------------------
    teams = [r["team"] for r in board.get("table", [])] or \
            sorted(set(fx["home"]) | set(fx["away"]))
    table = standings.actual_standings(played, teams)

    # NEVER REGRESS. When fixturedownload times out, fetch_fixtures silently falls
    # back to a snapshot -- 5 hours old in the run that prompted this guard -- and
    # returns it with NO flag saying so. The publish path has sanity_check to catch
    # that; this fast path had nothing, and would cheerfully overwrite a 14-match
    # table with the 12-match one a stale snapshot describes.
    #
    # A season's played count only ever goes UP, so a decrease is proof the input
    # is stale rather than news. Refuse the league outright and leave the board as
    # it was; the next run with a working feed carries it forward.
    was = sum(r.get("played", 0) for r in board.get("standings") or []) // 2
    fresh = sum(r["played"] for r in table) // 2
    if fresh < was:
        return {"league": league,
                "skipped": (f"feed reports {fresh} matches played, board already "
                            f"has {was} -- stale snapshot, refusing to regress")}

    board["record"] = picks.record(everything)
    board["standings"] = table
    board["unrecorded"] = standings.unrecorded_fixtures(played, log, log_key)

    # A finished fixture must stop being advertised as upcoming.
    done = {str(m["match_id"]) for _, m in played.iterrows()}
    before = len(board.get("matches") or [])
    board["matches"] = [m for m in (board.get("matches") or [])
                        if str(m.get("id")) not in done]

    board["results_updated"] = pd.Timestamp.now("UTC").isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return {"league": league, "played": int(len(played)), "newly_graded": newly,
            "record": f"{board['record']['correct']}-{board['record']['wrong']}",
            "unrecorded": len(board["unrecorded"]),
            "fixtures_dropped": before - len(board["matches"])}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    leagues = [a.upper() for a in argv] or list(FILES)
    changed = 0
    for league in leagues:
        if league not in FILES:
            print(f"skip {league!r}: unknown league")
            continue
        try:
            info = refresh_league(league)
        except Exception as exc:
            # One league's feed failing must not stop the other three updating.
            print(f"  {league}: FAILED {type(exc).__name__}: {exc}")
            continue
        if info.get("skipped"):
            print(f"  {league}: {info['skipped']}")
            continue
        changed += info["newly_graded"]
        print(f"  {league}: {info['played']} played, {info['newly_graded']} newly "
              f"graded, record {info['record']}, {info['unrecorded']} unrecorded, "
              f"{info['fixtures_dropped']} fixture(s) off the upcoming list")
    print(f"{changed} pick(s) newly graded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
