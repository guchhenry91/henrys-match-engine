"""Grade finished NBA games between NBA runs. No model, no paid API.

WHY (owner, 2026-10-23). NBA picks were graded only inside the NBA board run, at
15:00 and 21:30 UTC, so a night's games (tip-offs 23:00-03:30 UTC) sat ungraded
until the next afternoon -- up to 13 hours after the final buzzer.

The full run learns which games are final from The Odds API (/scores, 2 credits,
because the NBA's CDN refuses Cloudflare). This needs neither: stats.nba.com's
season game log, through the Worker's own stats proxy (free), holds every finished
game's player rows, and the team totals in it say who won. Rows are re-keyed to the
schedule's game id exactly as the full run does (current.sync_from_stats), from a
schedule rebuilt out of the frozen picks themselves.

Rides the 10-minute lock job, like grade_nfl. Downloads nothing until a frozen pick's
game is GAME_LENGTH_HOURS past tip-off and still ungraded, and then at most every
RETRY_MINUTES. Rewrites the board only when the record changed. A game the log does
not hold yet stays pending -- never graded from partial data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "data" / "nba" / "board.json"
GAME_LENGTH_HOURS = 3.5          # a game plus overtime and the log catching up
RETRY_MINUTES = 30
ATTEMPT = ROOT / "data-raw" / "nba" / "_grade_attempt.json"


def due(log: dict, now) -> list:
    """Frozen, ungraded picks whose game should be over by now."""
    out = []
    for key, e in (log or {}).items():
        if str(key).startswith("_") or not isinstance(e, dict) or e.get("graded") is not None:
            continue
        tip = pd.Timestamp(e.get("tipoff") or "NaT")
        if pd.isna(tip):
            continue
        tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
        if (now - tip).total_seconds() / 3600.0 >= GAME_LENGTH_HOURS:
            out.append(key)
    return out


def _recently_tried(now) -> bool:
    try:
        last = pd.Timestamp(json.loads(ATTEMPT.read_text(encoding="utf-8"))["at"])
    except Exception:
        return False
    return pd.Timedelta(0) <= (now - last) < pd.Timedelta(minutes=RETRY_MINUTES)


def schedule_from_log(log: dict, keys) -> pd.DataFrame:
    """The games the waiting picks belong to, schedule-shaped for sync_from_stats."""
    from nba import current
    games = {}
    for k in keys:
        e = log[k]
        gid = str(e.get("game_id"))
        teams = ({e.get("home"), e.get("away")} if e.get("kind") == "winner"
                 else {e.get("team"), e.get("opponent")}) - {None}
        if len(teams) != 2:
            continue
        day = current._eastern_date(e["tipoff"])
        home, away = sorted(teams)
        games[gid] = {"game_id": gid, "game_date": day, "home_team": home, "away_team": away,
                      "stage": current._odds_stage(day), "tipoff": e["tipoff"]}
    return pd.DataFrame(list(games.values()),
                        columns=["game_id", "game_date", "home_team", "away_team", "stage", "tipoff"])


def finals_from_rows(log: dict, rows: pd.DataFrame) -> dict:
    """{game_id: "home"/"away"} for frozen winner picks, from the box score's team points."""
    out = {}
    if rows.empty:
        return out
    pts = rows.groupby([rows["GAME_ID"].astype(str), "TEAM_ABBREVIATION"])["PTS"].sum()
    for e in log.values():
        if not isinstance(e, dict) or e.get("kind") != "winner":
            continue
        gid = str(e.get("game_id"))
        try:
            h, a = float(pts[(gid, e["home"])]), float(pts[(gid, e["away"])])
        except (KeyError, TypeError, ValueError):
            continue
        if h != a:
            out[gid] = "home" if h > a else "away"
    return out


def refresh_board(payload: dict, log: dict) -> None:
    """The record, proving panel, performance and settled list, as nba.publish sets them."""
    from nba import picks as picks_mod
    from tracking import performance, proving
    rebuilt = {k: e for k, e in log.items() if isinstance(e, dict)
               and e.get("engine") == proving.ENGINE and e.get("kind") != "play"}
    payload["record"] = picks_mod.record(log)
    payload["record_rebuilt"] = picks_mod.record(rebuilt)
    payload["record_plays"] = picks_mod.record_plays(log)
    payload["proving"] = proving.summary(rebuilt.values())
    payload["performance"], payload["performance_retired"] = performance.split_by_model(
        [e for e in log.values() if isinstance(e, dict) and e.get("graded")], performance.market_of)
    payload["settled"] = picks_mod.settled(log, limit=100_000)


def main(now=None, fetch=None) -> int:
    from nba import current
    from nba import picks as picks_mod

    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now("UTC")
    if not BOARD.exists():
        print("no NBA board published; nothing to grade")
        return 0
    log = picks_mod.load_log()
    waiting = due(log, now)
    if not waiting:
        print("NBA: no finished game awaiting a grade")
        return 0
    if _recently_tried(now):
        print(f"NBA: {len(waiting)} pick(s) await a grade; log checked under {RETRY_MINUTES} min ago")
        return 0
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPT.write_text(json.dumps({"at": now.isoformat()}) + "\n", encoding="utf-8")
    sched = schedule_from_log(log, waiting)
    for stage in sorted(set(sched["stage"])):
        try:
            added = current.sync_from_stats(sched, stage, fetch=fetch, only=set(sched["game_id"]))
            print(f"  stats.nba.com stage {stage}: {added} new game(s)")
        except Exception as exc:                      # noqa: BLE001
            print(f"  stats.nba.com stage {stage} unavailable ({type(exc).__name__}: {exc})")
    rows = current.player_games()
    covered = set(rows["GAME_ID"].astype(str)) if not rows.empty else set()
    box = {(str(r.GAME_ID), int(r.PLAYER_ID)): {"PTS": r.PTS, "REB": r.REB, "AST": r.AST, "FG3M": r.FG3M}
           for r in rows.itertuples()} if not rows.empty else {}
    before = picks_mod.record(log)
    picks_mod.grade(log, finals_from_rows(log, rows), box, covered)
    after = picks_mod.record(log)
    if after == before:
        print(f"NBA: {len(waiting)} pick(s) still waiting; the game log does not hold their games yet")
        return 0
    picks_mod.save_log(log)
    payload = json.loads(BOARD.read_text(encoding="utf-8"))
    refresh_board(payload, log)
    tmp = BOARD.with_suffix(".json.tmp")
    from scripts import json_safe
    tmp.write_text(json_safe.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(BOARD)
    r = after["props"]
    print(f"NBA graded: props {r['correct']}-{r['wrong']} ({r['pending']} pending), "
          f"winners {after['team_winner']['correct']}-{after['team_winner']['wrong']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
