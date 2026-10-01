"""Build the published MLB board: data/mlb/board.json.

LIVE: today's games (and the postseason as it comes) from MLB's StatsAPI, team
winners from the Elo, and every player prop and team total on a BOOKMAKER'S line
(mlb/board.py), frozen before first pitch and graded from the box score
(mlb/picks.py). The gate's fifteen-season evidence is published alongside. If
the schedule feed does not answer, the board falls back to evidence only.
"""
import json

from scripts import json_safe
from tracking import performance, trust, value
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mlb import board, book_lines, config, current
from mlb import picks as picks_mod

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mlb" / "board.json"
REPORT = ROOT / "data-raw" / "mlb" / "backtest_report.json"
KEYS = ("released", "n", "brier", "baseline_brier", "accuracy", "ece", "failures",
        "seasons_scored", "by_line_step", "home_win_rate", "base_rate")


def build(now=None, sched=None) -> dict:
    now = now or datetime.now(timezone.utc)
    try:
        raw = json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    markets = raw.get("markets") or {}
    evidence = {m: {k: v.get(k) for k in KEYS if k in v} for m, v in markets.items()}
    released = sorted(m for m, v in markets.items() if v.get("released"))
    withheld = sorted(m for m, v in markets.items() if not v.get("released"))
    live = board.build(sched, now, released) if sched is not None else None
    status = "live" if live is not None else "evidence_only"
    return {
        "updated": now.isoformat(),
        "status": status,
        "status_note": (
            "Live board. Team winners are the Elo model's; every player prop and team "
            "total stands on a bookmaker's line (Pinnacle, DraftKings or FanDuel) and "
            "the probability is the model's for THAT line. Home runs are shown where "
            "the model rates the player above the book's own price."
            if status == "live" else
            "MLB's schedule feed did not answer this run, so nothing is priced; the "
            "numbers below are what the model scored on seasons it never saw."),
        "seasons_loaded": list(config.SEASONS),
        "seasons_scored": list(config.SCORED_SEASONS),
        "markets_released": released,
        "markets_withheld": withheld,
        "games": (live or {}).get("games") or [],
        "props": {m: {"released": m in released,
                      "picks": ((live or {}).get("props") or {}).get(m) or []}
                  for m in config.MARKETS},
        "odds": {"checked_at": (live or {}).get("odds_checked_at"),
                 "books": ["Pinnacle", "DraftKings", "FanDuel"], "source": "The Odds API"},
        "evidence": evidence,
        "caveats": [
            "Props and team totals are trained and gated across a SPREAD of lines around "
            "each player's median and must be calibrated at every step, so the model can "
            "be asked about the bookmaker's line.",
            "A player's team is his newest box score this season; one the book lists in a "
            "game his team is not in is skipped rather than guessed onto a side.",
            "The team-winner Elo's settings were fitted on 2007-2010 only and then held.",
            "Edges are against the book's de-vigged price; home runs are priced one-sided "
            "where only an over is quoted. A large edge against a sharp book is more often "
            "the model's error than the market's -- the record is the test.",
        ],
    }


def void_unplayed(log: dict, now, opener=None) -> int:
    """VOID every pick whose game was cancelled or postponed -- as bet365 settles it.

    A game that is never played never gets a final score or a box score, so its
    picks waited forever: 17 from the 27 Sep game 823490 (cancelled) were still
    pending four days later, found by the health check. Asked by game id, so it
    works however long ago the game was scheduled."""
    waiting = {str(e["game_id"]) for e in log.values()
               if isinstance(e, dict) and not e.get("graded") and e.get("tipoff")
               and (now - datetime.fromisoformat(str(e["tipoff"]).replace("Z", "+00:00"))) > timedelta(hours=24)}
    if not waiting:
        return 0
    try:
        status = current.game_status(waiting, **({"opener": opener} if opener else {}))
    except Exception as exc:
        print(f"  could not check unplayed games ({type(exc).__name__}); left pending")
        return 0
    n = 0
    for e in log.values():
        if isinstance(e, dict) and not e.get("graded") and status.get(int(e.get("game_id") or 0)) in current.NOT_PLAYED:
            e["graded"], e["actual"] = "void", None
            e["void_reason"] = f"game {status[int(e['game_id'])].lower()}"
            n += 1
    if n:
        print(f"  voided {n} pick(s) on games that were not played")
    return n


def freeze_and_grade(payload: dict, sched, now=None) -> dict:
    now = now or datetime.now(timezone.utc)
    log = picks_mod.load_log()
    # Pull every pick toward the book by the weight the live record supports
    # (tracking/trust.py), THEN judge value -- so only a real price gap reads as value.
    trust.apply_line_board(payload, trust.weights()["mlb"])
    value.annotate_line_board(payload)
    frozen = picks_mod.freeze(payload, log, now, book_lines.is_last_run_before)
    # The first price seen and the price frozen on, for profit and closing-line
    # value (tracking/performance.py). Uses only odds already fetched.
    performance.stamp_line_log(log, ROOT / "data-raw" / "mlb" / "odds_api.json")
    finals, box, covered = {}, {}, set()
    if sched is not None:
        done = sched[sched["status"].isin(current.FINISHED)]
        for g in done.itertuples():
            try:
                h, a = float(g.home_score), float(g.away_score)
            except (TypeError, ValueError):
                continue
            if h != a:
                finals[str(g.game_pk)] = "home" if h > a else "away"
    for kind, stat_cols in (("batting", ("H", "HRR", "HR", "RBI")), ("pitching", ("K",)),
                            ("team", ("RUNS",))):
        rows = current.load(kind)
        if rows.empty:
            continue
        covered |= {str(g)[3:] for g in rows["GAME_ID"]}
        for r in rows.to_dict("records"):
            box[(str(r["GAME_ID"])[3:], str(r["PLAYER_ID"]))] = {c: r[c] for c in stat_cols}
    picks_mod.grade(log, finals, box, covered)
    void_unplayed(log, now)
    picks_mod.save_log(log)
    for g in payload["games"]:
        e = log.get(picks_mod.game_key(g))
        if e:
            g.update({"pick": e["pick"], "p_pick": e["p_pick"], "locked": True})
    for block in payload["props"].values():
        for p in block["picks"]:
            e = log.get(picks_mod.prop_key(p))
            if e:
                p.update({"line": e["line"], "probability": e["probability"],
                          "book_p": e["book_p"], "book_price": e["book_price"],
                          "edge": e["edge"], "locked": True})
    value.annotate_line_board(payload)     # on the frozen line and price
    payload["record"] = picks_mod.record(log)
    payload["performance"] = performance.by_market(
        [e for e in log.values() if isinstance(e, dict) and e.get("graded")],
        performance.market_of)
    # Every graded pick: the Results tab lists them per market, and a list cut at
    # 200 disagreed with the record's own counts (243 graded on 2026-10-01).
    payload["settled"] = picks_mod.settled(log, limit=100_000)
    print(f"  froze {frozen} pick(s); record {payload['record']['team_winner']['correct']}-"
          f"{payload['record']['team_winner']['wrong']} winners, "
          f"{payload['record']['props']['correct']}-{payload['record']['props']['wrong']} props")
    return payload


def main() -> int:
    now = datetime.now(timezone.utc)
    today = now.date()
    try:
        sched = current.schedule((today - timedelta(days=4)).isoformat(),
                                 (today + timedelta(days=3)).isoformat())
    except Exception as exc:
        print(f"MLB schedule unavailable ({exc})")
        sched = None
    payload = build(now, sched)
    payload = freeze_and_grade(payload, sched, now) if sched is not None else payload
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json_safe.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(f"MLB board: released={payload['markets_released']} "
          f"withheld={payload['markets_withheld']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
