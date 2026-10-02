"""Build the published NBA board: data/nba/board.json.

WHAT THIS BOARD HONESTLY CONTAINS TODAY IS EVIDENCE, NOT PICKS. There is no
fixture feed wired in yet -- the API comes later -- so there is nothing to project
onto. Publishing an empty picks list with the reason stated is the whole point:
the tab shows what fifteen seasons of validation actually found, and says plainly
that no game has been priced.

The alternative was a tab that looks live and does nothing, which this codebase
already decided is worse than no tab at all. This is the middle path: real
content, no invented numbers.

WHEN THE FIXTURE FEED ARRIVES, `games` and `props` fill in and nothing else about
the payload's shape changes -- the UI already renders empty sections with their
reason, so a board with picks needs no second design.
"""
import json

from scripts import json_safe
from tracking import manifest, release
from tracking import news_edge, performance, trust, value
from datetime import datetime, timezone
from pathlib import Path

from nba import board, book_lines, config, current
from nba import picks as picks_mod
from nfl.news import load_player_news

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nba"
REPORT = ROOT / "data-raw" / "nba" / "backtest_report.json"
NEWS = ROOT / "data-raw" / "nba" / "news.json"


def _news_edge(picks: list, out_state: dict) -> list:
    """Tag picks lifted by a teammate ruled out in the last few hours."""
    news_edge.annotate(picks, out_state)
    return picks


def apply_news(picks: list, news: dict) -> list:
    """Remove players ruled OUT and flag doubts, from data-raw/nba/news.json.

    Written by the cloud team-news routine. It has nothing to act on until the
    fixture feed exists and props are published, but it is wired in now so the
    first priced slate is already filtered -- a ruled-out star on the first NBA
    board would be the worst possible debut.
    """
    kept = []
    for pick in picks:
        entry = news.get(pick.get("player"))
        if entry and entry["status"] == "out":
            continue
        if entry and entry["status"] == "doubt":
            pick = {**pick, "availability": "doubt", "injury_note": entry["detail"]}
        kept.append(pick)
    return kept


def evidence() -> dict:
    """The gate's own findings, per market, exactly as measured."""
    try:
        raw = json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for market, block in (raw.get("markets") or {}).items():
        out[market] = {
            "released": block.get("released"),
            "n": block.get("n"),
            "brier": block.get("brier"),
            "baseline_brier": block.get("baseline_brier"),
            "accuracy": block.get("accuracy"),
            "ece": block.get("ece"),
            "seasons_scored": block.get("seasons_scored"),
            "failures": block.get("failures") or [],
            # THE NUMBER THAT STOPS THE HEADLINE FLATTERING. MIN_LINE floors a
            # line too low to quote, and where it binds the "line" is a constant
            # rather than the player's median -- a far more predictable question.
            "floor_share": block.get("floor_share"),
            "above_floor": block.get("above_floor"),
        }
    return out


def build(now=None, sched=None) -> dict:
    """The board: the live slate when the NBA's schedule feed answers, and the
    gate's evidence always."""
    now = now or picks_mod.now_utc()
    ev = evidence()
    news = load_player_news(NEWS)
    released = sorted(m for m, v in ev.items()
                      if v.get("released") and m != "team_winner")
    withheld = sorted(m for m, v in ev.items()
                      if not v.get("released") and m != "team_winner")
    if sched is None:
        try:
            sched = current.schedule()
        except Exception as exc:
            print(f"NBA schedule unavailable ({exc}); publishing evidence only")
    ruled_out = [n for n, e in news.items() if (e or {}).get("status") == "out"]
    out_state = news_edge.remember(ROOT / "data-raw" / "nba" / "out_since.json",
                                   {n: e for n, e in news.items() if (e or {}).get("status") == "out"})
    live = board.build(sched, now, released, ruled_out=ruled_out) if sched is not None else         {"games": [], "props": {m: [] for m in config.MARKETS}, "held_back": {},
         "odds_checked_at": None}
    season_started = sched is not None and bool(
        ((sched["stage"] == current.REGULAR) & (sched["status"] == current.FINAL)).any())
    status = "live" if sched is not None else "evidence_only"
    note = (
        "Live board. Team-winner picks are the Elo model's; every player prop "
        "stands on a bookmaker's line (Pinnacle, DraftKings or FanDuel) and the "
        "probability is the model's for THAT line. A player no book quotes is not "
        "priced." if status == "live" else
        "The NBA schedule feed did not answer this run, so no games are priced. "
        "The numbers below are what the model scored on seasons it never saw.")
    if status == "live" and not season_started:
        note += (" The regular season has not tipped off yet: the first slate is "
                 "shown as soon as it is within a week, and player props appear once "
                 "the books post lines, usually on game day.")
    return {
        "updated": now.isoformat(),
        "season": config.CURRENT_SEASON,
        "seasons_backtested": list(config.SEASONS),
        "status": status,
        "status_note": note,
        "markets_released": released,
        "markets_withheld": withheld,
        "games": live["games"],
        "props": {m: {"released": m in released,
                      "picks": _news_edge(apply_news(live["props"].get(m) or [], news), out_state),
                      "below_trained_floor": (live.get("held_back") or {}).get(m, 0)}
                  for m in config.MARKETS},
        # Every priced prop per game, for the game pages (index.html viewGame).
        "game_props": live.get("game_props") or {},
        "odds": {"checked_at": live.get("odds_checked_at"),
                 "books": ["Pinnacle", "DraftKings", "FanDuel"],
                 "source": "The Odds API"},
        "news": {"players_flagged": len(news),
                 "source": "data-raw/nba/news.json (cloud team-news routine)",
                 "applied_to": "player props: OUT removed, doubtful flagged"},
        "evidence": ev,
        "caveats": [
            "Player props are asked at the BOOKMAKER'S line. The model is trained "
            "across a spread of lines around each player's median and was gated at "
            "every step of that spread (calibration within 0.04 at each), so it can "
            "be asked about the book's number and mean it.",
            "Edges are against the book's de-vigged price. A large edge against a "
            "sharp book such as Pinnacle is more often the model's error than the "
            "market's -- the record, not the edge, is the test.",
            "A player's team is his newest box score this season, preseason "
            "included. A player whose known team is not in the game the book lists "
            "him for is skipped rather than guessed onto a side.",
        ] + withheld_caveat(ev, withheld),
    }


def _show_frozen(payload: dict, log: dict) -> None:
    """Once frozen, the board shows the FROZEN pick -- the one the record grades."""
    for g in payload["games"]:
        entry = log.get(picks_mod.game_key(g))
        if entry:
            g.update({"pick": entry["pick"], "p_pick": entry["p_pick"], "locked": True})
    for block in payload["props"].values():
        for p in block["picks"]:
            entry = log.get(picks_mod.prop_key(p))
            if entry:
                p.update({"line": entry["line"], "probability": entry["probability"],
                          "line_source": entry["line_source"], "book": entry["book"],
                          "book_p": entry["book_p"], "book_price": entry["book_price"],
                          "edge": entry["edge"], "locked": True})


def freeze_and_grade(payload: dict, sched, now=None) -> dict:
    now = now or picks_mod.now_utc()
    log = picks_mod.load_log()
    # Pull every pick toward the book by the weight the live record supports
    # (tracking/trust.py), THEN judge value -- so only a real price gap reads as value.
    nba_trust = trust.weights()["nba"]
    trust.apply_line_board(payload, nba_trust)
    trust.apply_game_props(payload.get("game_props"), nba_trust)
    value.annotate_line_board(payload)
    value.annotate_game_props(payload.get("game_props"))
    frozen = picks_mod.freeze(payload, log, now, book_lines.is_last_run_before)
    # The first price seen and the price frozen on, for profit and closing-line
    # value (tracking/performance.py). Uses only odds already fetched.
    performance.stamp_line_log(log, ROOT / "data-raw" / "nba" / "odds_api.json")
    finals, box, covered = {}, {}, set()
    if sched is not None:
        done = sched[(sched["stage"] == current.REGULAR) & (sched["status"] == current.FINAL)]
        for g in done.itertuples():
            try:
                h, a = float(g.home_score), float(g.away_score)
            except (TypeError, ValueError):
                continue
            if h != a:
                finals[str(g.game_id)] = "home" if h > a else "away"
    rows = current.player_games()
    if not rows.empty:
        covered = set(rows["GAME_ID"].astype(str))
        for r in rows.itertuples():
            box[(str(r.GAME_ID), int(r.PLAYER_ID))] = {
                "PTS": r.PTS, "REB": r.REB, "AST": r.AST, "FG3M": r.FG3M}
    picks_mod.grade(log, finals, box, covered)
    manifest.stamp(log.values(), "nba")      # which model made each new pick
    release.mark(log.values(), "nba", performance.market_of)   # official or tracked
    picks_mod.save_log(log)
    _show_frozen(payload, log)
    value.annotate_line_board(payload)     # on the frozen line and price
    payload["record"] = picks_mod.record(log)
    payload["performance"] = performance.by_market(
        [e for e in log.values() if isinstance(e, dict) and e.get("graded")],
        performance.market_of)
    # Every graded pick, not the newest 200: the Results tab lists them per market
    # and must agree with the record's own counts (the MLB list fell short at 200).
    payload["settled"] = picks_mod.settled(log, limit=100_000)
    print(f"  froze {frozen} pick(s); record {payload['record']['team_winner']['correct']}-"
          f"{payload['record']['team_winner']['wrong']} winners, "
          f"{payload['record']['props']['correct']}-{payload['record']['props']['wrong']} props")
    return payload


def withheld_caveat(ev: dict, withheld: list) -> list:
    """Name the withheld markets and the gate's own reason, or say nothing.

    DERIVED, NEVER HARDCODED. This line used to be the literal string "Points is
    WITHHELD: it lost to the baseline in 2023, 2024 and 2025" -- true when it was
    written and false the moment the training window was capped and points
    started clearing the gate. A caveat that outlives the condition it describes
    is worse than no caveat: it is the page confidently reporting a failure that
    is not happening any more.
    """
    if not withheld:
        return []
    out = []
    for market in withheld:
        why = "; ".join((ev.get(market) or {}).get("failures") or []) or "failed the gate"
        out.append(f"{market.capitalize()} is WITHHELD: {why}.")
    return out


def main() -> int:
    now = picks_mod.now_utc()
    try:
        sched = current.schedule()
    except Exception as exc:
        print(f"NBA schedule unavailable ({exc})")
        sched = None
    payload = freeze_and_grade(build(now, sched), sched, now)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "board.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json_safe.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(f"NBA board: status={payload['status']}, "
          f"released={payload['markets_released']}, "
          f"withheld={payload['markets_withheld']}, "
          f"{len(payload['seasons_backtested'])} seasons backtested")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
