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
from datetime import datetime, timezone
from pathlib import Path

from nba import config

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nba"
REPORT = ROOT / "data-raw" / "nba" / "backtest_report.json"


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


def build() -> dict:
    ev = evidence()
    released = sorted(m for m, v in ev.items()
                      if v.get("released") and m != "team_winner")
    withheld = sorted(m for m, v in ev.items()
                      if not v.get("released") and m != "team_winner")
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "season": config.CURRENT_SEASON,
        "seasons_backtested": list(config.SEASONS),
        "status": "evidence_only",
        "status_note": (
            "The model and its fifteen-season backtest are live; no fixtures are "
            "priced yet because the NBA data feed for upcoming games is not wired "
            "in. No pick on this tab is a prediction about a future game -- the "
            "numbers below are what the model scored on seasons it never saw."),
        "markets_released": released,
        "markets_withheld": withheld,
        # Empty until a fixture feed exists. Shape kept so nothing about the
        # payload changes when it does.
        "games": [],
        "props": {m: {"released": m in released, "picks": []} for m in config.MARKETS},
        "evidence": ev,
        "caveats": [
            "Lines are each player's own entering median, floored at what a book "
            "would actually quote -- not a sportsbook price. No NBA odds have been "
            "fetched, so the board makes NO claim to beat a bookmaker.",
            "Where that floor binds, the line is a CONSTANT rather than the "
            "player's median, and the market becomes 'will a rotation player clear "
            "a fixed number' -- far more predictable than a balanced prop. It binds "
            "on 79% of assists rows and 80% of threes rows, so each market "
            "publishes both its headline and its above-the-floor number.",
        ] + withheld_caveat(ev, withheld),
    }


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
    payload = build()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "board.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(f"NBA board: status={payload['status']}, "
          f"released={payload['markets_released']}, "
          f"withheld={payload['markets_withheld']}, "
          f"{len(payload['seasons_backtested'])} seasons backtested")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
