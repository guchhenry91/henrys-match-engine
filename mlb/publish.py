"""Build the published MLB board: data/mlb/board.json.

EVIDENCE, NOT PICKS -- for now. The model and its fifteen-season walk-forward are
real; what is not wired yet is the 2027 schedule, probable pitchers and lineups
(MLB StatsAPI) and bookmakers' lines (The Odds API). The 2026 season is over bar
the postseason, so the first live slate is Opening Day 2027. Until then the tab
publishes exactly what the gate measured, market by market, including what it
withheld and why.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from mlb import config

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mlb" / "board.json"
REPORT = ROOT / "data-raw" / "mlb" / "backtest_report.json"
KEYS = ("released", "n", "brier", "baseline_brier", "accuracy", "ece", "failures",
        "seasons_scored", "by_line_step", "home_win_rate", "base_rate")


def build() -> dict:
    try:
        raw = json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    markets = raw.get("markets") or {}
    evidence = {m: {k: v.get(k) for k in KEYS if k in v} for m, v in markets.items()}
    released = sorted(m for m, v in markets.items() if v.get("released"))
    withheld = sorted(m for m, v in markets.items() if not v.get("released"))
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "status": "evidence_only",
        "status_note": (
            "MLB is backtested, not yet live. Every number below is what the model "
            "scored on seasons it never saw (Retrosheet game logs, 2011-2025). Live "
            "picks -- team winners, team totals and player props on bookmakers' "
            "lines -- start with the 2027 season."),
        "seasons_loaded": list(config.SEASONS),
        "seasons_scored": list(config.SCORED_SEASONS),
        "markets_released": released,
        "markets_withheld": withheld,
        "evidence": evidence,
        "caveats": [
            "Player props and team totals are trained and gated across a SPREAD of "
            "lines around each player's median, and must be calibrated at every "
            "step, because live picks will stand on bookmakers' lines.",
            "The team-winner Elo's settings are fitted on 2007-2010 only and then "
            "held, so no scored season influenced them.",
            "2020 was a 60-game season; it is scored like any other.",
        ],
    }


def main() -> int:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(f"MLB board: released={payload['markets_released']} "
          f"withheld={payload['markets_withheld']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
