"""Rebuild the record_history rows that CI zeroed. One-off; safe to re-run.

WHY. tests/leagues/test_publish_multi.py ran publish.main() on empty stub boards
without redirecting PICKS_DIR, and CI runs the tests after publishing and before
committing. Every automated refresh therefore committed a 0-0 row for that day
into data-raw/leagues/record_history.json, and the next publish read it back, so
most days since 2026-08-20 read 0-0 in both copies.

HOW. A zero row is rebuilt from the settled picks on the published boards: the
record "so far" on day D counts every graded pick whose match was played BEFORE
D, which is what a publish that morning would have seen. Only rows that read 0-0
where the rebuild finds graded picks are replaced, and each is marked
`reconstructed: true` so it can never be mistaken for a row a publish wrote.
Real rows are left exactly as they are. Both copies are written identically.
"""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data-raw" / "leagues" / "record_history.json"
OUT = ROOT / "data" / "leagues" / "record_history.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _played_before(item, day: str) -> bool:
    when = str(item.get("date") or item.get("kickoff") or "")[:10]
    return bool(when) and when < day


def _tally(items):
    graded = [i for i in items if i.get("graded") in ("correct", "wrong")]
    correct = sum(i["graded"] == "correct" for i in graded)
    return {"correct": correct, "wrong": len(graded) - correct, "total": len(graded)}, graded


def rebuild(history: list, best_settled: list, player_settled: list) -> tuple[list, int]:
    fixed = 0
    out = []
    for row in history:
        zero = (row["best"]["total"] == 0 and row["players"]["total"] == 0)
        if not zero:
            out.append(row)
            continue
        day = row["date"]
        best, graded = _tally([s for s in best_settled if _played_before(s, day)])
        players, _ = _tally([s for s in player_settled if _played_before(s, day)])
        if best["total"] == 0 and players["total"] == 0:
            out.append(row)                 # genuinely nothing graded yet that day
            continue
        stated = (sum(s.get("p_pick") or 0 for s in graded) / len(graded)) if graded else None
        actual = (best["correct"] / best["total"]) if best["total"] else None
        out.append({"date": day, "best": best, "players": players,
                    "stated_pct": None if stated is None else round(100 * stated, 1),
                    "actual_pct": None if actual is None else round(100 * actual, 1),
                    "reconstructed": True})
        fixed += 1
    return out, fixed


def main() -> int:
    # The longer history wins as the base; for each date prefer a real (non-zero)
    # row from either copy before rebuilding what is still zero.
    rows = {}
    for path in (RAW, OUT):
        if path.exists():
            for row in _load(path):
                held = rows.get(row["date"])
                if held is None or (held["best"]["total"] == 0 and row["best"]["total"] > 0):
                    rows[row["date"]] = row
    history = [rows[d] for d in sorted(rows)]
    best = _load(ROOT / "data" / "leagues" / "best.json").get("settled") or []
    players = _load(ROOT / "data" / "leagues" / "player_picks.json").get("settled") or []
    history, fixed = rebuild(history, best, players)
    text = json.dumps(history, indent=2)
    for path in (RAW, OUT):
        path.write_text(text, encoding="utf-8")
    print(f"rebuilt {fixed} zeroed row(s); {len(history)} rows, through "
          f"{history[-1]['date'] if history else '-'} (today {date.today()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
