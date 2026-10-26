"""Write every upcoming soccer kick-off (five leagues + Champions League) for the Worker.

WHY (owner, 2026-10-26, Cloudflare bill). The full soccer refresh ran every 30 minutes
from 11:00 to 23:00 UTC every day, games or not -- ~44 container-hours a week. The
Worker now keeps the half-hour runs (the :47 slots) only on a UTC day with a soccer
kick-off on it, and reads that from this file (`data-raw/leagues/kickoffs.json`);
quiet days keep the hourly :17 runs and the 3-hourly ones. Locking and results run in
the 10-minute lock job either way, so no pick or result waits on this.

Read from the boards just published, so it lists exactly the games the site shows.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "leagues" / "kickoffs.json"
LEAGUE_FILES = ("pl.json", "laliga.json", "bundesliga.json", "ligue1.json", "seriea.json")


def kickoffs(root=ROOT) -> list:
    out = set()
    for name in LEAGUE_FILES:
        try:
            board = json.loads((root / "data" / "leagues" / name).read_text(encoding="utf-8"))
        except Exception:
            continue
        out.update(str(m["date"]) for m in board.get("matches") or [] if "T" in str(m.get("date", "")))
    try:
        ucl = json.loads((root / "data" / "ucl" / "board.json").read_text(encoding="utf-8"))
        out.update(str(m["kickoff"]) for m in ucl.get("matches") or [] if m.get("kickoff"))
    except Exception:
        pass
    return sorted(out)


def main(root=ROOT, out=OUT) -> int:
    starts = kickoffs(root)
    if not starts:
        print("no soccer kick-offs on the boards; kickoffs.json left as it was")
        return 0
    text = json.dumps(starts) + "\n"
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(f"{len(starts)} soccer kick-offs, next {next(iter(starts), None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
