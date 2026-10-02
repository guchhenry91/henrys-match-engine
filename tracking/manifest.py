"""Which model made this pick? A version stamp on every newly frozen pick.

WHY (improvement report, point #6). The models changed four times in one week --
the 2026 NFL data fix, the teammates-out correction (later switched off), park
factors, model trust. A graded record that cannot say which version produced each
pick cannot tell which version earned or lost money. Every pick frozen from now on
carries:

  commit   the code version (the GitHub Actions commit, or the local HEAD)
  report   a short hash of the sport's release-gate report -- the evidence the
           released markets stood on at that moment
  trust    the model-trust weight in force for that pick (US sports)

Stamped only on picks frozen in the last STAMP_WINDOW_HOURS that carry no stamp
yet, so an old frozen pick is never back-dated with a version it did not use.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMP_WINDOW_HOURS = 3
REPORTS = {"nfl": "data-raw/nfl/backtest_report.json", "nba": "data-raw/nba/backtest_report.json",
           "mlb": "data-raw/mlb/backtest_report.json", "soccer": "data-raw/leagues/backtest_report.json"}
_COMMIT = None


def commit() -> str | None:
    global _COMMIT
    if _COMMIT is None:
        sha = os.environ.get("GITHUB_SHA")
        if not sha:
            try:
                sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True, timeout=10).stdout.strip()
            except Exception:
                sha = ""
        _COMMIT = (sha or "")[:10] or None
    return _COMMIT


def report_hash(sport: str) -> str | None:
    try:
        return hashlib.sha256((ROOT / REPORTS[sport]).read_bytes()).hexdigest()[:12]
    except Exception:
        return None


def stamp(entries, sport: str, now=None) -> int:
    """Stamp every recently frozen, unstamped pick in `entries`. Returns how many."""
    now = now or datetime.now(timezone.utc)
    version = {"commit": commit(), "report": report_hash(sport)}
    n = 0
    for e in entries:
        if not isinstance(e, dict) or "model_version" in e or not e.get("locked_at"):
            continue
        try:
            locked = datetime.fromisoformat(str(e["locked_at"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        if locked.tzinfo is None:
            locked = locked.replace(tzinfo=timezone.utc)
        if now - locked > timedelta(hours=STAMP_WINDOW_HOURS):
            continue
        e["model_version"] = {**version, "trust": e.get("trust_w")}
        n += 1
    return n
