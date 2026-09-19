"""Hand-curated player availability, kept by the cloud team-news routine.

The NFL board already removes a player the API-NFL injury report lists as OUT.
This adds what that feed misses or reports late -- a player ruled out at the
final practice, or on the inactives list 90 minutes before kickoff -- written to
data-raw/<sport>/news.json by the scheduled routine from verified reporting.

It can only ever make the board MORE cautious: a manual "out" removes a player, a
manual "doubt" flags him, and a manual "doubt" never overrides an official OUT.
News older than MAX_AGE_DAYS is ignored, because last week's inactive is not this
week's, and an undated entry is not trusted at all.

Shared with the NBA board (nba.publish), the same way nba already reuses nfl.model.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

MAX_AGE_DAYS = 8
STATUSES = ("out", "doubt")


def load_player_news(path, now=None) -> dict:
    """player name -> {"status", "team", "detail", "source"} from a news file."""
    now = now or datetime.now(timezone.utc)
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for name, entry in (raw.get("players") or {}).items():
        if not isinstance(entry, dict) or entry.get("status") not in STATUSES:
            continue
        try:
            checked = datetime.fromisoformat(str(entry.get("checked")).replace("Z", "+00:00"))
        except ValueError:
            continue                          # undated news is not trusted
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=timezone.utc)
        if now - checked > timedelta(days=MAX_AGE_DAYS):
            continue                          # stale: last week's news
        out[name] = {"status": entry["status"], "team": entry.get("team"),
                     "detail": f"manual: {entry.get('note') or entry['status']}",
                     "source": "manual"}
    return out


def merge(api: dict, manual: dict) -> dict:
    """The API report with manual news laid over it -- never less cautious."""
    merged = dict(api)
    for name, entry in manual.items():
        held = merged.get(name) or {}
        if entry["status"] == "out" or held.get("status") != "out":
            merged[name] = entry
    return merged
