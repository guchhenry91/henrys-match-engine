"""2026-27 fixtures (and live results) from fixturedownload.com JSON feeds."""
import json
import time
from pathlib import Path
import urllib.request

import pandas as pd

from leagues import config
from leagues.names import canonical

FEED = "https://fixturedownload.com/feed/json/{slug}"


def parse_fixtures(raw: list[dict], league: str) -> pd.DataFrame:
    """Pure parser — takes the decoded JSON list, returns a clean DataFrame."""
    rows = []
    for r in raw:
        hg, ag = r.get("HomeTeamScore"), r.get("AwayTeamScore")
        played = hg is not None and ag is not None
        rows.append({
            "match_id": r["MatchNumber"],
            "round": r["RoundNumber"],
            "date": pd.to_datetime(r["DateUtc"], utc=True),
            "venue": r.get("Location") or "",
            "home": canonical(r["HomeTeam"], league),
            "away": canonical(r["AwayTeam"], league),
            "home_goals": int(hg) if played else pd.NA,
            "away_goals": int(ag) if played else pd.NA,
            "played": played,
        })
    return pd.DataFrame(rows, columns=["match_id", "round", "date", "venue",
                                       "home", "away", "home_goals", "away_goals",
                                       "played"])


SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "_snapshots"


def fetch_fixtures(league: str) -> pd.DataFrame:
    """Download the season's fixtures+results for one league.

    fixturedownload is the ONLY source for this. ClubElo went dark for days once
    and was removed for exactly that reason, but results and fixtures still have no
    second provider -- so a successful fetch is snapshotted, and an outage falls
    back to the last good copy rather than taking the whole model down.

    The fallback is deliberately loud and deliberately limited: it CANNOT invent
    results that happened during the outage, so anything it returns is at best as
    fresh as the last successful run. The staleness gate in sanity_check will still
    catch a file built from an old snapshot, which is the correct outcome -- this
    exists so a brief outage degrades instead of failing, not so a long one passes
    unnoticed.
    """
    slug = config.get(league).fixture_slug
    snap = SNAPSHOT_DIR / f"fixtures_{league.lower()}.json"
    req = urllib.request.Request(
        FEED.format(slug=slug),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        if not snap.exists():
            raise                       # no fallback to offer; fail as before
        age_h = (time.time() - snap.stat().st_mtime) / 3600.0
        print(f"WARNING: fixtures feed unavailable for {league} ({exc}); "
              f"falling back to a snapshot {age_h:.0f}h old -- results since then "
              f"are MISSING and the staleness gate should catch this")
        raw = json.loads(snap.read_text(encoding="utf-8"))
        return parse_fixtures(raw, league)

    fx = parse_fixtures(raw, league)
    try:                                # snapshot only a fetch that parsed cleanly
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = snap.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw), encoding="utf-8")
        tmp.replace(snap)
    except Exception as exc:
        print(f"note: could not snapshot {league} fixtures ({exc})")
    return fx
