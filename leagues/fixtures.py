"""2026-27 fixtures (and live results) from fixturedownload.com JSON feeds."""
import json
import time
from pathlib import Path
import urllib.request

import pandas as pd

from leagues import config
from leagues.names import canonical

FEED = "https://fixturedownload.com/feed/json/{slug}"

OVERRIDES = (Path(__file__).resolve().parent.parent / "data-raw" / "leagues"
             / "fixture_times.json")
RESULTS = (Path(__file__).resolve().parent.parent / "data-raw" / "leagues"
           / "results_override.json")

# A domestic European fixture kicks off between these UTC hours. Deliberately
# WIDE -- it is a nonsense detector, not a schedule. The real 2026-27 spread is
# 11:00Z (a 12:30 BST Saturday) to 20:00Z (a 21:00 BST Monday), so this leaves
# room either side and still catches what the feed actually gets wrong:
# La Liga times published 10 hours early (04:00-09:00Z), Bundesliga rounds with
# a 00:00Z placeholder, and Ligue 1 rounds at 22:00-23:00Z.
KICKOFF_UTC_EARLIEST = 10
KICKOFF_UTC_LATEST = 21


def _time_overrides(league: str) -> dict:
    """'Home|Away' -> true kickoff UTC, from the hand-verified override file."""
    if not OVERRIDES.exists():
        return {}
    try:
        raw = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    except Exception as exc:               # never let a bad override file stop a publish
        print(f"WARNING: could not read {OVERRIDES.name} ({exc}); no time overrides")
        return {}
    return {k: v["utc"] for k, v in (raw.get(league) or {}).items()
            if isinstance(v, dict) and v.get("utc")}


def _result_overrides(league: str) -> dict:
    """'Home|Away' -> verified final result, for fixtures the feed has not scored.

    The feed is the authority and this never overrules it -- see the caller: an
    override is used only where the feed's score is MISSING, and a disagreement
    once the feed catches up is reported rather than hidden.
    """
    if not RESULTS.exists():
        return {}
    try:
        raw = json.loads(RESULTS.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: could not read {RESULTS.name} ({exc}); no result overrides")
        return {}
    out = {}
    for k, v in (raw.get(league) or {}).items():
        if not isinstance(v, dict):
            continue
        hg, ag = v.get("home_goals"), v.get("away_goals")
        # Only a FULL-TIME result may grade a pick. A live or half-time score
        # would silently enter the record as final.
        if isinstance(hg, int) and isinstance(ag, int) and v.get("status") == "FT":
            out[k] = (hg, ag)
    return out


def parse_fixtures(raw: list[dict], league: str) -> pd.DataFrame:
    """Pure parser — takes the decoded JSON list, returns a clean DataFrame.

    Applies the verified kickoff-time overrides and flags any REMAINING time that
    is not a plausible kickoff hour. The feed's DateUtc cannot be trusted on its
    own: see fixture_times.json for the 10-hour La Liga error that shipped on
    2026-08-15.
    """
    over = _time_overrides(league)
    results = _result_overrides(league)
    rows = []
    for r in raw:
        hg, ag = r.get("HomeTeamScore"), r.get("AwayTeamScore")
        played = hg is not None and ag is not None
        home = canonical(r["HomeTeam"], league)
        away = canonical(r["AwayTeam"], league)
        verified = results.get(f"{home}|{away}")
        if verified:
            if not played:
                # The feed has not scored this fixture yet. Fill in the verified
                # full-time result so the pick can grade on schedule instead of
                # waiting on the feed.
                hg, ag = verified
                played = True
            elif (hg, ag) != verified:
                # The feed caught up and disagrees. The FEED WINS -- it is the
                # source of record and self-corrects -- but a silent divergence
                # would hide either a bad override or a bad feed, so say so.
                print(f"WARNING: {league}: {home} v {away}: verified override "
                      f"{verified[0]}-{verified[1]} disagrees with the feed's "
                      f"{hg}-{ag}; using the FEED. Re-check results_override.json.")
        fixed = over.get(f"{home}|{away}")
        date = pd.to_datetime(fixed or r["DateUtc"], utc=True)
        rows.append({
            "match_id": r["MatchNumber"],
            "round": r["RoundNumber"],
            "date": date,
            "venue": r.get("Location") or "",
            "home": home,
            "away": away,
            "home_goals": int(hg) if played else pd.NA,
            "away_goals": int(ag) if played else pd.NA,
            "played": played,
            # An override is verified, so it is never suspect. Anything else
            # outside the plausible window is a time we do not believe.
            "time_suspect": bool(fixed is None and not
                                 KICKOFF_UTC_EARLIEST <= date.hour <= KICKOFF_UTC_LATEST),
        })
    return pd.DataFrame(rows, columns=["match_id", "round", "date", "venue",
                                       "home", "away", "home_goals", "away_goals",
                                       "played", "time_suspect"])


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
