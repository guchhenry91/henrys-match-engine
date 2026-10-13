"""ESPN's free Champions League match data: player shots INCLUDING blocked shots.

Why it exists: bet365 settles player shot markets on Opta, whose shot count includes
blocked shots. API-Football's player "shots" leaves blocked shots out (9.6 a side
against ~17), so "2+ shots" cannot be graded from it. ESPN's per-player totalShots
does include them (a team's players sum to its total, blocked included), so the
2+ shots market is tested and graded from ESPN. No key; one scoreboard request per
match day and one summary per match.
"""
from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
import urllib.request

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions"
PACE = 0.4
ALIASES = {"internazionale": "inter", "inter milan": "inter", "milan": "ac milan"}


def get(path: str, opener=urllib.request.urlopen):
    req = urllib.request.Request(f"{BASE}/{path}", headers={"User-Agent": "Mozilla/5.0"})
    with opener(req, timeout=30) as r:
        out = json.load(r)
    time.sleep(PACE)
    return out


def _n(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(fc|cf|ac|as|sc|ssc|club|de|afc|cd|vfb|vfl|sv|st|saint|munchen|munich)\b", " ", s)
    s = " ".join(re.sub(r"[^a-z ]", " ", s).split())
    return ALIASES.get(s, s)


def same_club(a: str, b: str) -> bool:
    x, y = _n(a), _n(b)
    if not x or not y:
        return False
    if x == y or set(x.split()) <= set(y.split()) or set(y.split()) <= set(x.split()):
        return True
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.75


def events_on(day: str, opener=urllib.request.urlopen) -> list:
    """[(event id, home name, away name)] for one day (YYYY-MM-DD)."""
    d = get(f"scoreboard?dates={day.replace('-', '')}", opener)
    out = []
    for e in d.get("events") or []:
        comp = (e.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): (c.get("team") or {}).get("displayName")
                 for c in comp.get("competitors") or []}
        out.append((str(e["id"]), teams.get("home"), teams.get("away")))
    return out


def find_event(events: list, home_names: tuple, away_names: tuple) -> str | None:
    hits = [eid for eid, h, a in events
            if any(same_club(h, n) for n in home_names if n)
            and any(same_club(a, n) for n in away_names if n)]
    return hits[0] if len(hits) == 1 else None


def players(event_id: str, opener=urllib.request.urlopen) -> list:
    """[{team, player, played, started, shots, sot, goals}] for one match."""
    d = get(f"summary?event={event_id}", opener)
    out = []
    for side in d.get("rosters") or []:
        team = (side.get("team") or {}).get("displayName")
        for p in side.get("roster") or []:
            st = {s.get("name"): s.get("value") for s in p.get("stats") or []}
            apps = st.get("appearances") or 0
            out.append({"team": team, "player": (p.get("athlete") or {}).get("displayName"),
                        "played": bool(apps), "started": bool(p.get("starter")),
                        "shots": int(st.get("totalShots") or 0),
                        "sot": int(st.get("shotsOnTarget") or 0),
                        "goals": int(st.get("totalGoals") or 0)})
    return out
