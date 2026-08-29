"""Load the cached Champions League history."""
import json
from pathlib import Path

import pandas as pd

from ucl import config

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data-raw" / "ucl" / "history.json"


def matches(seasons=None) -> pd.DataFrame:
    """Every finished match, oldest first.

    QUALIFYING ROUNDS ARE INCLUDED, deliberately. They are where the small clubs
    actually play: Viking, Sabah, Bodo/Glimt and Slovan Bratislava have far more
    European football in the qualifiers than in a league phase, and excluding
    those rounds would leave exactly the clubs with the least evidence with even
    less. They are real matches between real European sides, which is what the
    strength estimate needs.
    """
    try:
        raw = json.loads(HISTORY.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame(columns=["date", "home", "away", "home_goals",
                                     "away_goals", "round", "season"])
    wanted = set(str(s) for s in (seasons or config.SEASONS))
    rows = []
    for season, games in (raw.get("seasons") or {}).items():
        if season not in wanted:
            continue
        for game in games:
            rows.append({**game, "season": int(season)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    for column in ("home_goals", "away_goals"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["home_goals", "away_goals"])
    return frame.sort_values("date").reset_index(drop=True)


def api_name(name: str) -> str:
    """Our spelling of a drawn club -> the API's, where they differ."""
    return config.NAME_TO_API.get(name, name)


def drawn_clubs() -> dict:
    """API spelling -> pot, for the 36 clubs in the 2026/27 league phase."""
    out = {}
    for pot, names in config.POTS.items():
        for name in names:
            out[api_name(name)] = pot
    return out


def history_depth(frame: pd.DataFrame) -> dict:
    """club -> matches played, so thin clubs can be identified rather than
    silently seeded from nothing."""
    if frame.empty:
        return {}
    counts = {}
    for column in ("home", "away"):
        for name, n in frame[column].value_counts().items():
            counts[name] = counts.get(name, 0) + int(n)
    return counts


def results_by_id(season=None) -> dict:
    """Finished current-season matches keyed by API fixture id.

    The grading feed. Joined on the fixture ID rather than on club names: the
    board renders "Internazionale" where the draw list says "Inter Milan", and a
    name join settles a frozen bet against whichever club the string happened to
    match. An id either matches or it does not.

    Only rows that carry an id are returned. History cached before ids were
    recorded simply yields nothing here, which leaves those picks PENDING -- the
    honest outcome. Guessing at the join would be the alternative, and a wrong
    grade in an append-only record cannot be withdrawn.
    """
    season = str(season or config.CURRENT_SEASON)
    try:
        raw = json.loads(HISTORY.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for game in (raw.get("seasons") or {}).get(season, []):
        fixture_id = game.get("id")
        if fixture_id is None:
            continue
        if game.get("home_goals") is None or game.get("away_goals") is None:
            continue
        out[str(fixture_id)] = {
            "home": game.get("home"), "away": game.get("away"),
            "home_goals": int(game["home_goals"]),
            "away_goals": int(game["away_goals"]),
        }
    return out
