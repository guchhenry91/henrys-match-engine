"""Reconcile a player's nflverse club with the current API-NFL roster.

nflverse says where a man last PLAYED. The roster snapshot says where he IS. Those
agree all season and disagree all summer, which is exactly when this board has to
be right: the 2026 season opens on 10 September and the offseason moved real
players.

THE SNAPSHOT CORROBORATES, IT DOES NOT CONVICT. That distinction is the whole file
and it was learned expensively on the soccer side, where treating thin rosters as
proof deleted Real Madrid, Barcelona, PSG and 14 of 18 Bundesliga clubs because a
free feed happened to list fewer than 18 names for them.

  * Found on exactly one COMPLETE roster -> that is his club, even if nflverse
    disagrees. This is the reassignment that makes the board correct in September.
  * Found on no complete roster, but every roster is complete -> he is not in the
    league any more, and is dropped.
  * Any roster thin, missing or failed -> KEEP the nflverse attribution and flag
    it. Absence from incomplete evidence is not evidence of absence.
  * Found on two rosters -> keep nflverse's and flag. Two clubs listing one man
    is a feed artefact during camp cuts, and guessing between them is how a
    projection ends up on the wrong team while looking certain.
"""
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data-raw" / "nfl" / "rosters.json"

# Suffixes both feeds spell inconsistently. "Travis Etienne" and "Travis Etienne
# Jr." are one player; a strict key would call them two and drop the real one.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_key(name: str) -> str:
    """Accent, case, punctuation and suffix-insensitive identity key.

    Deliberately strict apart from those: a loose key here does not mislabel a
    player, it moves a projection onto a stranger's team.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    parts = [p for p in "".join(c if c.isalnum() or c.isspace() else " "
                                for c in text).split() if p]
    while parts and parts[-1] in _SUFFIXES:
        parts.pop()
    return "".join(parts)


def load() -> dict:
    try:
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def index(snapshot: dict) -> tuple[dict, bool]:
    """(name key -> [team codes on COMPLETE rosters], every roster complete?).

    Only complete rosters contribute. A thin one cannot prove a player is absent,
    so letting it into the index would make absence look like departure.
    """
    teams = (snapshot or {}).get("teams") or {}
    if not teams:
        return {}, False
    lookup = {}
    for code, entry in teams.items():
        if not entry.get("complete"):
            continue
        for player in entry.get("players") or []:
            lookup.setdefault(name_key(player), []).append(code)
    all_complete = bool(teams) and all(e.get("complete") for e in teams.values())
    all_complete = all_complete and not (snapshot.get("failed_teams") or [])
    return lookup, all_complete


def reconcile(player: str, nflverse_team: str, lookup: dict,
              all_complete: bool) -> tuple[str | None, str]:
    """Return (team or None to drop, reason).

    None means "do not publish him": either he is on no current roster and the
    evidence is complete enough to say so, or the snapshot is unusable and we are
    not guessing.
    """
    if not lookup:
        # No usable snapshot at all -- fall back to nflverse and say so, rather
        # than dropping a whole board because one fetch failed.
        return nflverse_team, "no roster snapshot; using last appearance"

    found = lookup.get(name_key(player)) or []
    if len(found) == 1:
        team = found[0]
        if team == nflverse_team:
            return team, "confirmed"
        return team, f"moved: {nflverse_team} -> {team}"
    if len(found) > 1:
        # Two clubs listing one man happens during camp cuts. Guessing is how a
        # projection lands on the wrong team while looking certain.
        return nflverse_team, f"listed by {len(found)} teams; kept last appearance"
    if all_complete:
        return None, "on no current roster"
    return nflverse_team, "rosters incomplete; kept last appearance"
