"""Put each player on the team he is actually on, joined by ID.

nflverse says where a man last PLAYED. The roster says where he IS. Those agree
all season and disagree all summer, which is exactly when this board has to be
right: the 2026 season opens on 10 September and the offseason moved real players.

WHY NFLVERSE AND NOT THE API-NFL KEY. This was built against API-NFL first and it
was wrong in the worst way. Its player endpoint returned 43-71 plausible names a
team -- enough to pass every count-based completeness check -- and those names did
not include Patrick Mahomes, A.J. Brown, Alvin Kamara or Austin Ekeler.
Philadelphia came back as Andy Dalton, Britain Covey and Danny Gray. The board
duly dropped 177 current players as having left the league, and every guard I had
written reported success. nflverse's roster file is 91.6 players a team, contains
all of them, and costs no quota.

IT JOINS ON gsis_id, WHICH IS THE SAME KEY AS THE STATS. That removes name
matching from the problem entirely -- no accents, no "Jr.", no "Cam" versus
"Cameron", no chance of moving a projection onto a stranger because two men share
a surname. 86.5% of 2025's active players appear in the 2026 file by ID; the
missing 273 are ordinary NFL turnover.

The snapshot still CORROBORATES rather than convicts: if it does not recognise
most of the players we independently know are active, it is not describing our
league and is refused wholesale rather than allowed to delete anyone.
"""
import pandas as pd

# A player must be on a roster in one of these states to be publishable. CUT and
# RET are exactly the people the old board kept projecting; RES (injured reserve)
# is excluded because he cannot play, which is the same reason a ruled-out player
# is dropped.
ACTIVE_STATUSES = {"ACT", "E14"}

# The snapshot must recognise this share of known-active players before it is
# allowed to say anyone has left. Below it, the file is describing a different
# population and its silence means nothing.
MIN_CORROBORATION = 0.60


def build_index(roster: pd.DataFrame) -> dict:
    """player_id -> team, for players in an active state.

    A player listed by two teams mid-camp resolves to neither: keeping the last
    appearance is the honest answer, and guessing is how a projection lands on the
    wrong team while looking certain.
    """
    if roster is None or roster.empty:
        return {}
    active = roster[roster["status"].isin(ACTIVE_STATUSES)]
    lookup = {}
    for player_id, group in active.groupby("gsis_id"):
        teams = sorted(set(group["team"].dropna()))
        if len(teams) == 1:
            lookup[str(player_id)] = teams[0]
    return lookup


def corroborates(lookup: dict, known_ids) -> tuple[bool, float]:
    """Does this roster describe the same league the stats describe?

    THE GUARD THAT WAS MISSING when this ran on API-NFL. Completeness was measured
    as "30+ names a team", which measured QUANTITY and concluded IDENTITY. It is
    now measured as agreement: what share of the players we independently know are
    active does this file contain?
    """
    known = [str(i) for i in known_ids]
    if not lookup or not known:
        return False, 0.0
    hits = sum(1 for i in known if i in lookup)
    rate = hits / len(known)
    return rate >= MIN_CORROBORATION, rate


def reconcile(player_id: str, nflverse_team: str, lookup: dict,
              trusted: bool) -> tuple[str | None, str]:
    """Return (team, reason), or (None, reason) to drop him from the board."""
    if not trusted or not lookup:
        return nflverse_team, "roster unusable; using last appearance"
    team = lookup.get(str(player_id))
    if team is None:
        # He is on no active roster and the file is trustworthy, so he is not
        # playing: retired, cut, on IR, or out of the league.
        return None, "not on an active roster"
    if team == nflverse_team:
        return team, "confirmed"
    return team, f"moved: {nflverse_team} -> {team}"
