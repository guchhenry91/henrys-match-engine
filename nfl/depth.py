"""Who starts, and who is only on the roster.

THE HOLE THIS CLOSES was stated on the board for weeks rather than fixed: "no
depth charts -- a backup quarterback carries a low line and can top a market he
may not play in". It is not hypothetical. On the 2026 week 1 board, six of
nineteen passing picks were backups; Marcus Mariota (WAS QB2) was the second
highest passing pick on the board at 62%, and Cleveland published its QB2 and QB3
while its starter appeared nowhere.

The mechanism is a trap rather than a bug. A backup's line is his own entering
median, and he has only ever played in relief, so his median is low -- 150 to 180
passing yards against a starter's 210 to 265. "Over" against a low line looks
easy, and the model is right that he would beat it IF HE PLAYED. The thing the
model cannot see is that he will not take a snap.

IT ALSO ANSWERS A STALENESS QUESTION THE ROSTER CANNOT. Four days after the cut
to 53, nflverse's season roster still listed 90 active players a team, and so did
its weekly file; neither could say who had been cut. The depth chart is
republished continuously -- on 2026-08-30 it carried a snapshot from 12:30 that
morning -- and only holds players a team is actually carrying.

THE CORROBORATION GATE IS THE POINT, not decoration. `rosters.corroborates` exists
because an API roster with 43-71 plausible names a team passed every count-based
completeness check while omitting Patrick Mahomes, A.J. Brown and Alvin Kamara,
and the board silently dropped 177 current players. A depth chart that does not
recognise most of the board's own players is far more likely to be broken than to
be evidence that the board is wrong, so below `MIN_DEPTH_COVERAGE` it is ignored
wholesale and the board says the gate did not run.
"""
from nfl import config


def build_index(chart) -> dict:
    """gsis_id -> {"pos": str, "rank": int}. Empty dict if there is no chart."""
    if chart is None or len(chart) == 0:
        return {}
    out = {}
    for _, row in chart.iterrows():
        rank = row.get("pos_rank")
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            continue                      # a rank we cannot read is not a rank
        out[str(row["gsis_id"])] = {"pos": row.get("pos_abb"), "rank": rank}
    return out


def coverage(index: dict, player_ids) -> float:
    """Share of the board's own players the chart recognises."""
    ids = [str(p) for p in player_ids]
    if not index or not ids:
        return 0.0
    return sum(1 for i in ids if i in index) / len(ids)


def usable(index: dict, player_ids) -> tuple[bool, float, int]:
    """(trusted, coverage, teams-ish size). See the corroboration note above."""
    rate = coverage(index, player_ids)
    return rate >= config.MIN_DEPTH_COVERAGE, rate, len(index)


def verdict(market: str, entry: dict | None) -> tuple[bool, str]:
    """Should this player be published in this market, and why not if not.

    A player the chart does not list at all is KEPT, not dropped. Absence from the
    chart is not evidence he was cut -- it is equally evidence the chart is thin at
    his position -- and dropping on absence is precisely how 177 current players
    were deleted in August. Only an explicit rank too far down removes him.
    """
    cap = config.MAX_DEPTH_RANK.get(market)
    if cap is None or entry is None:
        return True, ""
    rank = entry.get("rank")
    if rank is None:
        return True, ""
    if rank <= cap:
        return True, ""
    pos = entry.get("pos") or "?"
    if market == "passing_yards":
        why = (f"{pos}{rank}: one quarterback takes essentially every drop-back, "
               f"so a backup is usually none of this market rather than a "
               f"smaller share of it")
    else:
        why = f"{pos}{rank}: too far down the depth chart to expect real snaps"
    return False, why


def annotate(pick: dict, entry: dict | None) -> dict:
    """Attach the depth position to a published card, so a reader can see it.

    Published even when the player passes the gate: "QB1" and "WR3" are exactly
    the context that tells someone whether a low line is a soft spot or a warning.
    """
    pick["depth_pos"] = None if entry is None else entry.get("pos")
    pick["depth_rank"] = None if entry is None else entry.get("rank")
    pick["depth_label"] = (None if entry is None
                           else f"{entry.get('pos')}{entry.get('rank')}")
    return pick
