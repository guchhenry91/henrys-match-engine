"""A player's LAST FIVE games, as numbers, for the prop cards in every sport.

The actual games rather than an average: "0, 3, 0, 1, 4" and "1.6 average" are
different players, and only the first says whether an over is a habit or a hope.
Read from the raw game rows (not the model's training frame, whose role filters
drop some games), so the strip is what the player actually did.
"""
from __future__ import annotations

import pandas as pd


def last_n(rows: pd.DataFrame, ids, stat: str, n: int = 5, date_col: str = "game_date",
           id_col: str = "PLAYER_ID", played: str | None = None) -> dict:
    """{id: [stat, ...]} oldest first, for the ids asked about. `played` names a
    column that must be > 0 for a game to count (NBA minutes), so a DNP row is
    not shown as a zero."""
    if rows is None or rows.empty or stat not in rows or date_col not in rows:
        return {}
    want = rows[rows[id_col].isin(set(ids))]
    want = want[want[stat].notna()]
    if played and played in want:
        want = want[pd.to_numeric(want[played], errors="coerce").fillna(0) > 0]
    want = want.sort_values(date_col)
    return {k: [round(float(v), 1) for v in g[stat].tail(n)]
            for k, g in want.groupby(id_col, sort=False)}
