"""Soccer price archive: one snapshot per change (leagues/odds_history.py)."""
from datetime import datetime, timezone

import pandas as pd

from leagues import odds_history as oh


def _row(h):
    return {"Div": "E0", "Date": "10/10/2026", "Time": "15:00", "HomeTeam": "Arsenal",
            "AwayTeam": "Leeds", "B365H": h, "B365D": 5.0, "B365A": 9.0, "BFEH": 1.42,
            "BFED": 5.2, "BFEA": 8.0, "AvgH": 1.40, "AvgD": 4.9, "AvgA": 8.5, "AHh": -1.25}


def test_snapshots_only_on_change_and_never_after_kickoff(tmp_path, monkeypatch):
    monkeypatch.setattr(oh, "canonical", lambda n, lg: n)
    path = tmp_path / "h.json"
    t1 = datetime(2026, 10, 8, 9, tzinfo=timezone.utc)
    assert oh.record(pd.DataFrame([_row(1.40)]), path, now=t1) == 1
    assert oh.record(pd.DataFrame([_row(1.40)]), path, now=t1.replace(hour=10)) == 0   # unchanged
    assert oh.record(pd.DataFrame([_row(1.36)]), path, now=t1.replace(hour=11)) == 1   # moved
    after = datetime(2026, 10, 10, 15, 30, tzinfo=timezone.utc)
    assert oh.record(pd.DataFrame([_row(1.20)]), path, now=after) == 0                 # kicked off
    import json
    snaps = json.loads(path.read_text())["fixtures"]["PL|2026-10-10|Arsenal|Leeds"]["snapshots"]
    assert [s["b365"]["home"] for s in snaps] == [1.4, 1.36]
