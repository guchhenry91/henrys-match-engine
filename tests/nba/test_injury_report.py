import io
from datetime import datetime, timezone

import pandas as pd

from nba import injury_report as ir


def test_slots_newest_first_both_formats():
    t = datetime(2026, 1, 15, 17, 40, tzinfo=ir.ET)
    s = ir.slots_back_from(t)
    assert s[:4] == ["05_30PM", "05_15PM", "05_00PM", "05PM"]
    assert s[-1] == "11AM" and "12PM" in s


def test_before_first_report_nothing_is_fetched():
    calls = []
    now = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)      # 9am ET
    assert ir.latest(now, opener=lambda *a, **k: calls.append(a)) is None
    assert calls == []


def test_clean_strips_matchup_and_team():
    assert ir.clean("BOS@NYK Boston Celtics Tatum, Jayson") == "Tatum, Jayson"


def test_refresh_writes_today_into_the_season(tmp_path, monkeypatch):
    wanted = "Injury-Report_2026-01-15_05_15PM.pdf"

    def opener(url, timeout=0):
        if url.endswith(wanted):
            return io.BytesIO(b"%PDF")
        raise OSError("404")

    monkeypatch.setattr(ir, "parse", lambda pdf: [("Tatum, Jayson", "Out"), ("Brown, Jaylen", "Questionable")])
    pd.DataFrame([{"date": "2026-01-15", "player": "Old, Row", "status": "Out"},
                  {"date": "2026-01-14", "player": "Kept, Row", "status": "Out"}]).to_csv(
        tmp_path / "2026.csv.gz", index=False, compression="gzip")
    now = datetime(2026, 1, 15, 22, 20, tzinfo=timezone.utc)     # 5:20pm ET
    out = ir.refresh(2026, now=now, opener=opener, base=tmp_path)
    assert out == ["Tatum, Jayson"]
    f = pd.read_csv(tmp_path / "2026.csv.gz")
    assert set(f["player"]) == {"Kept, Row", "Tatum, Jayson", "Brown, Jaylen"}   # today's rows replaced


def test_refresh_never_fatal(tmp_path):
    def opener(url, timeout=0):
        raise OSError("down")
    now = datetime(2026, 1, 15, 22, 20, tzinfo=timezone.utc)
    assert ir.refresh(2026, now=now, opener=opener, base=tmp_path) == []
