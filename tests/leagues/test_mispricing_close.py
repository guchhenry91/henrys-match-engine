"""Price edges checked against the sharp CLOSING price (leagues/mispricing.py)."""
import sys
import types

sys.modules.setdefault("penaltyblog", types.ModuleType("penaltyblog"))

from leagues import history, mispricing as m          # noqa: E402
from tracking import performance                       # noqa: E402

CSV = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365CH,B365CD,B365CA,BFECH,BFECD,BFECA,"
       "B365C>2.5,B365C<2.5,BFEC>2.5,BFEC<2.5,AHCh,B365CAHH,B365CAHA,BFECAHH,BFECAHA\n"
       "E0,10/10/2026,Arsenal,Leeds,2,0,1.30,5.5,10.0,1.33,5.6,11.0,"
       "1.70,2.15,1.74,2.24,-1.5,1.95,1.95,2.00,1.98\n")


def _edge(market, selection, b365, fair_p, line=None):
    return {"league_key": "PL", "home": "Arsenal", "away": "Leeds",
            "date": "2026-10-10T11:30:00+00:00", "market": market, "selection": selection,
            "line": line, "b365": b365, "exchange": 0, "fair_p": fair_p, "ev": 0.04,
            "book_price": b365, "book": "bet365"}


def test_an_edge_is_scored_against_the_exchange_close(tmp_path):
    closes = m.closing_rows({"PL": CSV})
    e = _edge("match_result", "home", 1.40, 0.72)
    assert m.stamp_close(e, closes[("PL", "2026-10-10", "Arsenal", "Leeds")])
    assert 0.73 < e["close_p"] < 0.74          # firmed on Arsenal: 72% at the flag
    assert e["close_ev"] > 0 and e["close_price"] == 1.30
    e["graded"] = "correct"
    assert performance.clv(e) == ("ours", e["close_ev"])


def test_a_moved_handicap_is_not_compared(tmp_path):
    closes = m.closing_rows({"PL": CSV})
    e = _edge("asian_handicap", "home", 2.05, 0.5, line=-1.25)
    assert not m.stamp_close(e, closes[("PL", "2026-10-10", "Arsenal", "Leeds")])
    assert "close_p" not in e


def test_update_stamps_the_close_and_grades(tmp_path):
    e = _edge("over_under_2.5", "under", 2.30, 0.46)
    log = tmp_path / "log.json"
    out = m.update([], {}, log_path=log)        # empty log first
    import json
    log.write_text(json.dumps({m.key(e): {**e, "first_seen": "x"}}))
    out = m.update([], {("PL", "2026-10-10", "Arsenal", "Leeds"): (2, 0)}, log_path=log,
                   closes=m.closing_rows({"PL": CSV}))
    s = out["settled"][0]
    assert s["graded"] == "correct" and "close_p" in s and "close_ev" in s


def test_the_live_season_file_is_downloaded_once_per_run(monkeypatch, tmp_path):
    calls = []

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): calls.append(1); return b"Div,Date\n"

    monkeypatch.setattr(history, "CACHE", tmp_path)
    monkeypatch.setattr(history, "_FETCHED", {})
    monkeypatch.setattr(history.urllib.request, "urlopen", lambda *a, **k: R())
    season = history.current_fd_season()
    history.fetch_csv(season, "E0")
    history.fetch_csv(season, "E0")
    assert len(calls) == 1
