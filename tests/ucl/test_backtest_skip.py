"""The UCL gate re-runs only when the finished matches or its code changed."""
import json

import pandas as pd

from scripts import ucl_backtest


def _frame(n=3):
    return pd.DataFrame({"date": ["2025-01-0%d" % (i + 1) for i in range(n)], "home": ["A"] * n,
                         "away": ["B"] * n, "home_goals": [1] * n, "away_goals": [0] * n})


def test_skips_when_inputs_unchanged(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    monkeypatch.setattr(ucl_backtest, "OUT", out)
    monkeypatch.setattr(ucl_backtest.data, "matches", lambda: _frame())
    calls = []
    monkeypatch.setattr(ucl_backtest.backtest, "walk_forward", lambda f: calls.append(1) or f)
    monkeypatch.setattr(ucl_backtest.backtest, "evaluate",
                        lambda f: {"released": True, "overall": {"n": len(f)}})
    ucl_backtest.main()
    assert len(calls) == 1 and json.loads(out.read_text())["_inputs"]
    ucl_backtest.main()
    assert len(calls) == 1                      # nothing changed: report kept
    monkeypatch.setattr(ucl_backtest.data, "matches", lambda: _frame(4))
    ucl_backtest.main()
    assert len(calls) == 2                      # a new result re-runs the gate
    assert json.loads(out.read_text())["overall"]["n"] == 4


def test_old_report_without_fingerprint_reruns(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    out.write_text(json.dumps({"released": True}))
    monkeypatch.setattr(ucl_backtest, "OUT", out)
    assert not ucl_backtest.up_to_date(ucl_backtest.fingerprint(_frame()))
