"""A board file must parse in a BROWSER, which rejects NaN/Infinity that Python
accepts -- the MLB tab went blank on 2026-09-27 over one "pitcher": NaN."""
import json

from scripts import json_safe, validate_data_json


def test_nan_and_infinity_become_null():
    out = json.loads(json_safe.dumps({"a": float("nan"), "b": [float("inf"), 1.5]}))
    assert out == {"a": None, "b": [None, 1.5]}


def test_the_validator_rejects_nan_like_a_browser(tmp_path, monkeypatch):
    bad = tmp_path / "data" / "x.json"
    bad.parent.mkdir()
    bad.write_text('{"pitcher": NaN}', encoding="utf-8")
    monkeypatch.setattr(validate_data_json, "ROOT", tmp_path)
    assert validate_data_json.check(["data/x.json"]) == 1
