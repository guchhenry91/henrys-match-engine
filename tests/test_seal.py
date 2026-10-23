"""Tamper-evident pick logs (tracking/seal.py)."""
import json

from tracking import seal


def _write(p, log):
    p.write_text(json.dumps(log))


def _pick(**kw):
    e = {"kind": "prop", "market": "points", "player": "A", "side": "under", "line": 18.5,
         "probability": 0.56, "locked_at": "2026-10-23T22:00:00+00:00", "tipoff": "2026-10-23T23:10:00Z"}
    e.update(kw)
    return e


def test_grading_and_closing_fields_do_not_break_the_seal(tmp_path):
    p = tmp_path / "picks_log.json"
    _write(p, {"G:points:1": _pick()})
    assert seal.seal(p) == 1 and seal.seal(p) == 0
    log = json.loads(p.read_text())
    log["G:points:1"].update(graded="correct", actual=12, closing={"p": 0.5}, tipoff="2026-10-23T23:40:00Z")
    _write(p, log)
    v = seal.verify(p)
    assert v["sealed"] == 1 and not v["broken"] and not v["missing"]


def test_an_edited_line_or_chance_is_caught(tmp_path):
    p = tmp_path / "picks_log.json"
    _write(p, {"G:points:1": _pick(), "G:points:2": _pick(player="B", locked_at="2026-10-23T22:01:00+00:00")})
    seal.seal(p)
    log = json.loads(p.read_text())
    log["G:points:1"]["line"] = 20.5
    _write(p, log)
    assert seal.verify(p)["broken"] == ["G:points:1"]


def test_a_deleted_pick_is_reported_and_nested_nfl_logs_are_sealed(tmp_path):
    p = tmp_path / "picks_log.json"
    _write(p, {"games": {"2026_07_NE_CHI": {"pick": "NE", "p_pick": 0.6, "locked_at": "x"}},
               "props": {"k1": _pick()}})
    assert seal.seal(p) == 2
    log = json.loads(p.read_text())
    del log["props"]["k1"]
    _write(p, log)
    assert seal.verify(p)["missing"] == ["props/k1"]


def test_a_released_and_relocked_pick_is_a_new_version_not_an_edit(tmp_path):
    p = tmp_path / "picks_log.json"
    _write(p, {"m1": _pick()})
    seal.seal(p)
    _write(p, {"m1": _pick(line=19.5, locked_at="2026-10-24T20:00:00+00:00")})   # kickoff moved
    assert seal.seal(p) == 1
    v = seal.verify(p)
    assert not v["broken"] and v["superseded"] == 1 and v["sealed"] == 2
