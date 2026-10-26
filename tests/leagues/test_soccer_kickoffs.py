import json

from scripts import soccer_kickoffs


def test_collects_league_and_ucl_kickoffs(tmp_path):
    (tmp_path / "data" / "leagues").mkdir(parents=True)
    (tmp_path / "data" / "ucl").mkdir(parents=True)
    (tmp_path / "data" / "leagues" / "pl.json").write_text(json.dumps(
        {"matches": [{"date": "2026-10-31T12:30:00+00:00"}, {"date": "2026-10-31"}]}))
    (tmp_path / "data" / "ucl" / "board.json").write_text(json.dumps(
        {"matches": [{"kickoff": "2026-11-03T20:00:00+00:00"}, {"kickoff": None}]}))
    out = tmp_path / "kickoffs.json"
    soccer_kickoffs.main(tmp_path, out)
    assert json.loads(out.read_text()) == ["2026-10-31T12:30:00+00:00", "2026-11-03T20:00:00+00:00"]


def test_empty_boards_keep_previous_file(tmp_path):
    out = tmp_path / "kickoffs.json"
    out.write_text('["x"]\n')
    soccer_kickoffs.main(tmp_path, out)
    assert out.read_text() == '["x"]\n'
