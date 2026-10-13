import json

from scripts import fetch_ucl_player_history as f


class Fake:
    def __init__(self):
        self.calls = []

    def get(self, path, **kw):
        self.calls.append(path)
        if path == "fixtures":
            if kw["season"] != 2024:
                return []
            row = lambda i, h, a, rnd: {"fixture": {"id": i, "date": "2024-10-01T19:00:00+00:00",
                                                   "status": {"short": "FT"}},
                                       "league": {"round": rnd},
                                       "teams": {"home": {"name": h}, "away": {"name": a}},
                                       "score": {"fulltime": {"home": 1, "away": 0}}}
            return [row(1, "Arsenal", "Bayern München", "League Stage - 2"),
                    row(2, "Arsenal", "Benfica", "League Stage - 3"),
                    row(3, "Arsenal", "Chelsea", "Play-offs")]
        return [{"team": {"name": "Arsenal"}, "players": [{"player": {"id": 9, "name": "B. Saka"},
                 "statistics": [{"games": {"minutes": 90, "substitute": False}, "shots": {"total": 3, "on": 1},
                                 "goals": {"total": None}}]}]}]


def test_fetches_only_big_five_pairs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "OUT", tmp_path / "h.json")
    c = Fake()
    f.main(client=c)
    saved = json.loads((tmp_path / "h.json").read_text())["fixtures"]
    assert list(saved) == ["1"]
    p = saved["1"]["players"][0]
    assert (p["shots"], p["sot"], p["goals"]) == (3, 1, 0)
    c2 = Fake()
    f.main(client=c2)
    assert "fixtures/players" not in c2.calls
