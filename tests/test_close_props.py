import json
from datetime import datetime, timedelta, timezone

from scripts import close_props as cp

NOW = datetime(2026, 10, 25, 23, 0, tzinfo=timezone.utc)


class FakeClient:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def get(self, path, **kw):
        self.calls.append((path, kw))
        return self.payload


def _payload():
    def outs(name, line, o, u):
        return [{"name": "Over", "description": name, "point": line, "price": o},
                {"name": "Under", "description": name, "point": line, "price": u}]
    return {"bookmakers": [{"key": "pinnacle", "markets": [
        {"key": "player_points", "outcomes": outs("Jalen Brunson", 26.5, 1.80, 2.05)
                                             + outs("Josh Hart", 11.5, 1.95, 1.87)}]}]}


def _setup(tmp_path, monkeypatch, entries):
    from nba import book_lines as nba_bl
    log, store = tmp_path / "log.json", tmp_path / "store.json"
    log.write_text(json.dumps(entries))
    store.write_text(json.dumps({"games": {"g1": {"event_id": "ev1"}}}))
    monkeypatch.setattr(cp, "sports", lambda: {"nba": {
        "log": log, "section": None, "store": store, "bl": nba_bl, "start": "tipoff",
        "store_key": lambda e: str(e.get("game_id"))}})
    return log


def _prop(player, line, side, mins=10, **kw):
    return {"kind": "prop", "game_id": "g1", "market": "points", "player": player,
            "line": line, "side": side, "book_price": 1.91,
            "tipoff": (NOW + timedelta(minutes=mins)).isoformat(),
            **kw}


def test_same_line_stamps_a_scoreable_close(tmp_path, monkeypatch):
    log = _setup(tmp_path, monkeypatch, {"a": _prop("Jalen Brunson", 26.5, "over")})
    client = FakeClient(_payload())
    out = cp.run(client, now=NOW)
    assert out == {"nba": {"games": 1, "picks": 1}}
    e = json.loads(log.read_text())["a"]
    assert e["closing"]["moved"] == "same" and e["close_price"] == 1.80
    assert 0.5 < e["close_p"] < 0.6                      # de-vigged over
    assert client.calls[0][1]["markets"] == "player_points"
    from tracking import performance
    assert performance.clv(e)[0] in ("ours", "against", "flat")


def test_moved_line_records_direction_only(tmp_path, monkeypatch):
    log = _setup(tmp_path, monkeypatch, {"b": _prop("Josh Hart", 10.5, "over")})
    cp.run(FakeClient(_payload()), now=NOW)
    e = json.loads(log.read_text())["b"]
    assert e["closing"]["moved"] == "ours"                # 10.5 -> 11.5 on an over
    assert "close_p" not in e and "close_price" not in e


def test_only_due_unclosed_props_cost_a_call(tmp_path, monkeypatch):
    entries = {"far": _prop("Jalen Brunson", 26.5, "over", mins=90),
               "done": _prop("Josh Hart", 11.5, "under", closing={"at": "x"}),
               "graded": _prop("Josh Hart", 11.5, "under", graded="correct"),
               "winner": {"kind": "winner", "game_id": "g1", "tipoff": NOW.isoformat()}}
    log = _setup(tmp_path, monkeypatch, entries)
    before = log.read_text()
    client = FakeClient(_payload())
    assert cp.run(client, now=NOW) == {}
    assert client.calls == [] and log.read_text() == before


def test_nfl_store_key_is_home_then_away():
    assert cp._nfl_store_key("2026_04_PIT_CLE") == "CLE|PIT"
