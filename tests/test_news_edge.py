"""News edge: picks lifted by a teammate ruled out in the last few hours."""
import json

from tracking import news_edge as ne


def test_remember_keeps_the_first_time_and_forgets_players_back_in(tmp_path):
    path = tmp_path / "out.json"
    s1 = ne.remember(path, {"A": {"team": "KC"}, "B": {"team": "BUF"}}, now="2026-10-04T10:00:00+00:00")
    s2 = ne.remember(path, {"A": {"team": "KC"}}, now="2026-10-04T15:00:00+00:00")
    assert s2["A"]["since"] == s1["A"]["since"] == "2026-10-04T10:00:00+00:00"
    assert "B" not in s2 and "B" not in json.loads(path.read_text())["players"]


def test_only_fresh_news_on_a_lifted_over_is_an_edge():
    state = {"A": {"team": "KC", "since": "2026-10-04T10:00:00+00:00"},
             "C": {"team": "BUF", "since": "2026-10-01T10:00:00+00:00"}}
    picks = [{"team": "KC", "vacated": 0.2, "side": "over"},
             {"team": "KC", "vacated": 0.2, "side": "under"},
             {"team": "KC", "vacated": None, "side": "over"},
             {"team": "BUF", "vacated": 0.3, "side": "over"}]
    assert ne.annotate(picks, state, now="2026-10-04T14:00:00+00:00") == 1
    assert picks[0]["news_edge"] == {"out": ["A"], "hours_ago": 4.0}
    assert picks[3]["news_edge"] is None                 # 3 days old: the price has it
