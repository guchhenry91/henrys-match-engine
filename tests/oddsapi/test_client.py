"""The Odds API client must enforce the agreed budget BEFORE a credit is spent.

Budget agreed 2026-09-26: off until switched on, 200 credits a run, 1,000 a day,
hard stop at 40,000 used a month, historical (10x) only with an approved
allowance, and the key -- which travels in the URL -- never printed.
"""
import io
import json
import urllib.error
from datetime import datetime, timezone

import pytest

from oddsapi import client as oc

KEY = "sekret-key-123"
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


class _Resp(io.BytesIO):
    def __init__(self, body, headers):
        super().__init__(json.dumps(body).encode())
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Fake:
    """Records every request and answers with a set charge and account usage."""
    def __init__(self, last=3, used=100, remaining=49_900, body=None):
        self.urls, self.last, self.used, self.remaining = [], last, used, remaining
        self.body = body if body is not None else []

    def __call__(self, request, timeout=None):
        self.urls.append(request.full_url)
        free = request.full_url.split("?")[0].endswith(("/sports", "/events"))
        charge = 0 if free else self.last
        self.used += charge
        self.remaining -= charge
        return _Resp(self.body, {"x-requests-last": str(charge),
                                 "x-requests-used": str(self.used),
                                 "x-requests-remaining": str(self.remaining)})


def _paths(fake):
    return [u.split("?")[0] for u in fake.urls]


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("ODDS_API_ENABLED", "true")
    monkeypatch.setenv("ODDS_API_KEY", KEY)


def _client(tmp_path, fake, **kw):
    return oc.Client(opener=fake, ledger_path=tmp_path / "ledger.json",
                     clock=lambda: NOW, sleeper=lambda s: None, **kw)


def _props(c, est=3):
    return c.get("sports/basketball_nba/events/abc/odds", sport="nba",
                 purpose="lock check", est=est, regions="us",
                 markets="player_points,player_rebounds,player_assists")


def test_it_ships_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_ENABLED", "false")
    fake = Fake()
    with pytest.raises(oc.Disabled):
        _client(tmp_path, fake).get("sports", sport="-", purpose="x")
    assert fake.urls == []


def test_no_key_means_off(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY")
    assert not oc.enabled()


def test_cost_is_markets_times_regions_and_ten_books_are_a_region():
    assert oc.cost(3) == 3
    assert oc.cost(4, regions=2) == 8
    assert oc.cost(4, bookmakers=10) == 4
    assert oc.cost(4, bookmakers=11) == 8


def test_free_endpoints_are_recognised():
    assert oc._is_free("sports") and oc._is_free("sports/americanfootball_nfl/events")
    assert not oc._is_free("sports/americanfootball_nfl/events/x/odds")
    assert not oc._is_free("sports/americanfootball_nfl/odds")


def test_a_paid_call_must_state_its_cost(tmp_path):
    with pytest.raises(ValueError):
        _client(tmp_path, Fake()).get("sports/x/odds", sport="x", purpose="y")


def test_first_paid_call_reads_usage_for_free_then_records_the_charge(tmp_path):
    fake = Fake(last=3, used=500)
    c = _client(tmp_path, fake)
    _props(c)
    assert _paths(fake)[0].endswith("/sports")                  # free usage check
    assert c.run_used == 3 and c.day_used() == 3
    assert c.month_used() == 503
    led = json.loads((tmp_path / "ledger.json").read_text())
    assert led["days"]["2026-10-03"]["by_sport"]["nba"] == 3
    assert led["calls"][-1]["purpose"] == "lock check"


def test_run_cap_refuses_before_sending(tmp_path):
    fake = Fake(last=100)
    c = _client(tmp_path, fake)
    _props(c, est=100)
    _props(c, est=100)
    sent = len(fake.urls)
    with pytest.raises(oc.BudgetExceeded, match="run cap"):
        _props(c, est=1)
    assert len(fake.urls) == sent


def test_day_cap_counts_every_run_today(tmp_path):
    (tmp_path / "ledger.json").write_text(json.dumps(
        {"days": {"2026-10-03": {"credits": 999, "by_sport": {}}}}))
    fake = Fake()
    with pytest.raises(oc.BudgetExceeded, match="day cap"):
        _props(_client(tmp_path, fake), est=3)
    assert all(p.endswith("/sports") for p in _paths(fake))    # only the free check


def test_monthly_hard_stop_uses_the_accounts_own_count(tmp_path):
    fake = Fake(used=39_999)
    with pytest.raises(oc.BudgetExceeded, match="hard stop"):
        _props(_client(tmp_path, fake), est=3)
    assert len(fake.urls) == 1                                  # the free check only


def test_a_stale_ledger_does_not_block_after_the_billing_reset(tmp_path):
    """Ledger last saw 40,000 used; the account has since reset to 0."""
    (tmp_path / "ledger.json").write_text(json.dumps({"account_used": 40_000}))
    fake = Fake(used=0)
    _props(_client(tmp_path, fake))
    assert len(fake.urls) == 2


def test_historical_needs_an_approved_allowance(tmp_path):
    with pytest.raises(oc.BudgetExceeded, match="historical"):
        _client(tmp_path, Fake()).get("historical/sports/x/odds", sport="x",
                                      purpose="backtest", est=10)
    c = _client(tmp_path, Fake(last=10), historical_allowance=10)
    c.get("historical/sports/x/odds", sport="x", purpose="backtest", est=10)
    with pytest.raises(oc.BudgetExceeded, match="historical"):
        c.get("historical/sports/x/odds", sport="x", purpose="backtest", est=10)


def test_dry_run_sends_nothing_and_reports_the_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_ENABLED", "false")        # dry runs work while off
    fake = Fake()
    c = _client(tmp_path, fake, dry_run=True)
    _props(c, est=3)
    assert fake.urls == [] and c.planned[0]["est"] == 3
    assert "none spent" in c.report()


def test_the_key_never_appears_in_errors_or_the_ledger(tmp_path):
    def boom(request, timeout=None):
        raise urllib.error.URLError(f"failed {request.full_url}")
    with pytest.raises(RuntimeError) as err:
        _client(tmp_path, boom).get("sports", sport="-", purpose="x")
    assert KEY not in str(err.value)
    _props(_client(tmp_path, Fake()))
    assert KEY not in (tmp_path / "ledger.json").read_text()
    assert oc._scrub(f"https://x/?apiKey={KEY}&a=1", KEY).count("***") == 1


def test_http_errors_are_not_retried_but_a_rate_limit_waits_once(tmp_path):
    calls = []

    def err(code):
        def open_it(request, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, code, "x", {}, None)
        return open_it
    with pytest.raises(RuntimeError, match="HTTP 422"):
        _client(tmp_path, err(422)).get("sports", sport="-", purpose="x")
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(RuntimeError, match="HTTP 429"):
        _client(tmp_path, err(429)).get("sports", sport="-", purpose="x")
    assert len(calls) == 2


def test_warnings_fire_once_at_10k_and_30k_and_for_a_heavy_day(tmp_path):
    fake = Fake(used=9_998, last=3)
    c = _client(tmp_path, fake)
    _props(c)
    _props(c)
    assert sum("10,000" in w for w in c.warnings) == 1
    fake.used = 30_001
    _props(c)
    assert any("30,000" in w for w in c.warnings)
    (tmp_path / "ledger.json").write_text(json.dumps(
        {"days": {"2026-10-03": {"credits": 599, "by_sport": {}}}}))
    c2 = _client(tmp_path, Fake(used=5, last=3))
    _props(c2)
    assert any("today" in w for w in c2.warnings)


def test_the_limits_are_the_agreed_ones():
    assert (oc.RUN_CAP, oc.DAY_CAP, oc.MONTH_HARD_STOP) == (200, 1_000, 40_000)
    assert oc.MONTH_WARN_AT == (10_000, 30_000) and oc.DAY_WARN_AT == 600


def test_ladder_up_climbs_from_the_main_line_and_never_beats_it():
    from oddsapi import props
    alt = {"book": "DraftKings", "ladder": [[0.5, 1.30], [1.5, 3.2], [2.5, 9.0], [3.5, 40.0]]}
    pick = {"line": 0.5, "probability": 0.66, "book_p": 0.64, "book_price": 1.48, "side": "over"}
    model = {1.5: 0.24, 2.5: 0.06}
    up = props.ladder_up(alt, pick, lambda line: model.get(line))
    assert [r["line"] for r in up] == [1.5, 2.5]          # 3.5 is ~2%, below the floor
    assert up[0]["p"] == 0.24 and all(r["p"] <= 0.66 for r in up)
    assert props.ladder_up(alt, dict(pick, side="under"), lambda l: 0.2) == []
