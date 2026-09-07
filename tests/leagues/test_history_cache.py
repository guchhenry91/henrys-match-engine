"""football-data.co.uk is fetched once per finished season, then never again.

WHY THIS EXISTS. `fetch_history` had no cache: every publish pulled 5 seasons x
5 leagues = 25 CSVs, and at a run every half hour that is roughly 1,200 requests
a day at a small free site for files that cannot change. On 2026-09-05 the site
began answering 503 to GitHub's runners and the publish failed for two days --
predictions frozen at 17-23 hours old while results stayed current, because the
fast lock path needs no history.

AN actions/cache ENTRY WOULD NOT HAVE FIXED IT. That cache only saves when a job
SUCCEEDS, and no job was succeeding: cold run -> 25 requests -> 503 -> failure ->
nothing saved -> cold run. The files are committed instead, so CI reads them out
of the checkout and asks the network for nothing.
"""
import urllib.error

import pytest

from leagues import config, history


class Boom:
    """A network that refuses, like the one that caused this."""
    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        raise urllib.error.HTTPError("https://www.football-data.co.uk/x.csv",
                                     503, "Service Temporarily Unavailable", {}, None)


def test_a_finished_season_never_touches_the_network(monkeypatch):
    """THE ONE THAT ENDS THE OUTAGE. Every configured history season is
    finished, so a publish should make zero football-data requests."""
    boom = Boom()
    monkeypatch.setattr(history.urllib.request, "urlopen", boom)
    for league, lg in config.LEAGUES.items():
        for season in lg.history_seasons:
            text = history.fetch_csv(season, lg.fd_code)
            assert text and "HomeTeam" in text, f"{league} {season} came back empty"
    assert boom.calls == 0, (
        f"{boom.calls} network calls for finished seasons -- they are cached")


def test_every_configured_season_is_actually_committed():
    """A cache that is not in the repo is not a cache in CI: the runner starts
    from a fresh checkout every time."""
    missing = [(lg.fd_code, s) for lg in config.LEAGUES.values()
               for s in lg.history_seasons
               if not history._cache_path(s, lg.fd_code).exists()]
    assert not missing, f"not cached: {missing}"


def test_no_configured_season_is_the_one_in_progress():
    """The cache is permanent for finished seasons only. If a season in progress
    ever enters history_seasons it must NOT be served from a frozen file, and
    this test is the alarm for that."""
    live = history.current_fd_season()
    bad = [(k, s) for k, lg in config.LEAGUES.items()
           for s in lg.history_seasons if s == live]
    assert not bad, (
        f"{bad} is the season in progress; results still land in it, so it "
        f"cannot be cached permanently")


@pytest.mark.parametrize("when,expected", [
    ("2026-09-07", "2627"),   # mid-season
    ("2026-06-30", "2526"),   # June still belongs to the season that started in 2025
    ("2026-07-01", "2627"),   # July starts the new one
    ("2027-01-15", "2627"),   # January is still the season that began in August
])
def test_the_season_code_rolls_over_at_midyear_not_new_year(when, expected):
    """A European season is named by both years, so a calendar-year rule would
    call January's fixtures the previous season and refetch the wrong file."""
    import datetime as dt
    assert history.current_fd_season(dt.datetime.fromisoformat(when)) == expected


def test_a_failed_fetch_of_a_live_season_falls_back_loudly(monkeypatch, capsys, tmp_path):
    """Stale data that SAYS it is stale beats no board. Silence is how a fixture
    list quietly goes a week out of date."""
    monkeypatch.setattr(history, "CACHE", tmp_path)
    live = history.current_fd_season()
    path = history._cache_path(live, "E0")
    path.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n", encoding="latin-1")
    monkeypatch.setattr(history.urllib.request, "urlopen", Boom())

    text = history.fetch_csv(live, "E0")
    assert "HomeTeam" in text
    assert "WARNING" in capsys.readouterr().out


def test_a_failed_fetch_with_no_cache_raises(monkeypatch, tmp_path):
    """Never invent history. With nothing cached and nothing fetched there is no
    honest answer, and publish's own abort keeps the league's last good file."""
    monkeypatch.setattr(history, "CACHE", tmp_path)
    monkeypatch.setattr(history.urllib.request, "urlopen", Boom())
    with pytest.raises(urllib.error.HTTPError):
        history.fetch_csv(history.current_fd_season(), "E0")


def test_the_cached_history_still_parses_into_the_model_frame():
    """The cache stores the source CSV verbatim, so the parser must be unchanged
    by it -- same rows, same odds coverage."""
    frame = history.fetch_history("PL")
    assert len(frame) == 1900
    assert sorted(frame["season"].unique()) == ["2122", "2223", "2324", "2425", "2526"]
    assert frame["odds_h"].notna().all()
