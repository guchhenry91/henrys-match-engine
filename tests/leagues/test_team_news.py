import datetime as dt
import json

from leagues import team_news


NOW = dt.datetime(2026, 8, 21, 12, tzinfo=dt.timezone.utc)


def _rss(title, link, source):
    return f"""<rss><channel><item>
      <title>{title}</title><link>{link}</link>
      <pubDate>Fri, 21 Aug 2026 10:00:00 GMT</pubDate>
      <source>{source}</source>
    </item></channel></rss>""".encode()


def test_one_report_is_evidence_but_cannot_remove_a_player():
    evidence = [{
        "player": "Ada Striker", "status": "out", "publisher": "Paper One",
        "url": "https://one.example/a",
    }]
    assert team_news.corroborated(evidence, "out") == set()


def test_two_publishers_can_confirm_an_explicit_absence():
    evidence = [
        {"player": "Ada Striker", "status": "out", "publisher": "Paper One",
         "url": "https://one.example/a"},
        {"player": "Ada Striker", "status": "out", "publisher": "Paper Two",
         "url": "https://two.example/b"},
    ]
    assert team_news.corroborated(evidence, "out") == {"Ada Striker"}


def test_same_publisher_in_two_search_indexes_is_only_one_source():
    evidence = [
        {"player": "Ada Striker", "status": "out", "publisher": "The Paper",
         "feed": "google-news", "url": "https://google.example/a"},
        {"player": "Ada Striker", "status": "out", "publisher": "The Paper",
         "feed": "bing-news", "url": "https://bing.example/b"},
    ]
    assert team_news.corroborated(evidence, "out") == set()


def test_vague_injury_headline_does_not_change_model_input():
    assert team_news.classify(
        "Ada Striker injury update before City match", ["Ada Striker"]) is None
    assert team_news.classify(
        "Ada Striker ruled out of City match", ["Ada Striker"]
    ) == ("Ada Striker", "out")


def test_refresh_preserves_manual_news_and_records_provenance(tmp_path):
    best = tmp_path / "best.json"
    news = tmp_path / "news.json"
    leagues = tmp_path / "leagues"
    leagues.mkdir()
    best.write_text(json.dumps({"upcoming": [{
        "league_key": "PL", "date": "2026-08-21T19:00:00Z",
        "home": "Alpha", "away": "Beta",
    }]}))
    news.write_text(json.dumps({"PL": {
        "Alpha": {"out": ["Manual Player"], "doubt": []},
        "Beta": {"out": [], "doubt": []},
    }}))
    (leagues / "pl.json").write_text(json.dumps({"matches": [{
        "props": [
            {"team": "Alpha", "player": "Ada Striker"},
            {"team": "Beta", "player": "Ben Forward"},
        ]
    }]}))

    def fetcher(url):
        if "Alpha" in url:
            source = "Paper One" if "google" in url else "Paper Two"
            domain = "one.example" if "google" in url else "two.example"
            return _rss("Ada Striker ruled out of Beta match",
                        f"https://{domain}/story", source)
        return b"<rss><channel></channel></rss>"

    result = team_news.refresh(
        news_path=news, best_path=best, league_dir=leagues,
        fetcher=fetcher, now=NOW)
    alpha = result["PL"]["Alpha"]
    assert set(alpha["out"]) == {"Manual Player", "Ada Striker"}
    assert alpha["checked"] == "2026-08-21T12:00:00Z"
    assert len(alpha["automation"]["evidence"]) == 2
    assert alpha["automation"]["policy"] == "two-independent-publishers"


def test_partial_feed_failure_does_not_claim_a_fresh_check(tmp_path):
    best = tmp_path / "best.json"
    news = tmp_path / "news.json"
    leagues = tmp_path / "leagues"
    leagues.mkdir()
    best.write_text(json.dumps({"upcoming": [{
        "league_key": "PL", "date": "2026-08-21T19:00:00Z",
        "home": "Alpha", "away": "Beta",
    }]}))
    news.write_text(json.dumps({"PL": {}}))
    (leagues / "pl.json").write_text(json.dumps({"matches": []}))

    def fetcher(url):
        if "bing" in url:
            raise OSError("feed down")
        return b"<rss><channel></channel></rss>"

    result = team_news.refresh(
        news_path=news, best_path=best, league_dir=leagues,
        fetcher=fetcher, now=NOW)
    assert "checked" not in result["PL"]["Alpha"]
    assert result["PL"]["Alpha"]["automation"]["feeds_ok"] == ["google-news"]
