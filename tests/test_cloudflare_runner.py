"""The Cloudflare job runner's safety rules (cloudflare/container/run_job.py).

The live data lives in Cloudflare D1, not the repo. These pin the rules that keep it
safe: an older copy pushed to the repo can never replace a live record, a failed
gate still keeps the credit ledger, and a deleted file stays deleted.
"""
import base64
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cloudflare" / "container"))
import jobs  # noqa: E402
import run_job  # noqa: E402


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class FakeWorker:
    """Just enough of the Worker's /internal/state API, in memory."""

    def __init__(self):
        self.files = {}            # path -> (bytes, repo_sha)
        self.deleted = []

    def api(self, method, path, body=None, **_):
        if path == "/internal/state/manifest":
            return [{"path": k, "sha": _sha(v), "repo_sha": r} for k, (v, r) in self.files.items()]
        if path.startswith("/internal/state/file?"):
            from urllib.parse import parse_qs, urlsplit
            q = parse_qs(urlsplit(path).query)
            got = self.files.get(q["path"][0])
            if not got:
                return None
            return {"chunks": 1, "sha": _sha(got[0]), "data": base64.b64encode(got[0]).decode()}
        if path == "/internal/state/file" and method == "PUT":
            self.files[body["path"]] = (base64.b64decode("".join(body["chunks"])), body["repo_sha"])
            return {"ok": True}
        if path == "/internal/state/delete":
            self.deleted.append(body["path"])
            self.files.pop(body["path"], None)
            return {"ok": True}
        return {"ok": True}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    fake = FakeWorker()
    monkeypatch.setattr(run_job, "CODE", tmp_path)
    monkeypatch.setattr(run_job, "api", fake.api)
    monkeypatch.setattr(run_job, "log", lambda m: None)

    def write(rel, text):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path, fake, write


def test_protected_covers_the_live_record_but_not_inputs():
    for path in ("data/leagues/best.json", "data-raw/leagues/pl/picks_log.json",
                 "data-raw/leagues/pl/player_picks_log.json", "data-raw/odds_api/ledger.json",
                 "data-raw/leagues/telegram_sent.json", "data-raw/leagues/parlays_log.json",
                 "data-raw/nfl/out_since.json"):
        assert run_job.PROTECTED.search(path), path
    for path in ("data-raw/leagues/news.json", "data-raw/leagues/transfers.json",
                 "data-raw/nfl/news.json", "data-raw/leagues/results_override.json"):
        assert not run_job.PROTECTED.search(path), path


def test_an_older_repo_copy_never_replaces_a_live_picks_log(repo):
    """THE ONE THAT PROTECTS THE RECORD. A local run pushed an old picks log."""
    root, fake, write = repo
    write("data-raw/leagues/pl/picks_log.json", '{"old": 1}')           # the repo's copy now
    fake.files["data-raw/leagues/pl/picks_log.json"] = (b'{"live": 2}', "sha-of-an-earlier-repo-copy")
    run_job.overlay_state()
    assert (root / "data-raw/leagues/pl/picks_log.json").read_text() == '{"live": 2}'


def test_a_named_override_is_the_only_way_to_replace_a_live_record(repo):
    root, fake, write = repo
    write("data-raw/leagues/pl/picks_log.json", '{"corrected": 1}')
    fake.files["data-raw/leagues/pl/picks_log.json"] = (b'{"live": 2}', "sha-of-an-earlier-repo-copy")
    write("cloudflare/state_overrides.json",
          '{"data-raw/leagues/pl/picks_log.json": "%s"}' % _sha(b'{"corrected": 1}'))
    run_job.overlay_state()
    assert (root / "data-raw/leagues/pl/picks_log.json").read_text() == '{"corrected": 1}'
    assert fake.files["data-raw/leagues/pl/picks_log.json"][0] == b'{"corrected": 1}'


def test_a_changed_input_in_the_repo_wins_and_is_stored(repo):
    root, fake, write = repo
    write("data-raw/leagues/news.json", '{"new": 1}')
    fake.files["data-raw/leagues/news.json"] = (b'{"old": 1}', "sha-of-an-earlier-repo-copy")
    run_job.overlay_state()
    assert (root / "data-raw/leagues/news.json").read_text() == '{"new": 1}'
    assert fake.files["data-raw/leagues/news.json"][0] == b'{"new": 1}'   # the site serves it too


def test_a_failed_gate_still_keeps_the_credit_ledger_and_nothing_else(repo):
    root, fake, write = repo
    write("data-raw/odds_api/ledger.json", "0")
    write("data/nba/board.json", "0")
    before = run_job.snapshot()
    write("data-raw/odds_api/ledger.json", "spent 8")
    write("data/nba/board.json", "half-built")
    after = run_job.save_changes(before, {}, only=jobs.KEEP_ON_FAILURE)
    assert "data-raw/odds_api/ledger.json" in fake.files
    assert "data/nba/board.json" not in fake.files
    # the unsaved board still counts as changed, so a later passing phase saves it
    assert after["data/nba/board.json"] != run_job.sha(b"half-built")


def test_a_deleted_file_is_deleted_from_d1_too(repo):
    root, fake, write = repo
    write("data-raw/nba/current/old.json", "x")
    before = run_job.snapshot()
    (root / "data-raw/nba/current/old.json").unlink()
    run_job.save_changes(before, {})
    assert fake.deleted == ["data-raw/nba/current/old.json"]


def test_odds_syncs_are_soft_like_the_workflows():
    """nba.yml / mlb.yml / nfl.yml ran these with continue-on-error."""
    steps = {cmd: kind for phases in jobs.JOBS.values() for phase in phases for _, cmd, kind in phase}
    for script in ("scripts.sync_nba_odds", "scripts.sync_mlb_odds", "scripts.sync_nfl_book_lines"):
        assert steps["python -m " + script] == "soft", script


def test_only_jobs_that_use_a_cache_restore_it():
    assert "cache/soccerdata.tar.gz" in jobs.CACHE_FOR["leagues"]
    for job in ("lock", "health", "nba", "mlb"):
        assert job not in jobs.CACHE_FOR
