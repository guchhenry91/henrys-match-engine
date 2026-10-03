"""The Cloudflare site folder holds only what the page serves (scripts/build_site.py)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_site_folder_has_the_pages_and_boards_and_nothing_private():
    subprocess.run([sys.executable, "scripts/build_site.py"], cwd=ROOT, check=True, capture_output=True)
    site = ROOT / "_site"
    names = {str(p.relative_to(site)).replace("\\", "/") for p in site.rglob("*") if p.is_file()}
    assert {"index.html", "worldcup.html", "site-version.json"} <= names
    assert any(n.startswith("data/nfl/") for n in names)
    for private in ("CLAUDE.md", "render.yaml", "wrangler.jsonc"):
        assert private not in names
    assert not any(n.startswith(("data-raw/", "scripts/", ".github/", "tracking/")) for n in names)
    assert len(json.loads((site / "site-version.json").read_text())["hash"]) == 16


def test_the_worker_schedule_matches_the_workflows_run_slots():
    """The cron starts NFL/NBA/MLB at the exact times their book_lines RUN_SLOTS assume."""
    worker = (ROOT / "cloudflare" / "worker.js").read_text(encoding="utf-8")
    from nfl import book_lines as nfl_bl
    from nba import book_lines as nba_bl
    from mlb import book_lines as mlb_bl
    for slots, job in ((nfl_bl.RUN_SLOTS, "nfl"), (nba_bl.RUN_SLOTS, "nba"), (mlb_bl.RUN_SLOTS, "mlb")):
        for weekday, hour, minute in slots:
            assert f'[{hour}, {minute}, "{job}"' in worker, (job, hour, minute)
