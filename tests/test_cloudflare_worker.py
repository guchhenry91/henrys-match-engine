"""Runs cloudflare/worker.js for real (Node + node:sqlite standing in for D1):
the odds relay's limits, the job queue, the watchdog and the news trigger.
The scenarios live in tests/cloudflare_worker_check.mjs."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUB = '''class Container { constructor(){} }
const getContainer = (_ns, name) => ({
  start: async (opts) => globalThis.__containers.start(name, opts),
  getState: async () => ({ status: globalThis.__containers.state(name) }),
  destroy: async () => {}, fetch: async () => new Response("proxy") });'''


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node 22+")
def test_worker_scenarios(tmp_path):
    src = (ROOT / "cloudflare" / "worker.js").read_text(encoding="utf-8")
    imp = 'import { Container, getContainer } from "@cloudflare/containers";'
    assert imp in src
    (tmp_path / "worker.mjs").write_text(src.replace(imp, STUB), encoding="utf-8")
    r = subprocess.run(["node", "--no-warnings", str(ROOT / "tests" / "cloudflare_worker_check.mjs"),
                        (tmp_path / "worker.mjs").as_uri(), str(ROOT / "cloudflare" / "schema.sql")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
