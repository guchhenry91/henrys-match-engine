"""Assemble the Cloudflare site folder: ONLY what the page serves.

Render published the whole repository root -- CLAUDE.md, raw pick logs, the API
credit ledger, scripts and workflows were all reachable on the site. The Worker's
assets come from _site, which holds just the pages and the published boards.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
FILES = ["index.html", "worldcup.html"]
DIRS = ["data"]


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for f in FILES:
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, OUT / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, OUT / d)
    for extra in ROOT.glob("*.png"):
        shutil.copy2(extra, OUT / extra.name)
    for extra in ROOT.glob("*.ico"):
        shutil.copy2(extra, OUT / extra.name)
    # A fingerprint of everything served, so the deploy can skip when the live
    # site already has exactly this content (deploy-cloudflare.yml).
    import hashlib
    import json
    h = hashlib.sha256()
    for f in sorted(x for x in OUT.rglob("*") if x.is_file()):
        h.update(str(f.relative_to(OUT)).encode())
        h.update(f.read_bytes())
    (OUT / "site-version.json").write_text(json.dumps({"hash": h.hexdigest()[:16]}), encoding="utf-8")
    n = sum(1 for _ in OUT.rglob("*") if _.is_file())
    print(f"built {OUT.relative_to(ROOT)}: {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
