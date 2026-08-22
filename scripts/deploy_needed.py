"""Did this publish change anything a reader would see? Exit 0 if yes.

WHY. Render meters build minutes, and on 2026-08-22 the account hit its spend
limit and stopped deploying entirely. The cause was mine: the matchday cadence
went from 30 to 15 minutes to stop picks voting late, which doubled runs -- and
EVERY run deployed, because publish rewrites the `updated` timestamp in all four
league files, so `git diff --quiet` was never quiet.

Most of those deploys shipped nothing. Measured earlier in the season, a typical
no-op publish differs only in `updated` and in lambda_goals at the SEVENTH
decimal -- optimiser float noise, invisible on the page.

So the cadence is not the thing to cut; the pointless deploys are. This compares
the freshly published payloads against HEAD with timestamps stripped and floats
rounded to what the page actually renders, and reports whether a deploy is
warranted.

ANY change under data-raw/leagues counts immediately and without comparison: that
is where a pick FREEZES, where a result is recorded and where the Telegram dedup
memory lives. Those must reach the repo on the run that makes them, whatever the
rendered payloads look like.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Keys that change on every run without changing what anyone sees.
NOISE_KEYS = {"updated", "locked_at", "lineup_checked_at", "lineup_api_attempted_at",
              "checked", "released_at", "_verified_on"}
ROUND_TO = 4          # publish rounds displayed probabilities to 3-4 dp


def _norm(x):
    """Strip timestamps and round floats to display precision."""
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in sorted(x.items()) if k not in NOISE_KEYS}
    if isinstance(x, list):
        return [_norm(v) for v in x]
    if isinstance(x, float):
        return round(x, ROUND_TO)
    return x


def staged() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def head_version(path: str):
    out = subprocess.run(["git", "show", f"HEAD:{path}"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return None                      # new file -> meaningful
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def main() -> int:
    files = staged()
    if not files:
        print("nothing staged; no deploy needed")
        return 1
    for path in files:
        if path.startswith("data-raw/leagues"):
            print(f"deploy needed: {path} changed (locks, results or send-memory)")
            return 0
    for path in files:
        if not path.endswith(".json"):
            print(f"deploy needed: {path} changed")
            return 0
        old = head_version(path)
        try:
            new = json.loads((ROOT / path).read_text(encoding="utf-8"))
        except Exception:
            print(f"deploy needed: {path} unreadable, not risking a stale site")
            return 0
        if old is None or _norm(old) != _norm(new):
            print(f"deploy needed: {path} changed beyond timestamps")
            return 0
    print(f"no deploy needed: {len(files)} file(s) differ only in timestamps "
          f"and sub-display-precision floats")
    return 1


if __name__ == "__main__":
    sys.exit(main())
