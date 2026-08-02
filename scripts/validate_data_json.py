"""Fail fast, and legibly, on any malformed data JSON.

WHY THIS EXISTS. On 2026-08-02 the local matchday-news task appended an entry to
data-raw/leagues/transfers.json and left out one comma. transfers.json is read by
every league, so all four publishes aborted with "0/4 league publish(es)
succeeded" -- a message that says nothing about which file, which line, or why.
The pipeline correctly refused to ship broken data, but the site sat frozen for a
full DAY before anyone read the logs closely enough to find a missing comma on
line 206.

This validator turns that day-long, cryptic failure into an immediate, specific
one: it parses every hand/agent-editable JSON file up front and, on the first bad
one, prints the exact file, line and column. Wired as the first step of the CI
publish (so a bad file fails in seconds, not after a slow model run) and as a
local pre-commit hook (so the task that writes these files can never COMMIT a
broken one in the first place).

Only the files humans or agents edit by hand are checked -- the model's own output
in data/leagues/ is written by json.dump and cannot be malformed this way.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hand/agent-editable JSON the publish path depends on. A parse error in any of
# these can abort a publish; the model's own data/leagues/*.json output is
# machine-written and deliberately not included.
TRACKED = [
    "data-raw/leagues/transfers.json",
    "data-raw/leagues/news.json",
    "data-raw/leagues/rosters.json",
    "data-raw/leagues/absence_impact.json",
    "data-raw/leagues/absence_impact_by_position.json",
]


def check(paths=None) -> int:
    paths = paths or TRACKED
    bad = []
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            continue                      # optional files (e.g. experiment output)
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            bad.append((rel, f"line {exc.lineno} column {exc.colno}: {exc.msg}"))
        except UnicodeDecodeError as exc:
            bad.append((rel, f"not valid UTF-8: {exc}"))
    if bad:
        print("INVALID DATA JSON -- publish would abort:", file=sys.stderr)
        for rel, why in bad:
            print(f"  {rel}: {why}", file=sys.stderr)
        return 1
    print(f"data JSON OK ({sum((ROOT / p).exists() for p in paths)} files checked)")
    return 0


if __name__ == "__main__":
    # Any paths passed on the command line (e.g. a pre-commit hook handing us just
    # the staged files) override the default set; otherwise check everything.
    raise SystemExit(check(sys.argv[1:] or None))
