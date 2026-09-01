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

IT NOW CHECKS THE PUBLISHED PAYLOADS TOO, and the reasoning that excluded them
was wrong. "Written by json.dump and cannot be malformed this way" is true of the
WRITER and says nothing about what can happen to a file afterwards. On 2026-08-26
a local task resolved an autostash conflict by committing the conflicted files
verbatim, and all eight published payloads went to the live site containing raw
`<<<<<<<` markers. Every board on the page was empty. The team-news step then
crashed on the same markers -- before publish or sanity_check could run -- so the
gate that would have caught it sat behind the thing it needed to catch.

A machine-written file is not immune to a merge landing on top of it.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every JSON file the site depends on, whoever wrote it. The first group is
# hand/agent-edited and feeds the model; the second is what the model publishes
# and the browser fetches.
TRACKED = [
    "data-raw/leagues/transfers.json",
    "data-raw/leagues/news.json",
    "data-raw/leagues/rosters.json",
    "data-raw/leagues/absence_impact.json",
    "data-raw/leagues/absence_impact_by_position.json",
    "data-raw/leagues/results_override.json",
    "data-raw/leagues/fixture_times.json",
    "data-raw/leagues/six_scores.json",
    # PUBLISHED PAYLOADS. Machine-written, and checked anyway -- see the module
    # docstring. These are what the site actually serves, so a broken one is not a
    # failed publish, it is an empty page.
    "data/leagues/best.json",
    "data/leagues/player_picks.json",
    "data/leagues/parlays.json",
    "data/leagues/six_scores.json",
    "data/leagues/pl.json",
    "data/leagues/laliga.json",
    "data/leagues/bundesliga.json",
    "data/leagues/ligue1.json",
    "data/nfl/board.json",
    "data/nba/board.json",
    "data/ucl/board.json",
]

# A conflict marker parses as a JSON error, but the message ("Expecting property
# name") describes the symptom and not the cause. Naming it turns a puzzling
# report into an obvious one.
CONFLICT_PREFIXES = ("<<<<<<<", "=======", ">>>>>>>")


def conflict_line(text: str):
    """1-indexed line of the first merge-conflict marker, or None.

    Checked line by line rather than as a substring: "=======" appears inside
    plenty of legitimate text, but a line that IS a marker starts with it.
    """
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.rstrip()
        for prefix in CONFLICT_PREFIXES:
            if stripped == prefix or stripped.startswith(prefix + " "):
                return number
    return None


def check(paths=None) -> int:
    paths = paths or TRACKED
    bad = []
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            continue                      # optional files (e.g. experiment output)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            bad.append((rel, f"not valid UTF-8: {exc}"))
            continue
        marker = conflict_line(text)
        if marker is not None:
            bad.append((rel, f"UNRESOLVED MERGE CONFLICT at line {marker} -- this "
                             f"file carries conflict markers"))
            continue
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            bad.append((rel, f"line {exc.lineno} column {exc.colno}: {exc.msg}"))
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
