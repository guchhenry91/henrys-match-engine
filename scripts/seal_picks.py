"""Seal every newly frozen pick into its log's hash chain (tracking/seal.py).

Runs right after each step that freezes picks, in the same phase, so a pick and its
seal are saved together. Never fatal to the job that froze the picks."""
from __future__ import annotations

from scripts.health_check import ROOT, SEAL_LOGS
from tracking import seal


def main() -> int:
    total = 0
    for rel in SEAL_LOGS:
        try:
            n = seal.seal(ROOT / rel)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {rel}: not sealed ({type(exc).__name__}: {exc})")
            continue
        if n:
            print(f"  {rel}: sealed {n} new pick(s)")
        total += n
    print(f"sealed {total} new pick(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
