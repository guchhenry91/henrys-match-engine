"""Trim Understat per-match exports to one row per player per match (leagues/forecast.trim).

  python -m scripts.trim_understat_matches --src <export dir> [--first 2018 --last 2026]
writes data-raw/leagues/understat_matches/<LEAGUE>_<season>.csv.gz (what the soccer
props rebuild trains on; small enough for main, unlike the raw shot files).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from leagues import forecast as fc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--first", type=int, default=2018)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--out", default=str(fc.MATCHES_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lg in fc.LEAGUES:
        for s in range(args.first, args.last + 1):
            p, sh = Path(args.src) / f"{lg}_{s}_players.csv.gz", Path(args.src) / f"{lg}_{s}_shots.csv.gz"
            if not p.exists() or not sh.exists():
                print(f"{lg} {s}: missing")
                continue
            t = fc.trim(pd.read_csv(p), pd.read_csv(sh), lg, s)
            t.to_csv(out / f"{lg}_{s}.csv.gz", index=False, compression="gzip")
            print(f"{lg} {s}: {len(t)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
