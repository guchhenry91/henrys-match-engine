"""Export the completed NBA seasons to the repo, once: data-raw/nba/history/.

stats.nba.com times out from home and is known to refuse cloud IPs, so the
scheduled board cannot fetch history there. A finished season never changes, so
it is trimmed to the columns the engine reads and gzipped -- about 6 MB for all
fifteen seasons -- and nba.data reads it before trying the network.

Run locally, from a machine with the stats.nba.com cache in data-raw/nba/_cache.
"""
import sys

import pandas as pd

from nba import config, data


def main() -> int:
    data.HISTORY.mkdir(parents=True, exist_ok=True)
    for season in config.SEASONS:
        for side, name in (("P", "player"), ("T", "team")):
            src = data.CACHE / f"{name}_{season}.csv"
            if not src.exists():
                print(f"missing {src.name}; fetch it first (python -m scripts.nba_backtest)")
                return 1
            frame = pd.read_csv(src, low_memory=False)
            keep = [c for c in data.HISTORY_COLUMNS[side] if c in frame.columns]
            out = data.HISTORY / f"{name}_{season}.csv.gz"
            frame[keep].to_csv(out, index=False, compression="gzip")
            print(f"{out.name}: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
