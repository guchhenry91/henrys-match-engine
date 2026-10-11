"""The gate for the rebuilt soccer player props (leagues/forecast.py).

Walk-forward: each scored season is forecast by a model trained on the
TRAIN_SEASONS before it, all five leagues pooled. Every candidate (anyone who
appeared for the team in its last few matches) is scored, and NOT APPEARING IS A
LOSS -- the way the record grades. Two readings:
  * before lineups (the role model guesses who starts);
  * with the confirmed XI (starters known; the bench's chance of coming on from
    the role model).
Per market: calibration over every candidate, and the PUBLISHED subset (at the
market's bar, top PER_MATCH per match) scored as said vs landed.

Writes data-raw/leagues/props_forecast_report.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import dist
from leagues import forecast as fc

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data-raw" / "leagues" / "props_forecast_report.json"
SCORED = (2022, 2023, 2024, 2025)
BARS = {"goal": 0.40, "shots": 0.62, "sot": 0.62}
PER_MATCH = 3


def published(frame, p, y, bar):
    f = pd.DataFrame({"m": frame["match_id"].to_numpy(), "p": p, "y": y})
    f = f[f["p"] >= bar].sort_values("p", ascending=False).groupby("m").head(PER_MATCH)
    if f.empty:
        return {"n": 0}
    return {"n": int(len(f)), "said": round(float(f["p"].mean()), 4),
            "landed": round(float(f["y"].mean()), 4)}


def lineup_roles(model, rows):
    r = model.roles(rows)
    started = (rows["role"] == 2).to_numpy()
    bench = r[:, 1] / np.clip(r[:, 0] + r[:, 1], 1e-6, None)
    out = np.zeros_like(r)
    out[started, 2] = 1.0
    out[~started, 1] = bench[~started]
    out[~started, 0] = 1 - bench[~started]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()
    t0 = time.time()
    d = fc.load(range(min(SCORED) - fc.TRAIN_SEASONS, max(SCORED) + 1))
    frame = fc.build(d)
    print(f"{len(frame)} candidate rows in {time.time() - t0:.0f}s", flush=True)
    acc = {m: {"pre": [], "xi": [], "y": [], "rows": []} for m in fc.MARKETS}
    for season in SCORED:
        train = frame[(frame["season"] < season) & (frame["season"] >= season - fc.TRAIN_SEASONS)]
        test = frame[frame["season"] == season]
        model = fc.PropForecast().fit(train)
        pre, xi = model.predict(test), model.predict(test, lineup_roles(model, test))
        for m, col in fc.MARKETS.items():
            acc[m]["pre"].append(pre[m]); acc[m]["xi"].append(xi[m])
            acc[m]["y"].append(test[col].to_numpy(float)); acc[m]["rows"].append(test[["match_id"]])
        print(f"  {season}: {len(test)} rows, {time.time() - t0:.0f}s", flush=True)
    report = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
              "scored_seasons": list(SCORED), "bars": BARS, "per_match": PER_MATCH, "markets": {}}
    for m in fc.MARKETS:
        y = np.concatenate(acc[m]["y"]); rows = pd.concat(acc[m]["rows"])
        out = {"n": int(len(y)), "base_rate": round(float(y.mean()), 4)}
        for k in ("pre", "xi"):
            p = np.concatenate(acc[m][k])
            ll = float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
            out[k] = {"calibration": dist.calibration(p, y), "logloss": round(ll, 4),
                      "published": published(rows, p, y, BARS[m]),
                      "published_55": published(rows, p, y, 0.55) if m != "goal" else published(rows, p, y, 0.35)}
        report["markets"][m] = out
        print(f"{m}: base {out['base_rate']} | before XI ECE {out['pre']['calibration']['ece']} "
              f"pub {out['pre']['published']} | with XI ECE {out['xi']['calibration']['ece']} "
              f"pub {out['xi']['published']}", flush=True)
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
