"""The gate for the rebuilt NBA prop model (nba/forecast.py, docs/REBUILD_PLAN.md).

Walk-forward by season: each scored season is forecast by a model trained only on the
TRAIN_SEASONS before it. Three questions, per stat:

1. Is the forecast calibrated on EVERY line? Each player-game is asked about seven
   lines around his own recent average (-3..+3 from the nearest half point) -- the
   old gate's mechanical lines, kept as one check among three.
2. Is it calibrated on lines an INFORMED book would set? The old model failed here
   (it read the line as information it did not have). A simulated book knows each
   player's actual minutes -- the thing books know best and models least -- and sets
   its line at the median of his own per-minute rate times those minutes. A model
   that never reads the line should stay honest about what it does not know; this
   measures how far the informed line moves the truth away from its forecast.
3. Do the picks it would PUBLISH win at the rate stated? Among the informed-book
   lines, the side the model favours at >= PUBLISH_BAR, scored on what landed.

Writes data-raw/nba/forecast_report.json. Read-only otherwise.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import dist
from nba import forecast as fc

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data-raw" / "nba" / "forecast_report.json"
SCORED = (2021, 2022, 2023, 2024, 2025, 2026)
OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
PUBLISH_BAR = 0.62
MIN_ROLE_MIN = 15.0
BOOK_MIN_SD = 5.0           # a book's minutes projection misses by ~5 minutes          # props are offered on rotation players, not the deep bench


def score_season(frame, stat, season):
    train = frame[(frame["season"] < season) & (frame["season"] >= season - fc.TRAIN_SEASONS)]
    test = frame[(frame["season"] == season) & (frame["games_before"] >= 5)
                 & (frame["min_ewm10"] >= MIN_ROLE_MIN)].copy()
    model = fc.StatModel(stat).fit(train)
    mu = model.mean(test)
    alpha = model.disp(mu)
    y = test[stat].to_numpy(float)
    base = np.floor(test[f"{stat}_pm10"].to_numpy() * test["min_ewm10"].to_numpy()) + 0.5
    # 1. mechanical lines around his recent average
    mech = {}
    for k in OFFSETS:
        line = np.maximum(base + k, 0.5)
        p = dist.p_over(mu, alpha, line)
        mech[k] = (p, (y > line).astype(float))
    # 2. informed book: knows the minutes he will play well, not perfectly
    #    (actual minutes + noise, sd BOOK_MIN_SD), and prices its own line.
    rng = np.random.default_rng(season)
    book_min = np.clip(test["MIN"].to_numpy() + rng.normal(0, BOOK_MIN_SD, len(test)), 1, 48)
    book_mu = test[f"{stat}_pm10"].to_numpy() * book_min
    book_line = np.maximum(np.floor(book_mu) + 0.5, 0.5)
    book_p = dist.p_over(book_mu, model.disp(book_mu), book_line)
    p_inf = dist.p_over(mu, alpha, book_line)
    y_inf = (y > book_line).astype(float)
    return {"mu": mu, "y": y, "naive": test[f"{stat}_pm10"].to_numpy() * test["min_ewm10"].to_numpy(),
            "mech": mech, "inf": (p_inf, y_inf, book_p)}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_combine(p_model, p_book, y):
    """final = sigmoid(a*logit(model) + b*logit(book) + c), fitted by logistic regression."""
    from sklearn.linear_model import LogisticRegression
    x = np.column_stack([_logit(p_model), _logit(p_book)])
    lr = LogisticRegression(C=1e3).fit(x, y)
    return lr, (round(float(lr.coef_[0][0]), 3), round(float(lr.coef_[0][1]), 3),
                round(float(lr.intercept_[0]), 3))


def published(p, y, bar):
    side_p = np.where(p >= 0.5, p, 1 - p)
    side_y = np.where(p >= 0.5, y, 1 - y)
    pick = side_p >= bar
    return {"n": int(pick.sum()),
            "said": round(float(side_p[pick].mean()), 4) if pick.any() else None,
            "landed": round(float(side_y[pick].mean()), 4) if pick.any() else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=",".join(fc.STATS))
    ap.add_argument("--seasons", default=",".join(map(str, SCORED)))
    args = ap.parse_args()
    t0 = time.time()
    rows = fc.load_history()
    frame = fc.build(rows)
    print(f"features built: {len(frame)} player-games in {time.time() - t0:.0f}s", flush=True)
    report = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
              "scored_seasons": [int(s) for s in args.seasons.split(",")],
              "train_seasons": fc.TRAIN_SEASONS, "publish_bar": PUBLISH_BAR, "stats": {}}
    for stat in args.stats.split(","):
        parts = [score_season(frame, stat, s) for s in report["scored_seasons"]]
        mu = np.concatenate([p["mu"] for p in parts])
        y = np.concatenate([p["y"] for p in parts])
        naive = np.concatenate([p["naive"] for p in parts])
        out = {"n": int(len(y)), "mae": round(float(np.mean(np.abs(mu - y))), 3),
               "mae_naive": round(float(np.mean(np.abs(naive - y))), 3), "lines": {}}
        allp, ally = [], []
        for k in OFFSETS:
            p = np.concatenate([q["mech"][k][0] for q in parts])
            t = np.concatenate([q["mech"][k][1] for q in parts])
            allp.append(p), ally.append(t)
            c = dist.calibration(p, t)
            out["lines"][str(k)] = {"ece": c["ece"], "said": round(float(p.mean()), 4),
                                    "landed": round(float(t.mean()), 4)}
        out["mechanical"] = dist.calibration(np.concatenate(allp), np.concatenate(ally))
        p_inf = np.concatenate([q["inf"][0] for q in parts])
        y_inf = np.concatenate([q["inf"][1] for q in parts])
        b_inf = np.concatenate([q["inf"][2] for q in parts])
        out["informed_book"] = {"model": dist.calibration(p_inf, y_inf),
                                "book": dist.calibration(b_inf, y_inf)}
        out["model_alone_would_publish"] = published(p_inf, y_inf, PUBLISH_BAR)
        out["book_alone_would_publish"] = published(b_inf, y_inf, PUBLISH_BAR)
        # 3. COMBINED: fitted on the first scored season, judged on the rest.
        first = len(parts[0]["inf"][0])
        lr, coefs = fit_combine(p_inf[:first], b_inf[:first], y_inf[:first])
        comb = lr.predict_proba(np.column_stack([_logit(p_inf[first:]), _logit(b_inf[first:])]))[:, 1]
        out["combined"] = {"weights_model_book_const": coefs,
                           "calibration": dist.calibration(comb, y_inf[first:]),
                           "would_publish": published(comb, y_inf[first:], PUBLISH_BAR),
                           "would_publish_70": published(comb, y_inf[first:], 0.70),
                           # ...and only where the FORECAST itself favours the same side
                           "would_publish_agree": published(
                               np.where((comb >= 0.5) == (p_inf[first:] >= 0.5), comb, 0.5),
                               y_inf[first:], PUBLISH_BAR)}
        report["stats"][stat] = out
        print(f"{stat}: n={out['n']} MAE {out['mae']} (naive {out['mae_naive']}) | mechanical ECE "
              f"{out['mechanical']['ece']} | vs informed book: model alone {out['model_alone_would_publish']}"
              f" book alone {out['book_alone_would_publish']} | COMBINED w={out['combined']['weights_model_book_const']}"
              f" ECE {out['combined']['calibration']['ece']} publish@62 {out['combined']['would_publish']}"
              f" @70 {out['combined']['would_publish_70']} agree@62 {out['combined']['would_publish_agree']}", flush=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
