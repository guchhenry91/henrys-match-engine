"""The gate for the rebuilt MLB prop model (mlb/forecast.py, docs/REBUILD_PLAN.md).

Walk-forward: each scored season is forecast by models trained only on the
TRAIN_SEASONS before it. Per market:

1. Calibration at book-shaped lines (hits 0.5/1.5, H+R+RBI 0.5-3.5, HR and RBI 0.5,
   strikeouts around each starter's recent level, team runs 2.5-6.5).
2. Against an INFORMED book: it knows each batter's actual plate appearances and each
   starter's actual batters faced (with noise) -- the opportunity books price best --
   and sets strikeout and team-run lines at its own median. Model alone, book alone,
   and COMBINED (book's weight held at 1, tracking/combine_fit.fit_offset, fitted on
   the first scored season); the published rule (combined >= PUBLISH_BAR, forecast
   on the same side; home runs "yes" only) scored on what landed.
3. REAL lines: the live MLB log's graded props (Pinnacle / DraftKings / FanDuel),
   re-asked of models trained on the seasons before -- model alone vs the book.

Writes data-raw/mlb/forecast_report.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from forecast import dist
from mlb import forecast as fc
from tracking import combine_fit

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data-raw" / "mlb" / "forecast_report.json"
SCORED = (2021, 2022, 2023, 2024, 2025)
PUBLISH_BAR = 0.62
REAL_LOG = Path("/tmp/claude-0/mlb_log_bak.json")
FIXED = {"hits": (0.5, 1.5), "hrr": (0.5, 1.5, 2.5, 3.5), "hr": (0.5,), "rbi": (0.5,),
         "team_runs": (2.5, 3.5, 4.5, 5.5, 6.5)}
# Plate appearances by batting-order slot (2021-25 starters).
SLOT_PA = {1: 4.49, 2: 4.39, 3: 4.29, 4: 4.19, 5: 4.06, 6: 3.94, 7: 3.79, 8: 3.63, 9: 3.3}
BOOK_LINE = {"hits": 0.5, "hrr": 1.5, "hr": 0.5, "rbi": 0.5}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# Fixed-line markets (hits 0.5, H+R+RBI 1.5, HR and RBI 0.5): the book's chance
# varies from player to player, so both weights are learned. Median-set lines
# (strikeouts, team runs) sit near 50/50, so the book's weight is held at 1.
FREE = {"hits", "hrr", "hr", "rbi"}


def fit_weights(market, p, b, y):
    if market not in FREE:
        return combine_fit.fit_offset(p, b, y)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0).fit(np.column_stack([_logit(p), _logit(b)]), y)
    return (round(float(lr.coef_[0][0]), 4), round(float(lr.coef_[0][1]), 4),
            round(float(lr.intercept_[0]), 4))


def vetoed(pm, pb, y, bar, over_only=False):
    """The batter rule: the MODEL's chance at the bar, and the book on the same side."""
    side_p = np.where(pm >= 0.5, pm, 1 - pm)
    side_y = np.where(pm >= 0.5, y, 1 - y)
    pick = (side_p >= bar) & ((pm >= 0.5) == (pb >= 0.5))
    if over_only:
        pick &= pm >= 0.5
    return {"n": int(pick.sum()),
            "said": round(float(side_p[pick].mean()), 4) if pick.any() else None,
            "landed": round(float(side_y[pick].mean()), 4) if pick.any() else None}


def published(p, y, bar, p_model=None, over_only=False):
    side_p = np.where(p >= 0.5, p, 1 - p)
    side_y = np.where(p >= 0.5, y, 1 - y)
    pick = side_p >= bar
    if p_model is not None:
        pick &= (p >= 0.5) == (p_model >= 0.5)
    if over_only:
        pick &= p >= 0.5
    return {"n": int(pick.sum()),
            "said": round(float(side_p[pick].mean()), 4) if pick.any() else None,
            "landed": round(float(side_y[pick].mean()), 4) if pick.any() else None}


def _book(test, market, rng):
    """(book P(over), line) for an informed book."""
    if market in ("hits", "hrr", "hr", "rbi"):
        # What a book knows BEFORE first pitch: the posted batting-order slot (and so
        # the plate appearances it usually brings), not the PAs he actually got.
        slot = test["lineup"].fillna(5).clip(1, 9).astype(int).to_numpy()
        noisy = np.maximum(np.round(np.array([SLOT_PA[k] for k in slot])
                                    * np.exp(rng.normal(0, 0.05, len(test)))), 1)
        col = {"hits": "h", "hr": "hr", "rbi": "rbi"}.get(market)
        line = np.full(len(test), BOOK_LINE[market])
        if market == "hrr":
            rate = (test["h_rate_e60"] + test["r_rate_e60"] + test["rbi_rate_e60"]).to_numpy(float)
            return 1 - stats.poisson.cdf(np.floor(line), np.clip(rate * noisy, 0.05, None)), line
        rate = np.clip(test[f"{col}_rate_e60"].to_numpy(float), 0.001, 0.9)
        return 1 - stats.binom.cdf(np.floor(line), noisy.astype(int), rate), line
    if market == "strikeouts":
        # His usual workload (recent batters faced), not the outing he actually had.
        bfp = np.maximum(test["bfp_e5"].to_numpy(float) * np.exp(rng.normal(0, 0.05, len(test))), 3)
        lam = np.clip(test["krate_e20"].to_numpy(float), 0.05, 0.5) * bfp
        line = np.floor(lam) + 0.5 - (lam - np.floor(lam) < 0.5)  # nearest x.5 to the mean
        return 1 - stats.poisson.cdf(np.floor(line), lam), line
    lam = np.clip(0.5 * (test["runs_e40"] + test["opp_allowed_e40"]).to_numpy(float)
                  * test["park_runs"].to_numpy(float) * np.exp(rng.normal(0, 0.08, len(test))), 1, None)
    line = np.floor(lam) + 0.5 - (lam - np.floor(lam) < 0.5)
    n, pr = 1 / 0.12, 1 / (1 + 0.12 * lam)
    return 1 - stats.nbinom.cdf(np.floor(line), n, pr), line


def score(frames, market, season, rng):
    kind, stat, _ = fc.MARKETS[market]
    f = frames[kind]
    train = f[(f["season"] < season) & (f["season"] >= season - fc.TRAIN_SEASONS)]
    test = f[(f["season"] == season) & fc.eligible(f, kind)].dropna(subset=[stat])
    model = fc.CountModel(market).fit(train)
    y_raw = test[stat].to_numpy(float)
    if market == "strikeouts":
        base = np.floor(test["k_e20"].to_numpy(float))
        lines = [np.maximum(base + d + 0.5, 0.5) for d in (-2, -1, 0, 1, 2)]
    else:
        lines = [np.full(len(test), l) for l in FIXED[market]]
    mech_p = np.concatenate([model.p_over(test, l) for l in lines])
    mech_y = np.concatenate([(y_raw > l).astype(float) for l in lines])
    book, line = _book(test, market, rng)
    p = model.p_over(test, line)
    return {"mech": (mech_p, mech_y), "inf": (p, (y_raw > line).astype(float), np.clip(book, 1e-3, 1 - 1e-3)),
            "mae": float(np.mean(np.abs(model.mean(test) - y_raw)))}


def real_lines(frames, models, weights):
    try:
        log = json.loads(REAL_LOG.read_text())
    except Exception:
        return {"note": "no live MLB log on hand"}
    out = {}
    for market, model in models.items():
        kind, stat, _ = fc.MARKETS[market]
        f = frames[kind].copy()
        f["key"] = f["GAME_ID"].astype(str) + "|" + f["PLAYER_ID"].astype(str)
        f = f.drop_duplicates("key").set_index("key")
        es = [e for e in log.values() if isinstance(e, dict) and e.get("market") == market
              and e.get("graded") in ("correct", "wrong") and e.get("book_p") is not None]
        keys = [f"MLB{e['game_id']}|{e['player_id']}" for e in es]
        ok = [k in f.index for k in keys]
        es = [e for e, o in zip(es, ok) if o]
        keys = [k for k, o in zip(keys, ok) if o]
        if len(es) < 10:
            out[market] = {"n": len(es)}
            continue
        rows = f.loc[keys]
        pm = model.p_over(rows, np.array([float(e["line"]) for e in es]))
        under = np.array([e.get("side") == "under" for e in es])
        pb = np.array([float(e["book_p"]) for e in es])
        pb = np.where(under, 1 - pb, pb)
        won = np.array([e["graded"] == "correct" for e in es])
        y = (won != under).astype(float)
        pc = combine_fit._apply(weights.get(market) or (0.0, 1.0, 0.0),
                                np.column_stack([_logit(pm), _logit(pb)]))
        ll = lambda q: float(-np.mean(y * np.log(np.clip(q, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-4, 1))))
        out[market] = {"n": int(len(y)), "logloss_model": round(ll(pm), 4), "logloss_book": round(ll(pb), 4),
                       "logloss_combined": round(ll(pc), 4), "over_rate": round(float(y.mean()), 4),
                       "model_said": round(float(pm.mean()), 4), "book_said": round(float(pb.mean()), 4),
                       "published": published(pc, y, PUBLISH_BAR, pm, over_only=market == "hr"),
                       "model_with_book_veto": vetoed(pm, pb, y, PUBLISH_BAR, over_only=market == "hr")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(fc.MARKETS))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()
    t0 = time.time()
    frames = fc.build(fc.load(range(min(SCORED) - fc.TRAIN_SEASONS - 1, max(SCORED) + 1), with_current=True))
    print(f"features in {time.time() - t0:.0f}s: " + ", ".join(f"{k} {len(v)}" for k, v in frames.items()), flush=True)
    report = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
              "scored_seasons": list(SCORED), "publish_bar": PUBLISH_BAR, "markets": {}}
    weights, live = {}, {}
    for market in args.markets.split(","):
        rng = np.random.default_rng(11)
        parts = [score(frames, market, s, rng) for s in SCORED]
        mp = np.concatenate([q["mech"][0] for q in parts]); my = np.concatenate([q["mech"][1] for q in parts])
        p = np.concatenate([q["inf"][0] for q in parts]); y = np.concatenate([q["inf"][1] for q in parts])
        b = np.concatenate([q["inf"][2] for q in parts])
        first = len(parts[0]["inf"][0])
        w = fit_weights(market, p[:first], b[:first], y[:first])
        weights[market] = w
        comb = combine_fit._apply(w, np.column_stack([_logit(p[first:]), _logit(b[first:])]))
        hr = market == "hr"
        out = {"n": int(len(y)), "mae": round(float(np.mean([q["mae"] for q in parts])), 3),
               "mechanical": dist.calibration(mp, my),
               "model_alone": published(p, y, PUBLISH_BAR, over_only=hr),
               "book_alone": published(b, y, PUBLISH_BAR, over_only=hr),
               "combined": {"weights_model_book_const": list(w),
                            "calibration": dist.calibration(comb, y[first:]),
                            "would_publish_agree": published(comb, y[first:], PUBLISH_BAR, p[first:], over_only=hr),
                            "would_publish_70": published(comb, y[first:], 0.70, p[first:], over_only=hr)}}
        out["model_with_book_veto"] = vetoed(p, b, y, PUBLISH_BAR, over_only=hr)
        report["markets"][market] = out
        print(f"  {market} model + book veto: {out['model_with_book_veto']}", flush=True)
        print(f"{market}: n={out['n']} MAE {out['mae']} mech ECE {out['mechanical']['ece']} | model alone "
              f"{out['model_alone']} | book alone {out['book_alone']} | COMBINED w={list(w)} "
              f"ECE {out['combined']['calibration']['ece']} publish {out['combined']['would_publish_agree']} "
              f"@70 {out['combined']['would_publish_70']}", flush=True)
        kind = fc.MARKETS[market][0]
        f = frames[kind]
        live[market] = fc.CountModel(market).fit(f[(f["season"] < 2026) & (f["season"] >= 2026 - fc.TRAIN_SEASONS)])
    report["real_lines_2026"] = real_lines(frames, live, weights)
    print("REAL lines 2026:", json.dumps(report["real_lines_2026"]), flush=True)
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
