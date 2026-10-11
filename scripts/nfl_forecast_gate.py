"""The gate for the rebuilt NFL prop model (nfl/forecast.py, docs/REBUILD_PLAN.md).

Walk-forward: each scored season is forecast by a model trained only on the
TRAIN_SEASONS before it. Per market:

1. Calibration on mechanical lines around each player's recent yards (the old
   gate's question, kept as one check).
2. Against an INFORMED book that knows each player's actual opportunities
   (targets / carries / attempts / touches, with 25% noise) -- usage is what books
   know best. Model alone, book alone, and the two COMBINED; the published rule
   (combined >= PUBLISH_BAR and the forecast on the same side) scored on what landed.
3. REAL bet365 lines: the live NFL picks log's priced, graded props of 2026, re-asked
   of the new model (trained only on 2021-2025) -- model alone vs the book's own
   de-vigged chance, on lines a real bookmaker set.

Writes data-raw/nfl/forecast_report.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import dist
from nfl import forecast as fc
from tracking import combine_fit

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data-raw" / "nfl" / "forecast_report.json"
SCORED = (2021, 2022, 2023, 2024, 2025)
PUBLISH_BAR = 0.62
ALT_BAR = 0.70
OPP = {"passing_yards": ("attempts", "passing_yards_per_attempts"),
       "rushing_yards": ("carries", "rushing_yards_per_carries"),
       "receiving_yards": ("targets", "receiving_yards_per_targets")}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def published(p, y, bar, p_model=None):
    side_p = np.where(p >= 0.5, p, 1 - p)
    side_y = np.where(p >= 0.5, y, 1 - y)
    pick = side_p >= bar
    if p_model is not None:
        pick &= (p >= 0.5) == (p_model >= 0.5)
    return {"n": int(pick.sum()),
            "said": round(float(side_p[pick].mean()), 4) if pick.any() else None,
            "landed": round(float(side_y[pick].mean()), 4) if pick.any() else None}


VACATED = True


def _alt(pc, pb, y, ids, pm, need_model):
    """One rung per player-game: the HIGHEST line clearing ALT_BAR (the best price),
    optionally only where the forecast is at least as sure as the book."""
    ok = pc >= ALT_BAR
    if need_model:
        ok &= pm >= pb
    f = pd.DataFrame({"id": ids[ok], "p": pc[ok], "y": y[ok], "b": pb[ok]})
    f = f.sort_values("p").groupby("id").head(1)            # lowest chance = highest line
    if f.empty:
        return {"n": 0}
    return {"n": int(len(f)), "said": round(float(f["p"].mean()), 4),
            "landed": round(float(f["y"].mean()), 4), "book_said": round(float(f["b"].mean()), 4)}


def score(frame, market, season, rng):
    train = frame[(frame["season"] < season) & (frame["season"] >= season - fc.TRAIN_SEASONS)]
    test = frame[(frame["season"] == season) & fc.eligible(frame, market)
                 & (frame["season_type"] == "REG")].copy()
    model = fc.MarketModel(market, vacated=VACATED).fit(train)
    stat = fc.MARKETS[market]["stat"]
    y_raw = test[stat].to_numpy(float)
    noise = np.exp(rng.normal(0, 0.25, len(test)))
    if market == "anytime_touchdown":
        p = model.p_over(test, 0.5)
        y = (y_raw > 0).astype(float)
        touches = (test["carries"] + test["receptions"]).to_numpy(float) * noise
        rate = ((test["rushing_tds_e8"] + test["receiving_tds_e8"])
                / (test["carries_e8"] + test["receptions_e8"]).clip(lower=0.5)).to_numpy(float)
        book = np.clip(1 - np.exp(-rate * touches), 0.01, 0.95)
        return {"mech": (p, y), "inf": (p, y, book), "mu": None}
    mu = model.mean(test)
    base = test[f"{stat}_e8"].to_numpy(float)
    mech_p, mech_y = [], []
    for f in (0.7, 0.85, 1.0, 1.15, 1.3):
        line = np.maximum(np.floor(base * f) + 0.5, 0.5)
        mech_p.append(model.spread.p_over(mu, line))
        mech_y.append((y_raw > line).astype(float))
    opp_col, per_col = OPP[market]
    book_mu = np.maximum(test[per_col].to_numpy(float) * test[opp_col].to_numpy(float) * noise, 0.5)
    # A real book sets a yardage line at its MEDIAN, near 50/50 (the 2026 bet365 lines'
    # fair over averages 0.500) -- not at the mean, which skew puts above the median.
    band = np.clip(np.searchsorted(model.spread.edges, book_mu, side="right") - 1, 0, len(model.spread.quant) - 1)
    med_ratio = model.spread.quant[band, model.spread.q // 2]
    book_line = np.maximum(np.floor(book_mu * med_ratio) + 0.5, 0.5)
    book = model.spread.p_over(book_mu, book_line)
    p = model.spread.p_over(mu, book_line)
    # THE ALTERNATE LADDER: lower lines the book rates likelier to go over (its own
    # 35th / 25th / 15th percentile -> about 65 / 75 / 85% over).
    alt_p, alt_b, alt_y, alt_id = [], [], [], []
    for qq in (0.35, 0.25, 0.15):
        qi = int(round(qq * (model.spread.q - 1)))
        rung = np.maximum(np.floor(book_mu * model.spread.quant[band, qi]) + 0.5, 0.5)
        ok = rung < book_line
        alt_p.append(model.spread.p_over(mu, rung)[ok]); alt_b.append(model.spread.p_over(book_mu, rung)[ok])
        alt_y.append((y_raw > rung).astype(float)[ok]); alt_id.append(np.arange(len(test))[ok] + season * 10**6)
    return {"mech": (np.concatenate(mech_p), np.concatenate(mech_y)),
            "inf": (p, (y_raw > book_line).astype(float), book),
            "alt": tuple(np.concatenate(v) for v in (alt_p, alt_b, alt_y, alt_id)),
            "mu": (mu, y_raw)}


def real_lines(frame, models, weights=None):
    """The live log's 2026 bet365 props, re-asked of the new model."""
    try:
        logs = json.loads(Path("/tmp/claude-0/live/logs.json").read_text())
        props = logs["data-raw/nfl/picks_log.json"]["props"].values()
    except Exception:
        return {"note": "no live picks log on hand"}
    rows = frame.set_index(["game_id", "player_id"])
    out = {}
    for market, model in models.items():
        es = [e for e in props if e.get("market") == market and e.get("graded") in ("correct", "wrong")
              and e.get("book_p") is not None and (e.get("line") is not None or market == "anytime_touchdown")]
        pm, pb, yy = [], [], []
        for e in es:
            key = (e["game_id"], e["player_id"])
            if key not in rows.index:
                continue
            r = rows.loc[[key]].iloc[[0]]
            line = 0.5 if market == "anytime_touchdown" else float(e["line"])
            p_over = float(model.p_over(r, line)[0])
            side_under = e.get("side") == "under"
            b_over = 1 - float(e["book_p"]) if side_under else float(e["book_p"])
            won = e["graded"] == "correct"
            pm.append(p_over), pb.append(b_over), yy.append(float(won != side_under))
        if len(yy) < 10:
            out[market] = {"n": len(yy)}
            continue
        pm, pb, yy = map(np.array, (pm, pb, yy))
        ll = lambda p: float(-np.mean(yy * np.log(np.clip(p, 1e-4, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-4, 1))))
        w = (weights or {}).get(market) or (0.0, 1.0, 0.0)
        pc = combine_fit._apply(w, np.column_stack([_logit(pm), _logit(pb)]))
        out[market] = {"n": int(len(yy)), "logloss_model": round(ll(pm), 4), "logloss_book": round(ll(pb), 4),
                       "logloss_combined": round(ll(pc), 4), "published": published(pc, yy, PUBLISH_BAR, pm),
                       "over_rate": round(float(yy.mean()), 4), "model_said": round(float(pm.mean()), 4),
                       "book_said": round(float(pb.mean()), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(fc.MARKETS))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--no-vacated", action="store_true", help="A/B: without teammates out")
    args = ap.parse_args()
    global VACATED
    VACATED = not args.no_vacated
    t0 = time.time()
    seasons = range(min(SCORED) - fc.TRAIN_SEASONS - 1, 2027)
    frame = fc.build(fc.load(seasons), fc.out_reports(seasons))
    print(f"teammates out: {int((frame['vac_target_share'] > 0).sum())} player-weeks with a "
          f"listed-Out teammate's targets", flush=True)
    print(f"features: {len(frame)} player-weeks in {time.time() - t0:.0f}s", flush=True)
    report = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
              "scored_seasons": list(SCORED), "publish_bar": PUBLISH_BAR, "markets": {}}
    live_models, weights = {}, {}
    for market in args.markets.split(","):
        rng = np.random.default_rng(7)
        parts = [score(frame, market, s, rng) for s in SCORED]
        mp = np.concatenate([q["mech"][0] for q in parts]); my = np.concatenate([q["mech"][1] for q in parts])
        p = np.concatenate([q["inf"][0] for q in parts]); y = np.concatenate([q["inf"][1] for q in parts])
        b = np.concatenate([q["inf"][2] for q in parts])
        out = {"n": int(len(y)), "mechanical": dist.calibration(mp, my),
               "model_alone": published(p, y, PUBLISH_BAR), "book_alone": published(b, y, PUBLISH_BAR)}
        if parts[0]["mu"] is not None:
            mu = np.concatenate([q["mu"][0] for q in parts]); yr = np.concatenate([q["mu"][1] for q in parts])
            out["mae"] = round(float(np.mean(np.abs(mu - yr))), 2)
        first = len(parts[0]["inf"][0])
        # The book's chance is taken as given (weight 1): at a median line it sits near
        # 50/50 and a free weight on it explodes. Only the forecast's pull is learned.
        w = combine_fit.fit_offset(p[:first], b[:first], y[:first])
        comb = combine_fit._apply(w, np.column_stack([_logit(p[first:]), _logit(b[first:])]))
        weights[market] = w
        out["combined"] = {"weights_model_book_const": list(w),
                           "calibration": dist.calibration(comb, y[first:]),
                           "would_publish_agree": published(comb, y[first:], PUBLISH_BAR, p[first:]),
                           "would_publish_70": published(comb, y[first:], 0.70, p[first:])}
        if parts[0].get("alt") is not None:
            # Alternate rungs, scored on the seasons after the first (weights from it).
            ap_ = [np.concatenate([q["alt"][i] for q in parts[1:]]) for i in range(4)]
            pc = combine_fit._apply(w, np.column_stack([_logit(ap_[0]), _logit(ap_[1])]))
            out["alternate"] = {"rungs": int(len(pc)),
                                "book_favourite_70": _alt(pc, ap_[1], ap_[2], ap_[3], ap_[1], False),
                                "forecast_backed_70": _alt(pc, ap_[1], ap_[2], ap_[3], ap_[0], True)}
            print(f"  {market} alternate: {out['alternate']}", flush=True)
        report["markets"][market] = out
        print(f"{market}: n={out['n']} MAE {out.get('mae')} mech ECE {out['mechanical']['ece']} | model alone "
              f"{out['model_alone']} | book alone {out['book_alone']} | COMBINED w={out['combined']['weights_model_book_const']} "
              f"ECE {out['combined']['calibration']['ece']} publish {out['combined']['would_publish_agree']} "
              f"@70 {out['combined']['would_publish_70']}", flush=True)
        live_models[market] = fc.MarketModel(market, vacated=VACATED).fit(
            frame[(frame["season"] < 2026) & (frame["season"] >= 2026 - fc.TRAIN_SEASONS)])
    report["real_bet365_lines_2026"] = real_lines(frame, live_models, weights)
    print("REAL bet365 lines 2026:", json.dumps(report["real_bet365_lines_2026"]), flush=True)
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
