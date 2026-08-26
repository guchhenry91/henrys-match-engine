"""Learn per-scoreline corrections for the exact-score grid, and prove them.

WHY. The fitted grid is biased in ways that are stable across seasons: measured
on 1,900 Premier League matches it over-predicts 0-0 by 35% and 1-1 by 10%,
while under-predicting 2-2 by 20% and 3-1 by 12%. Dixon-Coles' low-score
correction is the cause -- it lifts the level low cells, and in this league it
lifts them too far.

WHAT THIS IS NOT. It is not an accuracy fix, and it must not be sold as one.
Exact-score prediction in the Premier League tops out near 13%: that is the
mean of the highest cell in a correctly calibrated grid, and an empirical
lookup fitted ON the answers reaches the same place. The shipped board already
scores 12.84% on a holdout. There is no headroom and this does not find any.

WHAT IT IS FOR. Six identical 1-1s is a board nobody can use. Uncalibrated, the
grid names 1-1 on 89% of fixtures; corrected, on 74%, showing four times as many
distinct scorelines -- at the same hit rate. That is the whole trade: a board
that responds to the fixture, bought for nothing.

SCOPE, DELIBERATELY NARROW. These factors touch the SCORELINE DISPLAY only.
They are never applied to p_home/p_draw/p_away, to the match pick, or to
anything the record is graded on. The match model is calibrated and working
(stated 48.3% against 56.0% actual over its first 25 picks) and is not being
retuned to make a scoreline board prettier.

THE GATE. Corrections are learned on the earlier seasons and scored on a
held-out tail the learning never saw. They ship only if holdout accuracy is not
materially worse (TOLERANCE_PP) AND variety genuinely improves. A change that
cost real accuracy would be a bad trade at any level of prettiness.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import config, dataset, publish
from leagues.model import LeagueModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "leagues" / "score_calibration.json"

GRID = 6              # cells 0..5 each side; beyond that the data is too thin
MIN_SUPPORT = 15      # a cell needs this many real occurrences to be corrected
HOLDOUT_FRAC = 0.25
TOLERANCE_PP = 0.5    # holdout accuracy may fall by at most this, in points


def _grids(model, matches):
    g, y = [], []
    for _, r in matches.iterrows():
        try:
            grid = model.predict(r["home"], r["away"])["grid"]
        except Exception:
            continue
        g.append(np.asarray(grid)[:GRID, :GRID])
        y.append((int(r["home_goals"]), int(r["away_goals"])))
    return np.array(g), y


def learn(grids, actuals) -> np.ndarray:
    """actual frequency / mean predicted probability, per cell."""
    corr = np.ones((GRID, GRID))
    counts = Counter(actuals)
    n = len(actuals)
    for h in range(GRID):
        for a in range(GRID):
            if counts[(h, a)] < MIN_SUPPORT:
                continue
            predicted = grids[:, h, a].mean()
            if predicted > 0:
                corr[h, a] = (counts[(h, a)] / n) / predicted
    return corr


def argmax_score(grid, corr=None):
    g = grid[:GRID, :GRID].astype(float)
    if corr is not None:
        g = g * corr
    total = g.sum()
    if total <= 0:
        return (1, 1)
    g = g / total
    h, a = np.unravel_index(int(np.argmax(g)), g.shape)
    return int(h), int(a)


def score(grids, actuals, corr=None):
    picks = Counter()
    hit = 0
    for grid, truth in zip(grids, actuals):
        pick = argmax_score(grid, corr)
        picks[pick] += 1
        hit += pick == truth
    n = len(actuals)
    top = picks.most_common(1)[0][1] if picks else 0
    return {"hit_rate_pct": round(100.0 * hit / n, 2),
            "distinct_scores": len(picks),
            "modal_share_pct": round(100.0 * top / n, 1),
            "n": n}


def build(league="PL") -> dict:
    matches = (dataset.build_matches(league)
               .dropna(subset=["home_goals", "away_goals"])
               .sort_values("date"))
    ref = pd.Timestamp.now("UTC").tz_localize(None)
    model = LeagueModel(**publish.model_params(league)).fit(matches, ref=ref)
    grids, actuals = _grids(model, matches)
    n = len(grids)
    split = int(n * (1 - HOLDOUT_FRAC))

    corr = learn(grids[:split], actuals[:split])       # TRAIN ONLY
    before = score(grids[split:], actuals[split:])
    after = score(grids[split:], actuals[split:], corr)

    delta = after["hit_rate_pct"] - before["hit_rate_pct"]
    more_varied = (after["distinct_scores"] > before["distinct_scores"]
                   and after["modal_share_pct"] < before["modal_share_pct"])
    accepted = bool(delta >= -TOLERANCE_PP and more_varied)

    # Ship factors fitted on EVERYTHING once the holdout has done its job -- the
    # split exists to test the idea, not to throw away a quarter of the evidence.
    final = learn(grids, actuals) if accepted else np.ones((GRID, GRID))
    return {
        "_note": ("Per-scoreline corrections for the exact-score DISPLAY only. "
                  "Never applied to p_home/p_draw/p_away, the match pick, or "
                  "anything graded. Generated by scripts/calibrate_scorelines.py "
                  "-- do not hand-edit."),
        "league": league,
        "grid": GRID,
        "accepted": accepted,
        "holdout": {"before": before, "after": after,
                    "delta_pp": round(delta, 2),
                    "tolerance_pp": TOLERANCE_PP},
        "factors": {f"{h}-{a}": round(float(final[h, a]), 4)
                    for h in range(GRID) for a in range(GRID)
                    if abs(final[h, a] - 1.0) > 1e-9},
    }


def main():
    report = build("PL")
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    b, a = report["holdout"]["before"], report["holdout"]["after"]
    print(f"holdout n={b['n']}")
    print(f"  before: {b['hit_rate_pct']}%  {b['distinct_scores']} scores, "
          f"modal share {b['modal_share_pct']}%")
    print(f"  after : {a['hit_rate_pct']}%  {a['distinct_scores']} scores, "
          f"modal share {a['modal_share_pct']}%")
    print(f"  delta : {report['holdout']['delta_pp']:+} pp   "
          f"ACCEPTED={report['accepted']}")
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
