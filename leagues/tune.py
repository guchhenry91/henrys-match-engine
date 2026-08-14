"""The match-model gate: pick xi/xg_weight per league, then prove the winner on
data the sweep never saw.

TWO STAGES, because a grid search that reports its own best score is not a gate.

  SWEEP    -- score all twelve configs by walk-forward RPS on the seasons BEFORE
              the holdout, and take the lowest.
  HOLDOUT  -- re-score that winner, and the incumbent it would replace, on the
              final season only. Nothing about that season influenced the choice.

`min(rps)` over twelve configs scored on the same fixtures is a biased estimate of
the winner's true skill: some of the gap is the winner fitting that particular
sample's noise. Measured here at ~0.00075 RPS, which is small enough that it does
not overturn any conclusion -- and that is exactly why it is worth fixing rather
than arguing about. The number now means what the README says it means.

PROMOTION RULE. Winning the sweep does NOT ship a config. A challenger replaces
the incumbent only if it improves holdout RPS with a paired 95% bootstrap interval
entirely below zero -- the same bar the correct-score audit used. Paired, because
both models are scored on the identical fixtures, so fixture difficulty cancels.
Rejected candidates are written to the report rather than discarded: a gate that
only records its winners cannot be audited later.

The incumbent is whatever `LeagueModel()` defaults to, which is what publish.py
actually ships. Every config the rule promotes lands in release_policy.json, and
publish.py reads that file -- so the shipped parameters and the evidence for them
are produced by the same run and cannot drift apart.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

from leagues import backtest, config, dataset, second_tier
from leagues.model import XG_WEIGHT
from leagues.weights import XI_PER_DAY

XIS = [0.0018, 0.003, 0.0045]
XGWS = [0.0, 0.5, 0.75, 1.0]

# What publish.py ships today. A challenger must beat THIS, not the sweep's
# runner-up, or the gate would churn parameters on noise every time it runs.
INCUMBENT = {"xi": XI_PER_DAY, "xg_weight": XG_WEIGHT}

# The most recent completed season is never used to choose anything.
HOLDOUT_SEASONS = 1
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260813        # fixed so the gate's verdict is reproducible

# Probability tiers the Best Picks board is billed on (publish.BEST_PICK_MIN_PROB
# is one of these). Generated here rather than pasted into publish.py.
TIERS = [0.0, 0.60, 0.65, 0.70]


def holdout_start(league: str) -> pd.Timestamp:
    """First day of the held-out season. July boundary, matching season_start_year."""
    seasons = config.get(league).history_seasons
    first_held = seasons[-HOLDOUT_SEASONS]
    return pd.Timestamp(f"{second_tier.season_code_start_year(first_held)}-07-01")


def paired_bootstrap(challenger: np.ndarray, incumbent: np.ndarray,
                     n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """95% CI for mean(challenger - incumbent) per-match RPS, resampling FIXTURES.

    Negative throughout = the challenger is genuinely better. An interval that
    straddles zero means the sweep's improvement is indistinguishable from noise,
    however large it looks.
    """
    d = np.asarray(challenger, dtype=float) - np.asarray(incumbent, dtype=float)
    if len(d) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n": int(len(d))}


def tier_hit_rates(res: pd.DataFrame) -> list:
    """Hit rate of the model's argmax pick at each probability tier.

    This is what the Best Picks board is billed on. It was two literals in
    publish.py; now it is computed from the same walk-forward that justifies it.
    """
    if res.empty:
        return []
    p = res[["p_home", "p_draw", "p_away"]].to_numpy()
    y = res["outcome"].to_numpy()
    top = p.max(axis=1)
    hit = p.argmax(axis=1) == y
    out = []
    for t in TIERS:
        m = top >= t
        k = int(m.sum())
        out.append({"min_prob": t, "n": k,
                    "hit_rate_pct": round(100.0 * float(hit[m].mean()), 1) if k else None})
    return out


def _score_split(res: pd.DataFrame, boundary: pd.Timestamp):
    d = pd.to_datetime(res["date"])
    return res[d < boundary], res[d >= boundary]


def tune(league: str) -> dict:
    matches = dataset.build_matches(league)
    boundary = holdout_start(league)
    runs, best, best_rps = {}, None, float("inf")

    for xi, xgw in itertools.product(XIS, XGWS):
        res = backtest.walk_forward(matches, league, xi=xi, xg_weight=xgw)
        if res.empty:
            continue
        sweep, held = _score_split(res, boundary)
        if sweep.empty:
            continue
        s = backtest.score(sweep)
        runs[(xi, xgw)] = {"sweep": s, "sweep_rows": sweep, "held_rows": held}
        seeded = s.get("n_seeded", 0)
        print(f"  xi={xi:<7} xg_w={xgw:<5} n={s['n']:<5} seeded={seeded:<4} "
              f"acc={s['accuracy']:.3f} sweep_rps={s['rps']:.4f}")
        if s["rps"] < best_rps:
            best, best_rps = (xi, xgw), s["rps"]

    if best is None:
        return {}

    inc_key = (INCUMBENT["xi"], INCUMBENT["xg_weight"])
    if inc_key not in runs:
        # The incumbent must be inside the grid or there is nothing to promote
        # against. Fail loudly rather than silently comparing to the runner-up.
        raise RuntimeError(
            f"{league}: incumbent {inc_key} not in the sweep grid {XIS}x{XGWS}")

    winner_held = runs[best]["held_rows"]
    inc_held = runs[inc_key]["held_rows"]
    held_score = backtest.score(winner_held) if not winner_held.empty else {}
    inc_score = backtest.score(inc_held) if not inc_held.empty else {}

    # Pair on the identical fixtures. Both walks cover the same dates, but a fit
    # failure can drop a cutoff from one and not the other, so align explicitly.
    key = ["date", "home", "away"]
    pair = winner_held.merge(inc_held, on=key, suffixes=("_c", "_i"))
    ci = {"mean": None, "lo": None, "hi": None, "n": 0}
    if not pair.empty:
        c = backtest.rps_per_match(
            pair[["p_home_c", "p_draw_c", "p_away_c"]].to_numpy(), pair["outcome_c"])
        i = backtest.rps_per_match(
            pair[["p_home_i", "p_draw_i", "p_away_i"]].to_numpy(), pair["outcome_i"])
        ci = paired_bootstrap(c, i)

    is_incumbent = best == inc_key
    promoted = bool(ci["hi"] is not None and ci["hi"] < 0)
    shipped = best if (is_incumbent or promoted) else inc_key

    rejected = []
    for k, v in runs.items():
        if k == shipped:
            continue
        rejected.append({"xi": k[0], "xg_weight": k[1],
                         "sweep_rps": round(v["sweep"]["rps"], 5),
                         "sweep_n": v["sweep"]["n"],
                         "reason": ("lost the sweep" if k != best else
                                    "won the sweep but the paired holdout CI "
                                    "was not entirely below zero")})
    rejected.sort(key=lambda r: r["sweep_rps"])

    full = pd.concat([runs[shipped]["sweep_rows"], runs[shipped]["held_rows"]],
                     ignore_index=True)
    return {
        # Shipped parameters first: this block is what release_policy.json is
        # built from, and what publish.py runs.
        "xi": shipped[0], "xg_weight": shipped[1],
        "selection": ("incumbent retained" if not promoted else "challenger promoted"),
        "incumbent": {"xi": inc_key[0], "xg_weight": inc_key[1], **inc_score},
        "sweep_winner": {"xi": best[0], "xg_weight": best[1],
                         **runs[best]["sweep"]},
        "holdout": {"season_from": str(boundary.date()), **held_score},
        "holdout_vs_incumbent_rps_ci": ci,
        "promotion_rule": ("challenger ships only if the paired 95% bootstrap CI "
                           "of holdout RPS difference lies entirely below zero"),
        "rejected": rejected,
        "tiers_full": tier_hit_rates(full),
        "tiers_holdout": tier_hit_rates(runs[shipped]["held_rows"]),
        # Back-compatible top-level scores: the SHIPPED config over the whole
        # walk, which is what earlier versions of this file reported.
        **backtest.score(full),
    }


def pooled(report: dict) -> dict:
    """Cross-league tier hit rates -- the numbers the Best Picks board is billed
    on -- plus the per-league spread, because one pooled figure hides it. The
    pooled 77.4% at p>=0.65 has ranged from ~71% to ~86% by league."""
    tiers = {}
    for lg, b in report.items():
        for t in b.get("tiers_full", []):
            if t["n"]:
                slot = tiers.setdefault(t["min_prob"], {"hits": 0.0, "n": 0, "by_league": {}})
                slot["hits"] += t["hit_rate_pct"] / 100.0 * t["n"]
                slot["n"] += t["n"]
                slot["by_league"][lg] = {"n": t["n"], "hit_rate_pct": t["hit_rate_pct"]}
    out = []
    for t in sorted(tiers):
        s = tiers[t]
        rates = [v["hit_rate_pct"] for v in s["by_league"].values()]
        out.append({"min_prob": t, "n": s["n"],
                    "hit_rate_pct": round(100.0 * s["hits"] / s["n"], 1),
                    "league_min_pct": min(rates), "league_max_pct": max(rates),
                    "by_league": s["by_league"]})
    return {"tiers": out}


def release_policy(report: dict) -> dict:
    """The parameters publish.py runs, derived from the report in the same run.

    Hand-maintained constants that "happen to agree with the report" drift the
    moment either side is touched, and nothing fails when they do. Generating
    them makes the rule and its evidence the same artefact.
    """
    return {
        "_generated_by": "python -m leagues.tune",
        "_note": "DO NOT EDIT BY HAND. Regenerate with the gate; publish.py reads "
                 "this file to pick each league's xi/xg_weight.",
        "leagues": {lg: {"xi": b["xi"], "xg_weight": b["xg_weight"],
                         "selection": b.get("selection"),
                         "holdout_rps": b.get("holdout", {}).get("rps"),
                         "holdout_n": b.get("holdout", {}).get("n")}
                    for lg, b in report.items() if b},
    }


def main():
    leagues = sys.argv[1:] or list(config.LEAGUES)
    report = {}
    for lg in leagues:
        print(f"\n=== {config.get(lg).name} ===")
        b = tune(lg)
        report[lg] = b
        if not b:
            print("  NO RESULT")
            continue
        print(f"  SHIPPED  xi={b['xi']} xg_weight={b['xg_weight']}  ({b['selection']})")
        sw = b["sweep_winner"]
        print(f"  sweep won by xi={sw['xi']} xg_weight={sw['xg_weight']} "
              f"rps={sw['rps']:.4f} on n={sw['n']}")
        h, ci = b["holdout"], b["holdout_vs_incumbent_rps_ci"]
        if h.get("n"):
            print(f"  HOLDOUT ({h['season_from']}+): n={h['n']} "
                  f"acc {h['accuracy']:.1%} RPS {h['rps']:.4f}")
        if ci["mean"] is not None:
            print(f"  vs incumbent on holdout: {ci['mean']:+.5f} RPS "
                  f"[95% CI {ci['lo']:+.5f}, {ci['hi']:+.5f}] on n={ci['n']}")
        if "market_rps" in b:
            print(f"  MARKET: acc {b['market_accuracy']:.1%}  RPS {b['market_rps']:.4f}")
            gap = b["model_rps_on_market_subset"] - b["market_rps"]
            print(f"  GAP   : RPS {gap:+.4f}  (positive = market better, expected)")
        if b.get("n_seeded"):
            print(f"  {b['n_seeded']} of {b['n']} fixtures involved a club seeded "
                  f"from a prior (dropped entirely before M-01)")

    report["_pooled"] = pooled({k: v for k, v in report.items() if v and not k.startswith("_")})
    os.makedirs("data-raw/leagues", exist_ok=True)
    with open("data-raw/leagues/backtest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nWrote data-raw/leagues/backtest_report.json")

    if set(leagues) == set(config.LEAGUES):
        # Only rewrite the shipped policy from a COMPLETE run. A single-league
        # invocation must not silently drop the other three leagues' entries.
        pol = release_policy({k: v for k, v in report.items() if not k.startswith("_")})
        with open("data-raw/leagues/release_policy.json", "w") as f:
            json.dump(pol, f, indent=2, default=float)
        print("Wrote data-raw/leagues/release_policy.json")
    else:
        print(f"Partial run ({', '.join(leagues)}) -- release_policy.json left unchanged")

    for t in report["_pooled"]["tiers"]:
        print(f"  pooled p>={t['min_prob']:.2f}: {t['hit_rate_pct']}% of {t['n']} "
              f"(by league {t['league_min_pct']}%-{t['league_max_pct']}%)")


if __name__ == "__main__":
    main()
