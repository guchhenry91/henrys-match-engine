"""Grid-search xi and xg_weight per league by walk-forward RPS, then report
model vs de-vigged market. This is the credibility gate."""
import itertools
import json
import os
import sys

from leagues import backtest, config, dataset

XIS = [0.0018, 0.003, 0.0045]
XGWS = [0.0, 0.5, 0.75, 1.0]


def tune(league: str) -> dict:
    matches = dataset.build_matches(league)
    best, best_rps = None, float("inf")
    for xi, xgw in itertools.product(XIS, XGWS):
        res = backtest.walk_forward(matches, xi=xi, xg_weight=xgw)
        if res.empty:
            continue
        s = backtest.score(res)
        print(f"  xi={xi:<7} xg_w={xgw:<5} n={s['n']:<5} "
              f"acc={s['accuracy']:.3f} rps={s['rps']:.4f}")
        if s["rps"] < best_rps:
            best, best_rps = {"xi": xi, "xg_weight": xgw, **s}, s["rps"]
    return best


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
        print(f"  BEST xi={b['xi']} xg_weight={b['xg_weight']}")
        print(f"  MODEL : acc {b['accuracy']:.1%}  RPS {b['rps']:.4f}  Brier {b['brier']:.4f}")
        if "market_rps" in b:
            print(f"  MARKET: acc {b['market_accuracy']:.1%}  RPS {b['market_rps']:.4f}  "
                  f"Brier {b['market_brier']:.4f}")
            gap = b["model_rps_on_market_subset"] - b["market_rps"]
            print(f"  GAP   : RPS {gap:+.4f}  (positive = market better, expected)")
    os.makedirs("data-raw/leagues", exist_ok=True)
    with open("data-raw/leagues/backtest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nWrote data-raw/leagues/backtest_report.json")


if __name__ == "__main__":
    main()
