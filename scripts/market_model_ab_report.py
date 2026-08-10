"""Phase 4a documentation + constrained blend-weight tradeoff.

Produces the A/B split the user asked for:
  A. Independent Henry model  -- team data only, no odds.
  B. Market-calibrated consensus model -- de-vigged 1X2 + O/U, labelled as
     market-informed, not as Henry beating the market (optimal weight on
     Henry's own lambdas is 0.00 in every league per market_combined_experiment.json --
     this is "trust the market's total+direction," not "Henry has alpha").

Also: bookmaker-stability check (Bet365-only columns vs the Avg-across-many-
bookmakers columns already used), explicit market-only baseline, and a fixed-
weight tradeoff curve (0%, 10%, 20%, 30%, 40%, 50% retained on Henry's own
lambdas) since the OPEN optimum landing on 0% doesn't mean intermediate
weights are equally bad -- report the actual curve.

DATA PROVENANCE / LEAKAGE STATEMENT (read before citing these numbers as a
production justification):
  - Odds are football-data.co.uk's historical archive: "AvgC{H,D,A}" (closing
    average across many bookmakers, first choice), falling back to
    "B365C{H,D,A}" (Bet365 closing) falling back to "B365{H,D,A}" (Bet365
    pre-closing) -- see leagues/history.py's ODDS_SETS/OU_ODDS_SETS. These
    are CLOSING (or near-closing) lines: the best odds available right before
    kickoff, not odds available at whatever earlier moment this pipeline
    actually publishes a pick (MATCHWEEKS_AHEAD=1 means picks can be
    generated several days to a week before kickoff).
  - This backtest is therefore answering "if closing-market information were
    available, would it improve the grid" -- a valid test of whether market
    information helps AT ALL, but NOT a simulation of this pipeline's actual
    early-week publish cadence. A live deployment would need odds captured
    AT PUBLISH TIME, which are thinner/less mature than closing lines and
    would likely show a SMALLER benefit than reported here. This gap is not
    resolved by this script -- flagged for the final report's risk section.
  - No leakage within the backtest itself: each walk-forward cutoff only
    uses train matches with date < cutoff, and each held-out match's own
    odds are its own closing line, never a future match's odds or a later
    snapshot.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import dataset
from leagues.model import LeagueModel, scoreline_grid
from scripts.ou_market_experiment import devig_over_under, implied_total_lambda
from scripts.market_combined_experiment import devig_1x2, solve_split
from scripts.correct_score_benchmark import per_match_grid_probs, paired_ci

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]
FIXED_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]


def collect_rows(league: str, xi=0.003, xg_weight=0.75, min_train=760, step_days=7):
    df = (dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))
    rows = []
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        try:
            model = LeagueModel(xi=xi, xg_weight=xg_weight).fit(train, ref=cutoff)
        except Exception:
            continue
        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            row = {"hg": int(m["home_goals"]), "ag": int(m["away_goals"]),
                   "lh_model": lh, "la_model": la, "rho": model.rho}
            # Avg-across-bookmakers market (already the primary column set)
            if all(pd.notna(m.get(c)) for c in ["odds_h", "odds_d", "odds_a", "odds_over25", "odds_under25"]):
                p1x2 = devig_1x2(m["odds_h"], m["odds_d"], m["odds_a"])
                p_over = devig_over_under(m["odds_over25"], m["odds_under25"])
                total = implied_total_lambda(p_over)
                r = solve_split(total, model.rho, p1x2)
                row["lh_market"] = r * total
                row["la_market"] = (1 - r) * total
            rows.append(row)
    return rows


def bookmaker_stability_rows(league: str, xi=0.003, xg_weight=0.75, min_train=760, step_days=7):
    """Same split-solve method, but the DIRECTION comes from Bet365's own raw
    columns (odds_b365_h/d/a, always Bet365 regardless of what the primary
    fallback-resolved odds_h/d/a picked) instead of the Avg-across-bookmakers
    consensus. Total still comes from Avg O/U (no per-bookmaker O/U archive
    exists in this feed) -- isolates whether the 1X2 DIRECTION specifically is
    bookmaker-fragile or consistent, the piece a single-bookmaker column set
    can actually test."""
    df = (dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
          .sort_values("date").reset_index(drop=True))
    rows = []
    start = df.loc[min_train, "date"]
    for cutoff in pd.date_range(start, df["date"].max(), freq=f"{step_days}D"):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=step_days))]
        if train.empty or test.empty:
            continue
        try:
            model = LeagueModel(xi=xi, xg_weight=xg_weight).fit(train, ref=cutoff)
        except Exception:
            continue
        for _, m in test.iterrows():
            try:
                lh, la = model.lambdas(m["home"], m["away"])
            except KeyError:
                continue
            row = {"hg": int(m["home_goals"]), "ag": int(m["away_goals"]),
                   "lh_model": lh, "la_model": la, "rho": model.rho}
            has_ou = pd.notna(m.get("odds_over25")) and pd.notna(m.get("odds_under25"))
            if has_ou:
                p_over = devig_over_under(m["odds_over25"], m["odds_under25"])
                total = implied_total_lambda(p_over)
                if all(pd.notna(m.get(c)) for c in ["odds_h", "odds_d", "odds_a"]):
                    p1x2_avg = devig_1x2(m["odds_h"], m["odds_d"], m["odds_a"])
                    r_avg = solve_split(total, model.rho, p1x2_avg)
                    row["lh_market_avg"] = r_avg * total
                    row["la_market_avg"] = (1 - r_avg) * total
                if all(pd.notna(m.get(c)) for c in ["odds_b365_h", "odds_b365_d", "odds_b365_a"]):
                    p1x2_365 = devig_1x2(m["odds_b365_h"], m["odds_b365_d"], m["odds_b365_a"])
                    r_365 = solve_split(total, model.rho, p1x2_365)
                    row["lh_market_b365"] = r_365 * total
                    row["la_market_b365"] = (1 - r_365) * total
            rows.append(row)
    return rows


def eval_weight(rows_mkt, w):
    def gfn(r):
        lh = w * r["lh_model"] + (1 - w) * r["lh_market"]
        la = w * r["la_model"] + (1 - w) * r["la_market"]
        return scoreline_grid(lh, la, r["rho"])
    return per_match_grid_probs(rows_mkt, gfn)


def run():
    report = {"provenance": {
        "odds_columns_used": "AvgCH/CD/CA + Avg>2.5/<2.5 (fallback: B365CH/CD/CA, then B365H/D/A; "
                             "fallback: B365>2.5/<2.5, Max>2.5/<2.5) -- see leagues/history.py",
        "odds_timing": "CLOSING (or near-closing) lines from football-data.co.uk's historical archive, "
                      "NOT odds available at this pipeline's actual publish timestamp "
                      "(MATCHWEEKS_AHEAD=1 means picks can be generated days before kickoff). "
                      "This is a valid test of whether market info helps AT ALL, not a simulation "
                      "of the live publish cadence -- see this file's module docstring.",
        "correct_score_market_available": False,
        "correct_score_market_note": ("API-Football's own pre-match odds (including its 'Exact Score' "
                                      "market) returned empty for every fixture tested -- past, upcoming, "
                                      "and a Champions League final -- despite being listed in the "
                                      "dashboard's package features; only in-play/live odds actually "
                                      "returned data. See .github/workflows/odds-diagnostic.yml and this "
                                      "session's diagnostic runs. No correct-score market comparison is "
                                      "possible with currently-available data."),
    }, "leagues": {}}

    for league in LEAGUES:
        print(f"--- {league} ---")
        rows = collect_rows(league)
        rows_mkt = [r for r in rows if "lh_market" in r]
        n_total, n_mkt = len(rows), len(rows_mkt)

        half = n_mkt // 2
        te = rows_mkt[half:]  # held-out half, same split as market_combined_experiment

        weight_curve = {}
        for w in FIXED_WEIGHTS:
            p = eval_weight(te, w)
            weight_curve[w] = round(float(-np.log(p).mean()), 4)

        p_market_only = eval_weight(te, 0.0)
        p_henry_only = eval_weight(te, 1.0)
        lo, hi = paired_ci(-np.log(p_henry_only), -np.log(p_market_only))

        # Bookmaker stability: Bet365-only direction vs Avg-across-bookmakers
        # direction, same total, same held-out matches.
        bm_rows = bookmaker_stability_rows(league)
        bm_both = [r for r in bm_rows if "lh_market_avg" in r and "lh_market_b365" in r]
        bm_half = len(bm_both) // 2
        bm_te = bm_both[bm_half:]
        bm_stability = None
        if len(bm_te) >= 100:
            def gfn_avg(r):
                return scoreline_grid(r["lh_market_avg"], r["la_market_avg"], r["rho"])
            def gfn_365(r):
                return scoreline_grid(r["lh_market_b365"], r["la_market_b365"], r["rho"])
            p_avg = per_match_grid_probs(bm_te, gfn_avg)
            p_365 = per_match_grid_probs(bm_te, gfn_365)
            lo_bm, hi_bm = paired_ci(-np.log(p_avg), -np.log(p_365))
            bm_stability = {
                "n": len(bm_te),
                "avg_consensus_logloss": round(float(-np.log(p_avg).mean()), 4),
                "bet365_only_logloss": round(float(-np.log(p_365).mean()), 4),
                "logloss_delta_avg_minus_b365_95ci": [round(lo_bm, 4), round(hi_bm, 4)],
                "materially_different": bool(lo_bm > 0 or hi_bm < 0),
            }

        report["leagues"][league] = {
            "n_total": n_total, "n_with_odds": n_mkt, "test_n": len(te),
            "coverage_pct": round(100 * n_mkt / n_total, 1),
            "A_independent_henry_logloss": weight_curve[1.0],
            "B_market_calibrated_logloss": weight_curve[0.0],
            "logloss_delta_henry_minus_market_95ci": [round(lo, 4), round(hi, 4)],
            "market_significantly_better": bool(lo > 0),
            "fixed_weight_tradeoff_curve": {
                f"{int(w*100)}pct_henry_{int((1-w)*100)}pct_market": ll
                for w, ll in weight_curve.items()
            },
            "bookmaker_stability_avg_vs_bet365": bm_stability,
        }
        print(f"  A(100% Henry)={weight_curve[1.0]}  B(0% Henry/100% market)={weight_curve[0.0]}  "
              f"curve={weight_curve}  bookmaker_stability={bm_stability}")

    return report


def main():
    rep = run()
    path = ROOT / "data-raw" / "leagues" / "market_model_ab_report.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
