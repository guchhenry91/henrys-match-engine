"""Where does the total-goals under-prediction actually come from?

Phase 2/3 found predicted mean total goals consistently below actual, worst
in Bundesliga. Isolates each candidate mechanism against the FULL fitted
history window per league (not walk-forward -- this is diagnosing the fit,
not benchmarking a candidate; the correction candidates in
total_goals_correction_experiment.py ARE walk-forward and causal).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import dataset
from leagues.model import LeagueModel

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]


def run_league(league: str):
    df = dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    ref = df["date"].max()

    out = {}

    # A: does xG itself run below actual goals over the fit window?
    xg_rows = df.dropna(subset=["home_xg", "away_xg"])
    if not xg_rows.empty:
        out["xg_vs_actual_goals"] = {
            "n_with_xg": int(len(xg_rows)),
            "mean_actual_total": round(float((xg_rows["home_goals"] + xg_rows["away_goals"]).mean()), 3),
            "mean_xg_total": round(float((xg_rows["home_xg"] + xg_rows["away_xg"]).mean()), 3),
        }

    # B: does the blended model's OWN fitted lambda average match the
    # goal-model's own level (i.e. does the re-centering invariant hold)?
    model_75 = LeagueModel(xg_weight=0.75).fit(df, ref=ref)
    model_0 = LeagueModel(xg_weight=0.0).fit(df, ref=ref)   # pure goals, no xG channel
    model_100 = LeagueModel(xg_weight=1.0).fit(df, ref=ref)  # pure xG deviations

    def mean_predicted_total(model):
        totals = []
        for _, r in df.iterrows():
            try:
                lh, la = model.lambdas(r["home"], r["away"])
            except KeyError:
                continue
            totals.append(lh + la)
        return float(np.mean(totals)) if totals else None

    actual_total = float((df["home_goals"] + df["away_goals"]).mean())
    out["xg_weight_sweep"] = {
        "actual_mean_total": round(actual_total, 3),
        "xg_weight_0.00_pure_goals": round(mean_predicted_total(model_0), 3),
        "xg_weight_0.75_current": round(mean_predicted_total(model_75), 3),
        "xg_weight_1.00_pure_xg": round(mean_predicted_total(model_100), 3),
    }

    # C: home advantage / league intercept -- fit on full history vs. only the
    # most recent 1 season's worth of matches, compare home_adv and the
    # goal-model's own level (ga_mean+gd_mean proxy via mean fitted lambda).
    recent_cutoff = ref - pd.Timedelta(days=365)
    recent_df = df[df["date"] >= recent_cutoff]
    if len(recent_df) > 100:
        model_recent = LeagueModel(xg_weight=0.75).fit(recent_df, ref=ref)
        out["home_adv_recent_vs_full"] = {
            "full_history_home_adv": round(model_75.home_adv, 4),
            "recent_1yr_only_home_adv": round(model_recent.home_adv, 4),
            "full_history_n": int(len(df)),
            "recent_1yr_n": int(len(recent_df)),
            "recent_1yr_actual_mean_total": round(float((recent_df["home_goals"] + recent_df["away_goals"]).mean()), 3),
        }

    # D: season-by-season actual scoring trend, straight from results (no
    # model involved) -- is there a real recent uptrend the long half-life
    # would lag behind?
    if "season" in df.columns:
        by_season = df.groupby("season").apply(
            lambda g: round(float((g["home_goals"] + g["away_goals"]).mean()), 3),
            include_groups=False)
        out["goals_per_game_by_season"] = by_season.to_dict()

    # E: shrinkage impact -- prior_strength=3.0 (current) vs 0 (no shrinkage),
    # same full-history fit, mean predicted total.
    model_noshrink = LeagueModel(xg_weight=0.75, prior_strength=0.0).fit(df, ref=ref)
    out["shrinkage_impact"] = {
        "actual_mean_total": round(actual_total, 3),
        "prior_strength_3.0_current": round(mean_predicted_total(model_75), 3),
        "prior_strength_0_no_shrinkage": round(mean_predicted_total(model_noshrink), 3),
    }

    return out


def run():
    report = {}
    for league in LEAGUES:
        print(f"--- {league} ---")
        report[league] = run_league(league)
        print(json.dumps(report[league], indent=2))
    return report


def main():
    rep = run()
    path = ROOT / "data-raw" / "leagues" / "total_goals_mechanism_audit.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
