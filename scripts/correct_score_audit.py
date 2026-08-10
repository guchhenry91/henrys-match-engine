"""Phase 1: root-cause audit + fixture-level diagnostic table for the
correct-score engine, across all four leagues.

Read-only: fits the production model exactly (correct_score_common) and
inspects it. Writes no data/leagues output, changes no production behaviour.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leagues.model import scoreline_grid, outcome_probs, top_scorelines, score_for_outcome
from scripts.correct_score_common import fit_production_model, upcoming_fixtures, near_term_fixtures

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]


def fixture_diagnostics(model, home: str, away: str) -> dict:
    lh, la = model.lambdas(home, away)
    grid = scoreline_grid(lh, la, model.rho)
    ph, pdw, pa = outcome_probs(grid)
    pick_type = "home" if ph >= pdw and ph >= pa else ("draw" if pdw >= pa else "away")
    pick_team = {"home": home, "away": away, "draw": "Draw"}[pick_type]

    tops = top_scorelines(grid, n=6)
    p11 = next((t["pct"] for t in tops if t["score"] == "1-1"), None)
    if p11 is None:
        p11 = round(100.0 * float(grid[1, 1]), 1)
    gap_to_next = round(tops[0]["pct"] - tops[1]["pct"], 2) if len(tops) > 1 else None

    return {
        "home": home, "away": away,
        "lambda_home": round(lh, 3), "lambda_away": round(la, 3),
        "rho": round(model.rho, 4),
        "p_1_1": p11,
        "top6_scores": tops,
        "gap_top1_vs_top2_pp": gap_to_next,
        "p_home": round(ph, 4), "p_draw": round(pdw, 4), "p_away": round(pa, 4),
        "selected_1x2_pick": pick_team, "selected_1x2_pick_type": pick_type,
        "grid_mode": tops[0]["score"],
        "score_conditional_on_pick": score_for_outcome(grid, pick_type),
    }


def league_compression_diagnostics(model, squad_teams, matches, ref) -> dict:
    """Checks: is the attack/defence spread compressed? How much shrinkage
    exposure does each team have? What fraction of teams needed a fallback
    (rather than second-tier) prior?"""
    from leagues.weights import decay_weights
    w = decay_weights(matches["date"], ref=ref, xi=model.xi).to_numpy()

    eff_by_team = {}
    for team in squad_teams:
        mask = ((matches["home"] == team) | (matches["away"] == team)).to_numpy()
        eff_by_team[team] = round(float(w[mask].sum()), 1) if len(mask) == len(w) else 0.0

    atk_vals = [model.attack[t] for t in squad_teams if t in model.attack]
    def_vals = [model.defence[t] for t in squad_teams if t in model.defence]

    prior_source = getattr(model, "prior_source", {})
    n_second_tier = sum(1 for v in prior_source.values() if v == "second_tier")
    n_fallback = sum(1 for v in prior_source.values() if v == "weakest_fallback")

    return {
        "n_teams": len(squad_teams),
        "attack_std": round(float(np.std(atk_vals)), 4) if atk_vals else None,
        "attack_range": [round(float(min(atk_vals)), 3), round(float(max(atk_vals)), 3)] if atk_vals else None,
        "defence_std": round(float(np.std(def_vals)), 4) if def_vals else None,
        "min_effective_matches": round(float(min(eff_by_team.values())), 1) if eff_by_team else None,
        "median_effective_matches": round(float(np.median(list(eff_by_team.values()))), 1) if eff_by_team else None,
        "teams_below_10_effective_matches": sorted(t for t, e in eff_by_team.items() if e < 10),
        "prior_strength": model.prior_strength,
        "n_second_tier_priors": n_second_tier,
        "n_weakest_fallback_priors": n_fallback,
        "fallback_teams": sorted(t for t, v in prior_source.items() if v == "weakest_fallback"),
    }


def _grid_mode_stats(rows):
    n = len(rows)
    n11 = sum(1 for r in rows if r["grid_mode"] == "1-1")
    return {
        "n_fixtures": n,
        "n_grid_mode_is_1_1": n11,
        "pct_grid_mode_is_1_1": round(100 * n11 / n, 1) if n else None,
        "n_1x2_pick_is_draw": sum(1 for r in rows if r["selected_1x2_pick_type"] == "draw"),
    }


def run():
    report = {"leagues": {}}
    all_near_term_rows = []
    all_full_season_rows = []

    for league in LEAGUES:
        print(f"--- {league} ---")
        model, fx = fit_production_model(league)
        near_term = near_term_fixtures(fx)
        full_season = upcoming_fixtures(fx)
        squad_teams = sorted(set(fx["home"]) | set(fx["away"]))

        from leagues import dataset
        matches = dataset.build_matches(league)
        played = fx[fx["played"]].copy()
        if not played.empty:
            add = played[["date", "home", "away", "home_goals", "away_goals"]].copy()
            add["date"] = pd.to_datetime(add["date"]).dt.tz_localize(None)
            matches = pd.concat([matches, add], ignore_index=True)

        comp = league_compression_diagnostics(model, squad_teams, matches, pd.Timestamp.now("UTC"))

        def build_rows(fixture_frame, sink):
            rows = []
            for _, m in fixture_frame.iterrows():
                try:
                    d = fixture_diagnostics(model, m["home"], m["away"])
                except KeyError as exc:
                    print(f"  skip {m['home']} v {m['away']}: {exc}")
                    continue
                d["league"] = league
                rows.append(d)
                sink.append(d)
            return rows

        near_term_rows = build_rows(near_term, all_near_term_rows)
        full_season_rows = build_rows(full_season, all_full_season_rows)

        near_stats = _grid_mode_stats(near_term_rows)
        full_stats = _grid_mode_stats(full_season_rows)
        report["leagues"][league] = {
            "compression_diagnostics": comp,
            "near_term_window": near_stats,          # matches what the site actually ships
            "full_remaining_season": full_stats,      # larger-sample context
            "fixtures": near_term_rows,               # detailed table = near-term only
        }
        print(f"  rho={model.rho:.4f}  near-term 1-1 grid-mode: "
              f"{near_stats['n_grid_mode_is_1_1']}/{near_stats['n_fixtures']}  "
              f"full-season: {full_stats['n_grid_mode_is_1_1']}/{full_stats['n_fixtures']}  "
              f"attack_std={comp['attack_std']}  fallback_priors={comp['n_weakest_fallback_priors']}")

    report["combined_near_term"] = _grid_mode_stats(all_near_term_rows)
    report["combined_full_season"] = _grid_mode_stats(all_full_season_rows)
    return report


def main():
    rep = run()
    path = ROOT / "data-raw" / "leagues" / "correct_score_audit_phase1.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    print(f"\nCOMBINED (near-term, matches site): {rep['combined_near_term']}")
    print(f"COMBINED (full remaining season): {rep['combined_full_season']}")


if __name__ == "__main__":
    main()
