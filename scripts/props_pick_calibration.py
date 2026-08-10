"""Are the PUBLISHED player-prop probabilities honest? (the end-to-end gate)

props_backtest.py gates the per-90 RATE estimation. It deliberately stops there,
and its docstring explains why a general per-match calibration curve would be
biased: Understat's shot events contain only players who took a shot, so "played
but never shot" is indistinguishable from "did not play", and every player in
such a sample looks sharper than he is.

That argument is correct for a curve over ALL players. It does NOT block the
question this script asks, which is narrower and is the one a bettor actually
cares about:

    When the board publishes "80% to have 2+ shots", do 80% of THOSE PICKS win?

The selection problem disappears because we only score picks the model actually
published, and we grade them with the SAME harsh rule production already uses
live (leagues/publish.py + CLAUDE.md): a player with no shot row grades WRONG,
never void, because the feed cannot prove he played. Whatever bias that rule
carries, it is identical in the backtest and in the live record, so the measured
hit rate is directly comparable to the published probability.

CAUSAL: rates and the match model are fitted on seasons up to TRAIN_THROUGH only;
every scored fixture is from the later TEST_SEASON. Nothing from the test season
informs a prediction about it.

NOT WIRED INTO ANYTHING. Diagnostic only -- run it, read it, decide.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import config, players, props
from leagues.model import LeagueModel
from leagues.publish import PLAYER_PICK_MIN_PROB, PROP_FIELD
from leagues import dataset

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]
TRAIN_THROUGH = "2425"
TEST_SEASON = "2526"


def run_league(league: str) -> dict:
    lg = config.get(league)
    train_seasons = [s for s in lg.history_seasons if s <= TRAIN_THROUGH]

    # --- rates + minutes from TRAINING seasons only -------------------------
    logs = players.fetch_player_logs(league)
    logs_train = logs[logs["season"].isin(train_seasons)]
    if logs_train.empty:
        return {"league": league, "error": "no training logs"}
    ref = players.season_end(TRAIN_THROUGH)
    rates = props.player_rates(logs_train, ref=ref)
    # MIRROR PRODUCTION. leagues/publish.py filters the rate table to
    # current_squad() before building any prop. Without it the table carries five
    # seasons of departed players, and because the shots market is NOT rescaled to
    # a team total, a striker who left in 2022 keeps a high shot rate, clears the
    # 0.70 bar, never plays, and grades as a loss. That alone dragged the first run
    # of this script to an absurd 11% hit rate in Ligue 1 (the highest-churn league)
    # -- a bug in the measurement, not in the model.
    squad = players.current_squad(logs_train)
    rates = rates[rates["player"].isin(squad)]
    # Roster reconciliation (the second production filter) cannot be replayed:
    # no historical roster snapshots exist. So this still scores a few players who
    # left between the training season and the test season, which biases the
    # measured hit rate DOWN. Read any shortfall as an upper bound on
    # overconfidence, not a point estimate.
    league_matches = 2 * (lg.n_teams - 1)
    exp_minutes = players.expected_minutes(logs_train, matches_per_season=league_matches)
    playing_time = players.playing_time(logs_train, matches_per_season=league_matches)

    # --- match model fitted on TRAINING matches only ------------------------
    matches = dataset.build_matches(league).dropna(subset=["home_goals", "away_goals"])
    matches["date"] = pd.to_datetime(matches["date"])
    cutoff = players.season_end(TRAIN_THROUGH)
    train_m = matches[matches["date"] <= cutoff]
    model = LeagueModel().fit(train_m, ref=cutoff)

    # --- ground truth: per-player per-match actuals in the TEST season ------
    actual = players.match_player_stats(league, seasons=[TEST_SEASON])
    if actual.empty:
        return {"league": league, "error": "no shot-event actuals (known Bundesliga crash)"}
    actual["date"] = pd.to_datetime(actual["date"]).dt.normalize()
    got = {}
    for _, r in actual.iterrows():
        got[(r["date"], r["player"])] = (r["goals"], r["shots"], r["sot"])
    played_days = sorted(set(actual["date"]))
    active_players = set(actual["player"])

    # test fixtures = real results in the test season
    test_m = matches[matches["date"] > cutoff].copy()
    test_m["day"] = test_m["date"].dt.normalize()

    rows = []
    for _, m in test_m.iterrows():
        home, away, day = m["home"], m["away"], m["day"]
        if day not in played_days:
            continue                      # no shot feed that day -> cannot grade
        try:
            lh, la = model.lambdas(home, away)
        except KeyError:
            continue
        sq = props.match_props(rates, home, away, lh, la,
                               minutes=exp_minutes, playing_time=playing_time)
        for p in sq:
            for market, field in PROP_FIELD.items():
                prob = p.get(field)
                if prob is None:
                    continue
                prob = float(prob) / 100.0
                if prob < PLAYER_PICK_MIN_PROB[market]:
                    continue              # not published -> not scored
                g, s, so = got.get((day, p["player"]), (0, 0, 0))
                win = (g >= 1) if market == "goal" else (
                      (s >= 2) if market == "shots" else (so >= 1))
                rows.append({"market": market, "p": prob, "win": bool(win),
                             "player": p["player"], "team": p["team"],
                             # Did this player appear in the test season AT ALL?
                             # False means he had left the league, so grading him
                             # a loss measures the missing roster filter, not the
                             # model. Used for the tighter of the two bounds.
                             "active": p["player"] in active_players})

    if not rows:
        return {"league": league, "error": "no published-bar picks generated in test season"}
    df = pd.DataFrame(rows)

    out = {"league": league, "n_picks": int(len(df)),
           "pct_picks_on_players_absent_all_season":
               round(100 * float((~df["active"]).mean()), 1),
           "by_market": {}, "by_market_active_only": {}}
    for market in ["goal", "shots", "sot"]:
        sub = df[(df["market"] == market) & df["active"]]
        if not sub.empty:
            st, ac, n = float(sub["p"].mean()), float(sub["win"].mean()), len(sub)
            se = float(np.sqrt(max(st * (1 - st), 1e-9) / n))
            out["by_market_active_only"][market] = {
                "n": n, "mean_stated_pct": round(100 * st, 1),
                "actual_hit_pct": round(100 * ac, 1),
                "gap_pp": round(100 * (ac - st), 1),
                "overconfident": bool(ac < st - 1.96 * se)}
        sub = df[df["market"] == market]
        if sub.empty:
            continue
        stated, actual_rate = float(sub["p"].mean()), float(sub["win"].mean())
        n = len(sub)
        # Wilson-style SE using the STATED rate, not the observed one: at an
        # observed 0% the observed-rate SE collapses to ~0 and reports a
        # meaningless 26,000-sigma gap.
        se = float(np.sqrt(max(stated * (1 - stated), 1e-9) / n))
        out["by_market"][market] = {
            "n": n,
            "bar": PLAYER_PICK_MIN_PROB[market],
            "mean_stated_pct": round(100 * stated, 1),
            "actual_hit_pct": round(100 * actual_rate, 1),
            "gap_pp": round(100 * (actual_rate - stated), 1),
            "gap_in_standard_errors": round((actual_rate - stated) / se, 2) if se > 0 else None,
            "overconfident": bool(actual_rate < stated - 1.96 * se),
        }
    return out


def main():
    rep = {}
    for lg in LEAGUES:
        print(f"--- {lg} ---")
        try:
            rep[lg] = run_league(lg)
        except Exception as exc:
            rep[lg] = {"league": lg, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(rep[lg], indent=1))
    path = ROOT / "data-raw" / "leagues" / "props_pick_calibration.json"
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
