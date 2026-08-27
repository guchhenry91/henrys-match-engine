"""Orchestrator: fit -> sim -> props -> picks -> data/leagues/pl.json.

The ONLY module that knows the published JSON contract. Everything else returns
plain frames and dicts, which is what makes generalising to four leagues a loop
rather than a rewrite.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import (config, dataset, fixtures, odds, parlays, picks, players,
                     props, second_tier, sim, six_scores)
from leagues.model import (LeagueModel, promoted_priors, score_for_outcome,
                           top_scorelines, scoreline_grid, outcome_probs,
                           score_calibration)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "leagues"
PICKS_DIR = ROOT / "data-raw" / "leagues"
MATCHWEEKS_AHEAD = 1
# A pick is only FROZEN once kickoff is near. Locking early would freeze a model
# that cannot yet see late form, injuries or the closing market, and the frozen
# pick would then contradict the probabilities shown beside it. Until a fixture
# enters this window its pick is provisional and recomputed every run.
#
# 60 minutes. Held AT the confirmed starting XI, which clubs publish about an hour
# before kickoff -- the whole point is that a late team change reaches the model
# before the pick is committed.
#
# TIED TO THE PUBLISH CADENCE. A pick can only freeze on a run landing inside this
# window, so the matchday workflow runs every 15 minutes -- comfortably narrower
# than the window, giving each fixture about four chances to lock.
#
# WIDENED FROM 45 MINUTES ON 2026-08-21, and the cadence halved with it, exactly as
# the previous version of this comment prescribed. On the opening night GitHub had
# both the 18:00 and 18:30 runs still queued at 18:36; Arsenal v Coventry kicked off
# at 19:00 with its pick unfrozen and 21 minutes of the window already gone. It was
# locked by hand with 22 minutes to spare. Nobody would have noticed the void until
# the next morning -- and on a Saturday nine fixtures lock, not one.
#
# The lock window now lives in leagues.config so the fast locker can read it
# without importing this module's model stack. Re-exported here because a
# dozen call sites and several tests already reference publish.LOCK_WINDOW_HOURS.
LOCK_WINDOW_HOURS = config.LOCK_WINDOW_HOURS
# A pick joins the high-confidence board at this probability. Chosen from a pooled
# walk-forward over all four leagues, not guessed. The tier hit rates that justify
# it are no longer copied here: `leagues.tune` computes them at each of these
# thresholds and writes them to backtest_report.json's `_pooled` block, which
# `_backtested_tier_stats` reads. A number pasted into a comment cannot be wrong
# loudly -- it just quietly stops matching the model after the next refit.
# 0.65 trades a little hit rate for useful volume (roughly twice as many picks a
# matchweek as 0.70). Membership is decided from the FROZEN probability at lock
# time -- never recomputed after a result, or winners could be selected in
# hindsight.
BEST_PICK_MIN_PROB = 0.65
# A team needs at least this many players in the rates table before it gets props
# at all. The sigma-lambda rescale makes a team's players sum to the match model's
# lambda, so with a near-empty squad it hands ONE man the entire team's expected
# goals: promoted Schalke had a single player with top-flight history and came out
# at a 72.8% anytime scorer, when the next-best number in all four leagues was
# 50.8%. A team with one player is more dangerous than a team with none, because
# none is visibly a hole and one looks like a great pick. Teams below this get no
# props, exactly like teams with no data at all.
MIN_SQUAD_FOR_PROPS = 6


def _read_raw(name: str) -> dict:
    p = PICKS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _covered_sides(actuals) -> set:
    """(team, date) pairs the shot feed can actually speak about.

    A side that played always registers shots, so its presence on a date is a
    reliable signal the feed holds that half of that fixture. Absence means the
    feed is silent about it -- which is a different thing from the feed saying a
    player did nothing, and must not be graded as though it were.
    """
    if actuals is None or actuals.empty:
        return set()
    return {(r["team"], r["day"]) for _, r in actuals.iterrows()}


def _backtested_tier_stats(min_prob: float) -> dict:
    """Pooled hit rate for the board's tier, plus the per-league spread.

    One pooled number oversells its own stability: the same p>=0.65 tier has run
    from ~71% (Ligue 1) to ~86% (La Liga). The spread ships alongside it so a
    reader can see the range the headline is averaging over.

    Returns {} when the gate has not been run, so the page omits the claim
    entirely rather than showing a stale or invented one.
    """
    tiers = _read_raw("backtest_report.json").get("_pooled", {}).get("tiers", [])
    by_prob = {t["min_prob"]: t for t in tiers}
    tier, allp = by_prob.get(min_prob), by_prob.get(0.0)
    out = {}
    if tier and tier.get("hit_rate_pct") is not None:
        out["backtested_hit_rate_pct"] = tier["hit_rate_pct"]
        out["backtested_n"] = tier["n"]
        out["backtested_league_range_pct"] = [tier["league_min_pct"],
                                              tier["league_max_pct"]]
    if allp and allp.get("hit_rate_pct") is not None:
        out["backtested_all_picks_pct"] = allp["hit_rate_pct"]
    return out


def model_params(league: str) -> dict:
    """The league's xi/xg_weight, from the gate's generated release policy.

    These were hardcoded -- or rather, they were not set at all: every league ran
    `LeagueModel()` defaults while the backtest report named a different winner for
    three of the four. The gate now writes release_policy.json and this reads it,
    so the parameters that ship and the evidence for them come from one run.

    An absent or unreadable policy falls back to the model's own defaults, which
    is exactly the previous behaviour -- a missing gate artefact must not stop a
    publish, it just means nothing was promoted.
    """
    pol = _read_raw("release_policy.json").get("leagues", {}).get(league, {})
    out = {}
    if isinstance(pol.get("xi"), (int, float)):
        out["xi"] = float(pol["xi"])
    if isinstance(pol.get("xg_weight"), (int, float)):
        out["xg_weight"] = float(pol["xg_weight"])
    return out
# The bar for the cross-league player board, PER MARKET -- the markets have very
# different ceilings and a single number cannot serve all three.
#   shots (2+ shot attempts) at 0.70 is reachable by a good slice of forwards.
#   sot (1+ shot on target) is a HARDER market than 2+ shots: clearing 70% needs
#     ~1.2 expected shots on target, which basically only an elite-volume shooter
#     reaches, so a 0.70 bar published a one-name board (just Haaland) while genuine
#     candidates -- Vinicius, Thiago, Semenyo -- sat in the high 60s. That is the
#     same "bar sits at the market ceiling -> near-empty section" failure the
#     goalscorer bar was lowered to avoid, so sot is set to 0.62 to select a
#     comparable top slice instead of a single outlier.
#   goal CANNOT be 0.70: anytime scorer tops out around 50% for an elite striker in
#   a great matchup (the best in all four leagues today is 50.8%), because a team
#   only scores ~1.5 goals and one man takes a fraction of them. A 0.70 bar would
#   leave the goalscorer section permanently EMPTY, so it is set at the level that
#   selects a comparable top slice. Every card publishes its own probability, so
#   nothing here is presented as more certain than it is.
#
# RE-BASED ONTO THE CALIBRATED SCALE (2026-08-10). Published probabilities now
# carry the measured overconfidence correction (props.PROP_CALIBRATION), which
# lowers every number, so the old bars -- chosen against inflated ones -- became
# far stricter than intended: on the backtest they cut the published board from
# 144 picks to 18 and emptied the goalscorer section completely, the exact
# "bar sits at the market ceiling" failure described above.
#
# Each bar is therefore the OLD bar mapped through the SAME transform
# (new = old ** k), so the board selects EXACTLY the players it selected before
# while the probability shown against them is now the honest one:
#     goal  0.40 ** 1.437 = 0.276      shots 0.70 ** 1.954 = 0.497
#     sot   0.62 ** 1.376 = 0.511
# Selectivity is unchanged; only the labelling got truthful. A 28% anytime
# scorer reads weaker than the old 40% for the SAME player -- that is the point.
PLAYER_PICK_MIN_PROB = {"goal": 0.276, "shots": 0.497, "sot": 0.511}
PROP_FIELD = {"goal": "anytime_pct", "shots": "p_shots_2plus", "sot": "p_sot_1plus"}


def _confidence(p_pick: float) -> int:
    """1-5, matching the WC app's banding."""
    for threshold, conf in ((0.70, 5), (0.60, 4), (0.50, 3), (0.40, 2)):
        if p_pick >= threshold:
            return conf
    return 1


def _player_pick_publishable(hours_out: float, lineup_ready: bool) -> bool:
    """Whether a player pick may be published at all.

    It always may. Requiring both confirmed XIs before a pick could LOCK sounded
    like rigour and was in practice a kill switch: confirmed XIs are published
    about an hour before kickoff, the lock is 45 minutes, and news.json is filled
    in by hand -- so unless a human is at the keyboard inside that 15-minute
    window, no player pick ever enters the record. The board would show provisional
    picks that silently vanish at lock time and are never graded, and the Grades
    tab would sit permanently empty with nothing to explain why.

    A confirmed XI is a strong SIGNAL, not a precondition. It still overrides
    appearance probabilities in props.match_props, and every published pick carries
    `lineup_confirmed` so a reader (and the record) can separate the two tiers.
    """
    return True


def _why(model, home: str, away: str) -> dict | None:
    """The drivers behind a pick, in plain multipliers.

    The model computes lambda_home = exp(attack[home] + defence[away] + home_adv),
    so those three terms ARE the reasoning -- they were simply never published. A
    card showing 78% and nothing else asks to be taken on faith; showing that
    Arsenal attack +34% and Coventry defend -28% lets a reader check the pick
    against what they know about the fixture, and spot a wrong one BEFORE the
    result does.

    Expressed as percentage deviations from the league average, because the raw
    log-scale coefficients mean nothing to anyone reading a football page. Note the
    sign convention: `defence` is positive when a side concedes MORE, so it is
    negated here to read the way people expect (higher = better defence).
    """
    try:
        if home not in model.attack or away not in model.attack:
            return None
        att = list(model.attack.values())
        dfc = list(model.defence.values())
        a_mean = sum(att) / len(att)
        d_mean = sum(dfc) / len(dfc)
        pc = lambda x: round(100.0 * (float(np.exp(x)) - 1.0), 1)
        return {
            "home_attack_pct": pc(model.attack[home] - a_mean),
            "home_defence_pct": pc(-(model.defence[home] - d_mean)),
            "away_attack_pct": pc(model.attack[away] - a_mean),
            "away_defence_pct": pc(-(model.defence[away] - d_mean)),
            "home_advantage_pct": pc(model.home_adv),
        }
    except Exception:
        return None            # explanation is a nicety; never fail a publish for it


def _market_block(mkt: dict | None, pred: dict, pick_type: str) -> dict | None:
    """Attach the de-vigged market line + the model's edge on its pick.

    `edge` = model probability minus market probability on the picked outcome:
    positive means the model rates the pick higher than the bookmakers do. It is a
    disagreement measure, NOT a claim of profit. None when no line is posted
    (off-season, or a fixture not yet priced)."""
    if not mkt:
        return None
    model_p = pred[f"p_{pick_type}"] if pick_type != "draw" else pred["p_draw"]
    out = {**mkt, "edge": round(model_p - mkt[f"p_{pick_type}"], 3)}
    price = (mkt.get("odds") or {}).get(pick_type)
    if price:
        # The price on the PICKED side, and what the model's probability implies
        # at it. `ev` is expected return per 1 staked: 0.10 means the model rates
        # this a 10% edge AT THIS PRICE. It follows directly from the published
        # probability, so it is exactly as reliable as that probability and no
        # more -- and the model does not currently beat the closing line, which
        # is stated on the page. Presented as the model's own arithmetic, not
        # advice.
        out["pick_odds"] = price
        out["ev"] = round(model_p * price - 1.0, 3)
    return out


def actual_standings(played, all_teams) -> list[dict]:
    """The REAL league table from matches played SO FAR: points, played, W/D/L,
    goals for/against, goal difference. Distinct from the projected `table` (a
    season simulation) -- this is only what has actually happened. Pre-season it
    is every club on zero, ordered alphabetically; once results land it sorts by
    points, then goal difference, then goals scored, then name."""
    rows = {t: {"team": t, "played": 0, "won": 0, "drawn": 0, "lost": 0,
                "gf": 0, "ga": 0, "gd": 0, "points": 0} for t in all_teams}
    for _, m in played.iterrows():
        h, a = m["home"], m["away"]
        if h not in rows or a not in rows:
            continue
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        for t, gf, ga in ((h, hg, ag), (a, ag, hg)):
            r = rows[t]
            r["played"] += 1
            r["gf"] += gf
            r["ga"] += ga
            r["gd"] = r["gf"] - r["ga"]
        if hg > ag:
            rows[h]["won"] += 1; rows[h]["points"] += 3; rows[a]["lost"] += 1
        elif ag > hg:
            rows[a]["won"] += 1; rows[a]["points"] += 3; rows[h]["lost"] += 1
        else:
            rows[h]["drawn"] += 1; rows[a]["drawn"] += 1
            rows[h]["points"] += 1; rows[a]["points"] += 1
    return sorted(rows.values(),
                  key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"]))


def build(league: str = "PL") -> dict:
    lg = config.get(league)
    matches = dataset.build_matches(league)
    fx = fixtures.fetch_fixtures(league)
    logs = players.fetch_player_logs(league)
    ctx = players.team_shot_context(league)

    played = fx[fx["played"]].copy()
    remaining = fx[~fx["played"]].copy()

    # Fit on history plus whatever of the new season has already been played.
    if not played.empty:
        add = played[["date", "home", "away", "home_goals", "away_goals"]].copy()
        add["date"] = pd.to_datetime(add["date"]).dt.tz_localize(None)
        matches = pd.concat([matches, add], ignore_index=True)

    now = pd.Timestamp.now("UTC")           # utcnow() is deprecated in pandas 3+
    ref = now.tz_localize(None) if now.tzinfo else now
    squad_teams = sorted(set(fx["home"]) | set(fx["away"]))

    # Fit once to learn the strength scale, then refit with priors mapped onto it
    # -- promoted clubs have no top-flight history to fit on. Their prior comes from
    # their actual second-tier season (calibrated), NOT from ClubElo: ClubElo is a
    # single third-party point of failure, and the second-tier feed is the same
    # source as everything else here.
    params = model_params(league)
    base = LeagueModel(**params).fit(matches, ref=ref)
    warnings = []
    no_history = [t for t in squad_teams if t not in base.attack]
    priors = {}
    if no_history:
        try:
            priors = second_tier.second_tier_priors(base, league, no_history)
        except Exception as exc:
            print(f"WARNING: second-tier feed unavailable for {league} ({exc})")
        still_missing = [t for t in no_history if t not in priors]
        if still_missing:
            # A promoted club we could not resolve in the second-tier feed (e.g.
            # promoted from a lower level, or an unmapped spelling) -> the honest
            # weakest-side fallback, and say so.
            priors.update(promoted_priors(base, still_missing))
            names = (", ".join(still_missing[:-1]) + f" and {still_missing[-1]}"
                     if len(still_missing) > 1 else still_missing[0])
            warnings.append(
                f"No second-tier record found for {names}, so they are seeded at "
                f"the strength of the league's weakest sides rather than by their "
                f"own form. Their projected finish is a rough placeholder.")
            print(f"WARNING: {league}: no second-tier prior for {still_missing}; "
                  f"weakest-side fallback")
    model = LeagueModel(**params).fit(matches, ref=ref, priors=priors)
    # Scoreline-display corrections only (see model.score_calibration): the fitted
    # grid over-predicts 0-0 and 1-1 in this league, which is why the six-score
    # board printed 1-1 on 89% of fixtures. Applied when CHOOSING a scoreline to
    # show, never to p_home/p_draw/p_away or to anything the record grades.
    score_corr = score_calibration(league)

    # Squad freshness during an open transfer window. TWO independent things
    # decide how much a stale transfers.json actually matters, so the warning
    # states both rather than only the scarier one:
    #   * transfers.json is the MANUAL override list, aged by _verified_on;
    #   * the roster snapshot is the AUTOMATIC check, and where a club's list is
    #     complete and fresh a departed player is dropped without needing any
    #     manual entry at all (that path removed 435 departed players on the run
    #     this wording was written for).
    # Saying only "last checked N days ago" overstated the risk once the roster
    # feed became complete and current; saying only "rosters are fresh" would
    # understate it, because a player the feed spells unrecognisably is neither
    # matched nor dropped by name. Report both facts and let the reader judge.
    roster_status, roster_age = players.roster_snapshot_status(league)
    stale_days = players.transfers_age_days()
    if stale_days is not None and stale_days > 7:
        if roster_status == "ok":
            warnings.append(
                f"The manual transfer list was last re-checked {stale_days} days ago "
                f"and the window is still open. Current squads are verified "
                f"automatically against a roster feed refreshed "
                f"{round(roster_age)}h ago, which is what actually removes departed "
                f"players, so the residual risk is limited to players that feed "
                f"spells too differently to match.")
        else:
            warnings.append(
                f"Squad lists were last checked against transfer news {stale_days} days "
                f"ago and the window is still open, and the roster feed is "
                f"{roster_status}, so a player may appear at a club he has since left.")

    # Only players who actually appeared last season, with realistic minutes.
    # Otherwise five seasons of departed players share out the team's expected
    # goals and every real striker is crushed to a few percent.
    squad = players.current_squad(logs)
    league_matches = 2 * (lg.n_teams - 1)
    exp_minutes = players.expected_minutes(logs, matches_per_season=league_matches)
    playing_time = players.playing_time(logs, matches_per_season=league_matches)
    rates = props.player_rates(logs, ref=ref)
    rates = rates[rates["player"].isin(squad)]
    rates, roster_incomplete, roster_unmatched, roster_ambiguous = \
        players.reconcile_rates_to_roster(rates, league)
    # roster_status/roster_age already resolved above, where the transfer-list
    # staleness warning needs them.
    # Three DIFFERENT conditions, worded differently on purpose. Before this split
    # the page said "the roster source lists fewer than 18 players for these clubs"
    # for every one of them -- even when the real cause was that no snapshot file
    # existed at all, or the one we have had aged past the 72h limit. Those are not
    # the same problem and a reader cannot act on them if they read identically. In
    # no case are player markets withheld because of this: the squad just could not
    # be checked, and last season's attribution plus transfer overrides still show.
    if roster_status == "missing":
        warnings.append(
            "No current-roster evidence is available for this league at all, so no "
            "squad could be checked against it. Player numbers are based on last "
            "season's club plus our transfer overrides, and may include someone who "
            "has since left.")
    elif roster_status == "stale":
        age_desc = "of unknown age" if roster_age is None else f"{roster_age:.0f} hours old"
        warnings.append(
            f"The current-roster evidence is {age_desc}, past the "
            f"{players.MAX_ROSTER_AGE_HOURS:.0f}h limit, so no squad could be checked "
            f"against it this run. Player numbers are based on last season's club "
            f"plus our transfer overrides, and may include someone who has since left.")
    elif roster_incomplete:
        warnings.append(
            f"The current-roster source lists fewer than {players.MIN_COMPLETE_ROSTER} "
            f"players for {', '.join(roster_incomplete)}, so their squads specifically "
            f"could not be verified against it. Their player numbers still show, based "
            f"on last season's club plus our transfer overrides, and may include "
            f"someone who has since left.")
    if roster_unmatched:
        warnings.append(
            f"{len(roster_unmatched)} players were dropped from the player markets: "
            f"their club's current squad list is complete and they are not on it, so "
            f"they appear to have left.")
    if roster_ambiguous:
        warnings.append(
            f"{len(roster_ambiguous)} player identities could not be resolved because "
            f"the same name appears at two different clubs in the roster source: "
            f"{'; '.join(roster_ambiguous)}. Neither club's number for that name is "
            f"trusted, rather than guessing which one is current.")
    takers = players.penalty_takers(logs[logs["player"].isin(squad)])
    news = players.load_news(league)   # injuries/suspensions, Best Picks fixtures
    # Shot events are the ONLY per-match player feed, so without them a league can
    # neither offer a real shots-on-target number (the ratio would be a league
    # average, i.e. an assumption dressed as a measurement) nor grade any player
    # pick afterwards. Both consequences are surfaced rather than hidden.
    shots_ok = players.shot_events_available(league)
    # TWO QUESTIONS, NOT ONE. shots_ok asks whether the RATES behind a prop can be
    # measured; can_grade asks whether a pick can be SETTLED afterwards. They had
    # the same answer only while Understat was the sole per-match player feed, and
    # collapsing them cost Bundesliga its whole player record -- every pick there
    # published gradeable=false and was excluded from the record and from every
    # parlay, because its shot events crash upstream. API-Football settles those
    # fixtures perfectly well; it just cannot supply the seasons of history the
    # rates need. So Bundesliga's goal and shots picks now grade, while its
    # shots-on-target market stays withheld.
    can_grade = players.grading_feed_available(league)
    if not shots_ok:
        warnings.append(
            f"{lg.name} has no shot-level feed, so the shots-on-target market is "
            f"withheld entirely.")
    if not can_grade:
        # Losing the SOT market is a narrowing; losing gradeability is a pick that
        # can never settle. Different severity, so it gets said separately rather
        # than folded into the sentence above.
        warnings.append(
            f"{lg.name} player picks cannot be graded against actual match lines "
            f"-- neither the shot feed nor the fallback can settle them.")
    # Last five matches per player, from the SAME merged rows the record is
    # graded on. Built once per league rather than per pick: it is one pass over
    # the shot feed and would otherwise be repeated for every published name.
    try:
        form5 = players.recent_form(league)
    except Exception as exc:
        print(f"WARNING: {league} form strips unavailable ({exc})")
        form5 = {}

    concede = ctx["concede_factor"]
    pens_rate = ctx["pens_per_team_match"]

    table = sim.simulate_season(model, played, remaining, league)

    log_path = PICKS_DIR / lg.key.lower() / "picks_log.json"
    log = picks.load_log(log_path)
    # fixturedownload MatchNumbers reset to 1..N every season, but picks_log
    # persists across seasons, so namespace each entry by the season to stop next
    # season's fixture #1 inheriting (and being graded against) this season's pick.
    season_tag = lg.fixture_slug.rsplit("-", 1)[-1]
    log_key = lambda mid: f"{season_tag}:{mid}"

    # Player picks get their OWN frozen log, graded separately from the match picks.
    pl_log_path = PICKS_DIR / lg.key.lower() / "player_picks_log.json"
    pl_log = picks.load_log(pl_log_path)

    # Live bookmaker lines for upcoming fixtures (empty off-season -> every
    # match gets market: None; the card renders fine either way).
    market_odds = odds.fetch_fixture_odds(league)

    if remaining.empty:
        upcoming = remaining
    else:
        # The current matchweek is the round of the SOONEST unplayed fixture, not
        # the lowest round number: a single postponed early-round game would
        # otherwise make min() return that stale round and hide the imminent week.
        next_round = int(remaining.sort_values("date").iloc[0]["round"])
        upcoming = remaining[remaining["round"] < next_round + MATCHWEEKS_AHEAD]

    # A squad too thin to share out the team's goals sensibly is dropped entirely
    # (see MIN_SQUAD_FOR_PROPS) rather than allowed to concentrate the whole team
    # lambda on one or two names.
    thin_squads = props.thin_squads(rates, squad_teams, MIN_SQUAD_FOR_PROPS)
    if thin_squads:
        rates = rates[~rates["team"].isin(thin_squads)]
    missing_squads = sorted({t for t in squad_teams
                             if rates[rates["team"] == t].empty})

    out_matches = []
    suspect_times = []          # fixtures whose kickoff the feed got wrong
    released_locks = []         # picks freed because their kickoff moved
    for _, m in upcoming.iterrows():
        home, away = m["home"], m["away"]
        pred = model.predict(home, away)

        # TEAM NEWS REACHES THE MATCH MODEL. Until now a confirmed absence moved
        # only the player props: the model would still rate a side at full strength
        # with its main striker ruled out, which is precisely when the opponent
        # gains an edge. The penalty is measured (see props.ABSENCE_GOAL_COST), and
        # applied deliberately conservatively. Doubtful players are now weighted
        # too (props.DOUBTFUL_ABSENCE_WEIGHT) instead of being silently dropped --
        # the props board already gave them a ~50% chance of featuring, so the
        # match model was inconsistent with itself for the exact same fact.
        news_out, news_doubt = players.news_unavailable(news, (home, away))
        pen_h = props.absence_penalty(rates, home, news_out, news_doubt)
        pen_a = props.absence_penalty(rates, away, news_out, news_doubt)
        if pen_h or pen_a:
            # Floor at a quarter of a goal: a team missing its whole attack is
            # weakened, not incapable, and letting lambda collapse would produce
            # absurd scorelines. This also bounds how far ANY combination of
            # simultaneous confirmed absences can push lambda down.
            lh = max(pred["lambda_home"] - pen_h, 0.25)
            la = max(pred["lambda_away"] - pen_a, 0.25)
            grid = scoreline_grid(lh, la, model.rho)
            ph, pdw, pa = outcome_probs(grid)
            pred = {**pred, "p_home": ph, "p_draw": pdw, "p_away": pa,
                    "lambda_home": lh, "lambda_away": la, "grid": grid,
                    "absence_penalty": {"home": round(pen_h, 3),
                                        "away": round(pen_a, 3)}}
        # Confirmed-out defenders/keepers are invisible to absence_penalty's
        # shot-share proxy (see UNMODELED_ABSENCE_POSITIONS) -- surface them
        # rather than let a defensive absence look accounted for when it isn't.
        unmodeled_h = props.unmodeled_absentee_positions(rates, home, news_out)
        unmodeled_a = props.unmodeled_absentee_positions(rates, away, news_out)
        if unmodeled_h or unmodeled_a:
            pred = {**pred, "unmodeled_absences": {"home": unmodeled_h,
                                                    "away": unmodeled_a}}
        probs = {home: pred["p_home"], "Draw": pred["p_draw"], away: pred["p_away"]}
        pick = max(probs, key=probs.get)

        # Freeze only inside the lock window; before that the pick stays live.
        #
        # A fixture whose kickoff time we do not believe NEVER freezes. The lock
        # window is measured from the kickoff, so a wrong time freezes the pick at
        # the wrong moment -- on 2026-08-15 the feed had La Liga 10 hours early and
        # both openers locked before dawn, hours before any confirmed XI existed,
        # which is precisely what the 45-minute window is designed to prevent.
        # Staying provisional is the safe failure: the pick keeps updating, and it
        # will freeze correctly once the time is verified into fixture_times.json.
        hours_out = (pd.Timestamp(m["date"]) - now).total_seconds() / 3600.0
        if bool(m.get("time_suspect")):
            suspect_times.append(f"{m['home']} v {m['away']}")
        # If this fixture's kickoff has MOVED since its pick was locked -- a
        # corrected feed time or a postponement -- that lock was made against a
        # different fixture-time and is released so it can be re-made properly.
        # Only ever while the new kickoff is still in the future (see the
        # docstring); a played match is never re-locked.
        if picks.release_moved_lock(log, log_key(m["match_id"]), m["date"], now=now):
            released_locks.append(f"{m['home']} v {m['away']}")
        if hours_out <= LOCK_WINDOW_HOURS and not bool(m.get("time_suspect")):
            entry = picks.lock_pick(log, log_key(m["match_id"]), pick=pick,
                                    confidence=_confidence(probs[pick]),
                                    kickoff=m["date"], now=now,
                                    p_pick=probs[pick],
                                    board=probs[pick] >= BEST_PICK_MIN_PROB)
            provisional = False
        else:
            entry = {"pick": pick, "confidence": _confidence(probs[pick]),
                     "p_pick": round(float(probs[pick]), 4)}
            provisional = True
        # Everything the card shows must describe the FROZEN pick, not the fresh
        # argmax: on a re-run after the model flips, entry["pick"] is still the
        # locked side, so pick_type and the market edge below must be derived from
        # it -- otherwise the card shows one team but grades/edges another.
        frozen = entry["pick"]
        pick_type = ("home" if frozen == home
                     else "away" if frozen == away else "draw")
        # The model's committed single call. It is the most likely score GIVEN the
        # pick, so the card never contradicts itself -- the unconditional mode is
        # 1-1 in 68% of fixtures and would fight a home/away pick.
        score = score_for_outcome(pred["grid"], pick_type, corr=score_corr)
        # ...and the honest spread behind it. A single score is right ~12% of the
        # time; these three cover ~31%, and their probabilities show how thin the
        # call really is. `agrees_with_pick` marks which of them match the pick.
        spread = top_scorelines(pred["grid"], n=3, corr=score_corr)
        for s in spread:
            h, a = (int(x) for x in s["score"].split("-"))
            s["outcome"] = "home" if h > a else "away" if a > h else "draw"
            s["agrees_with_pick"] = (s["outcome"] == pick_type)

        # a player's shooting opportunity scales with how many shots his OPPONENT
        # concedes relative to the league average
        opp_factor = {home: concede.get(away, 1.0), away: concede.get(home, 1.0)}
        unavailable, doubtful = players.news_unavailable(news, (home, away))
        confirmed_starters, confirmed_bench = players.lineup_players(
            news, (home, away))
        lineup_ready = players.lineups_confirmed(news, (home, away))
        squad_props = props.match_props(
            rates, home, away, pred["lambda_home"], pred["lambda_away"],
            minutes=exp_minutes, pen_taker=takers, opp_shot_factor=opp_factor,
            exp_pens={home: pens_rate, away: pens_rate},
            unavailable=unavailable, doubtful=doubtful,
            playing_time=playing_time,
            confirmed_starters=confirmed_starters,
            confirmed_bench=confirmed_bench)

        # Player picks clearing their market's bar, frozen on the SAME schedule as
        # the match pick so both boards are graded under one discipline. A doubtful
        # player is deliberately still eligible: his halved expected minutes have
        # already pushed his probability down, so if he still clears the bar the
        # model is saying the pick survives the doubt.
        player_picks = []
        for market, field in PROP_FIELD.items():
            if market == "sot" and not shots_ok:
                continue          # synthetic ratio, not a measurement -- see above
            bar = PLAYER_PICK_MIN_PROB[market] * 100.0
            for p in squad_props:
                if p[field] < bar:
                    continue
                # A confirmed XI is a strong SIGNAL, not a precondition -- see
                # _player_pick_publishable. It always returns True; requiring both
                # confirmed XIs before a pick could lock made the record
                # permanently empty whenever nobody updated news.json in time.
                if not _player_pick_publishable(hours_out, lineup_ready):
                    continue
                pkey = f"{log_key(m['match_id'])}:{market}:{p['player']}"
                prob = p[field] / 100.0
                if hours_out <= LOCK_WINDOW_HOURS:
                    pe = picks.lock_prop(pl_log, pkey, market=market,
                                         player=p["player"], team=p["team"],
                                         p_pick=prob, confidence=_confidence(prob),
                                         kickoff=m["date"], now=now,
                                         bar=PLAYER_PICK_MIN_PROB[market],
                                         lineup_confirmed=lineup_ready,
                                         appearance_pct=p.get("appearance_pct"),
                                         expected_minutes=p.get("expected_minutes"),
                                         news_checked_hours_ago=players.news_checked_age_hours(
                                             news, (home, away)),
                                         doubt=p.get("doubt", False),
                                         unavailable=p["player"] in unavailable,
                                         team_attribution=p["team"])
                    pprov = False
                else:
                    pe = {"p_pick": round(prob, 4), "confidence": _confidence(prob)}
                    pprov = True
                player_picks.append({
                    "market": market,
                    "line": picks.PROP_MARKETS[market][2],
                    "player": p["player"],
                    "team": p["team"],
                    "position": p["position"],
                    "p_pick": pe["p_pick"],
                    "confidence": pe["confidence"],
                    "provisional": pprov,
                    "doubt": p.get("doubt", False),
                    "penalty_taker": p.get("penalty_taker", False),
                    "appearance_pct": p.get("appearance_pct"),
                    "expected_minutes": p.get("expected_minutes"),
                    # Expected COUNTS behind each market, so the card can show
                    # "0.6 on target" not just a probability: anytime goal chance,
                    # expected shot attempts, expected shots on target.
                    "anytime_pct": p.get("anytime_pct"),
                    "exp_shots": p.get("exp_shots"),
                    "exp_sot": p.get("exp_sot"),
                    "lineup_confirmed": lineup_ready,
                    "gradeable": can_grade,
                    # Audit fields, published so a lock made by
                    # scripts/lock_picks.py is identical to one made here rather
                    # than a thinner version of it. p_pick alone answers "was he
                    # graded right", never "was the prediction reasonable given
                    # what we knew" -- and that is the question a settled pick
                    # actually needs to answer later.
                    # THE LAST FIVE, the same treatment the NFL props get: the
                    # individual matches rather than an average, because
                    # "0, 2, 0, 1, 3 shots" and "1.2 average" describe very
                    # different players. Keyed to the stat THIS market settles on.
                    "last_five": (form5.get(p["player"], {})
                                  .get(picks.PROP_MARKETS[market][0], [])),
                    "last_five_line": picks.PROP_MARKETS[market][1],
                    "bar": PLAYER_PICK_MIN_PROB[market],
                    "news_checked_hours_ago": players.news_checked_age_hours(
                        news, (home, away)),
                    "unavailable": p["player"] in unavailable,
                    "team_attribution": p["team"],
                })
        player_picks.sort(key=lambda x: -x["p_pick"])

        out_matches.append({
            "id": int(m["match_id"]),
            "matchweek": int(m["round"]),
            "date": pd.Timestamp(m["date"]).isoformat(),
            "venue": m["venue"],
            "home": home,
            "away": away,
            # Published so scripts/lock_picks.py can honour the same guard this
            # module does. A locker that cannot see a suspect kickoff would freeze
            # a pick against a time the fixture does not have -- the exact failure
            # release_moved_lock exists to undo.
            "time_suspect": bool(m.get("time_suspect")),
            "prediction": {
                "p_home": round(pred["p_home"], 3),
                "p_draw": round(pred["p_draw"], 3),
                "p_away": round(pred["p_away"], 3),
                "pick": entry["pick"],           # the FROZEN pick, never a fresh one
                "pick_type": pick_type,
                "score": score,
                "top_scores": spread,
                "confidence": entry["confidence"],
                "provisional": provisional,   # True = not yet frozen; will be re-picked
                "p_pick": entry.get("p_pick"),
                "why": _why(model, home, away),
                "best_pick": bool((entry.get("p_pick") or 0) >= BEST_PICK_MIN_PROB),
                "reasons": [
                    f"Model: {home} {pred['p_home']:.0%} / draw {pred['p_draw']:.0%} "
                    f"/ {away} {pred['p_away']:.0%}",
                    f"Expected goals: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}",
                ],
                # Confirmed-out defenders/keepers absence_penalty cannot price in
                # (see props.UNMODELED_ABSENCE_POSITIONS) -- empty unless relevant,
                # so a reader/consumer can tell a genuine limitation from silence.
                "unmodeled_absences": pred.get("unmodeled_absences",
                                               {"home": [], "away": []}),
            },
            "props": (props.top_props(squad_props, home)
                      + props.top_props(squad_props, away)),
            "player_picks": player_picks,
            "market": _market_block(odds.market_for(market_odds, home, away),
                                    pred, pick_type),
            "result": None,
            "graded": None,
            "void": False,
        })

    # Grade every played fixture we had locked a pick for, against the FROZEN pick.
    graded = []
    for _, m in played.iterrows():
        k = log_key(m["match_id"])
        entry = log.get(k)
        if not entry:
            continue
        g = picks.grade(entry, {"home": m["home"], "away": m["away"],
                                "home_goals": m["home_goals"],
                                "away_goals": m["away_goals"]})
        log[k].update({"graded": g["graded"], "void": g["void"]})
        graded.append(log[k])

    picks.save_log(log, log_path)
    picks.save_log(pl_log, pl_log_path)

    # Whole-season fixture list: every match, played or not, with its frozen pick
    # and grade. Deliberately WITHOUT props — 380 fixtures of scorer data would
    # bloat the payload; the props live on the current matchweek's cards only.
    season = []
    for _, m in fx.sort_values(["round", "date"]).iterrows():
        entry = log.get(log_key(m["match_id"])) or {}
        played_row = bool(m["played"])
        season.append({
            "id": int(m["match_id"]),
            "matchweek": int(m["round"]),
            "date": pd.Timestamp(m["date"]).isoformat(),
            "home": m["home"],
            "away": m["away"],
            "result": ({"home_goals": int(m["home_goals"]),
                        "away_goals": int(m["away_goals"])} if played_row else None),
            "pick": entry.get("pick"),
            "graded": entry.get("graded"),
            "void": bool(entry.get("void", False)),
        })

    if released_locks:
        print(f"NOTE: {league}: kickoff moved since lock; pick released to be "
              f"re-made at the real time: {released_locks}")
    if suspect_times:
        names = ", ".join(suspect_times[:4]) + (
            f" and {len(suspect_times) - 4} more" if len(suspect_times) > 4 else "")
        warnings.append(
            f"The fixture feed gives an implausible kickoff time for {names}, so the "
            f"time shown may be wrong and those picks stay provisional rather than "
            f"freezing at the wrong moment. Verified times go in fixture_times.json.")
        print(f"WARNING: {league}: implausible kickoff time, pick NOT locked, for "
              f"{len(suspect_times)} fixture(s): {suspect_times[:6]}")

    def _read(path):
        p = PICKS_DIR / path
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # Props gate is stored per league (PL in props_report.json, others suffixed);
    # reading the bare PL file for every league published PL's props numbers on
    # La Liga / Bundesliga / Ligue 1 pages.
    props_file = ("props_report.json" if league == "PL"
                  else f"props_report_{league.lower()}.json")

    return {
        "league": lg.name,
        "updated": datetime.now(timezone.utc).isoformat(),
        "record": picks.record(graded),
        "matches": out_matches,
        "season": season,
        "table": table.to_dict(orient="records"),
        # The ACTUAL table from results so far (see actual_standings). The Tables
        # tab shows this; `table` above stays the projected finish.
        "standings": actual_standings(played, [r["team"] for r in table.to_dict(orient="records")]),
        "backtest": _read("backtest_report.json").get(league, {}),
        "props_backtest": _read(props_file),
        "missing_squads": missing_squads,
        "thin_squads": thin_squads,
        "roster_incomplete": roster_incomplete,
        "roster_snapshot_age_hours": (
            None if roster_age is None else round(roster_age, 1)),
        "roster_unmatched_count": len(roster_unmatched),
        "data_warnings": warnings,
    }


FILE_FOR = {"PL": "pl.json", "LALIGA": "laliga.json",
            "BUNDESLIGA": "bundesliga.json", "LIGUE1": "ligue1.json"}


def _publish_one(league: str, fname: str) -> bool:
    """Build and atomically write one league. Returns True on success."""
    try:
        payload = build(league)
    except Exception as exc:              # one league's outage must not sink the rest
        print(f"ABORT {league}: {exc}; leaving its file untouched")
        return False
    path = OUT / fname
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)                     # atomic: never publish a half-written file
    print(f"wrote {path} - {len(payload['matches'])} fixtures, "
          f"{len(payload['table'])} teams")
    if payload.get("thin_squads"):
        print(f"  WARNING {league}: squad too thin for props "
              f"{payload['thin_squads']} (<{MIN_SQUAD_FOR_PROPS} players with "
              f"top-flight history) - no props, to stop one man absorbing the "
              f"whole team lambda")
    if payload.get("missing_squads"):
        # Do NOT assert a cause here. This list is now fed by two very different
        # situations -- a promoted club with no top-flight history, and a club whose
        # players were all dropped by roster reconciliation. Blaming promotion
        # printed "promoted clubs have no top-flight history" against Real Madrid,
        # Barcelona and PSG, which sent anyone reading the logs after the wrong bug.
        print(f"  WARNING {league}: no player data for {payload['missing_squads']} "
              f"- they get no props (newly promoted, or no roster match)")
    return True


def build_best_picks() -> dict:
    """The high-confidence board: every league's strongest picks in one place.

    Assembled AFTER the leagues publish, by reading each league's frozen picks_log
    and its fixture results. Membership is decided by the probability recorded at
    LOCK time (>= BEST_PICK_MIN_PROB), so a pick cannot be promoted onto the board
    after it wins -- the same freezing discipline as the main record.

    Graded SEPARATELY from the all-picks record, so this tier can be judged on its
    own and shown to be earning its billing (or not).
    """
    upcoming, settled, incomplete = [], [], []
    seen_upcoming = set()          # (league_key, match id) already on the board
    for league, fname in FILE_FOR.items():
        lg = config.get(league)
        season_tag = lg.fixture_slug.rsplit("-", 1)[-1]
        log = picks.load_log(PICKS_DIR / lg.key.lower() / "picks_log.json")

        # UPCOMING comes from the freshly published payload, not the picks_log:
        # outside the 48h lock window a pick is deliberately provisional and has no
        # log entry yet, so reading only frozen picks would leave the board empty
        # all week. Those entries are marked provisional so the page can say the
        # pick may still move.
        try:
            payload = json.loads((OUT / fname).read_text(encoding="utf-8"))
        except Exception:
            payload = {"matches": []}
        for m in payload.get("matches", []):
            p = m.get("prediction", {})
            if not p.get("best_pick"):
                continue
            seen_upcoming.add((league, int(m["id"])))
            upcoming.append({
                "league": lg.name, "league_key": league,
                "id": m["id"], "matchweek": m["matchweek"], "date": m["date"],
                "home": m["home"], "away": m["away"],
                "pick": p["pick"], "confidence": p.get("confidence"),
                "p_pick": p.get("p_pick"), "score": p.get("score"),
                "provisional": bool(p.get("provisional")),
                # The price on the picked side, carried onto the board so a
                # reader does not have to open each match to see it. Absent
                # (None) whenever the fixture is not priced yet -- the odds feed
                # only reaches about a week ahead.
                "odds": (m.get("market") or {}).get("pick_odds"),
                "book": (m.get("market") or {}).get("book"),
            })

        # SETTLED comes only from the frozen log -- graded honestly.
        #
        # A failure here must NOT be swallowed. `settled` and `record` are rebuilt
        # from scratch every run, so skipping a league silently deletes its entire
        # graded history from the published record -- and the deletion always
        # removes losses as readily as wins, i.e. it flatters the model on a five
        # second timeout. The board is refused entirely instead (see `incomplete`),
        # leaving the last good file in place.
        try:
            fx = fixtures.fetch_fixtures(league)
        except Exception as exc:
            print(f"  best-picks: CANNOT grade {league} ({exc}) -- refusing to "
                  f"publish a board that would drop its record")
            incomplete.append(lg.name)
            continue
        by_id = {int(r["match_id"]): r for _, r in fx.iterrows()}

        for key, entry in log.items():
            if not str(key).startswith(f"{season_tag}:"):
                continue                      # a previous season's entry
            # Membership was FROZEN at lock time. Only fall back to comparing
            # against the live constant for legacy entries written before `board`
            # existed -- never for new ones, or raising the bar would retroactively
            # delete settled picks from the record.
            on_board = entry.get("board")
            if on_board is None:
                on_board = (entry.get("p_pick") or 0) >= BEST_PICK_MIN_PROB
            if not on_board:
                continue
            mid = int(str(key).split(":", 1)[1])
            row = by_id.get(mid)
            if row is None:
                continue
            item = {
                "league": lg.name, "league_key": league,
                "id": mid, "matchweek": int(row["round"]),
                "date": pd.Timestamp(row["date"]).isoformat(),
                "home": row["home"], "away": row["away"],
                "pick": entry["pick"], "confidence": entry.get("confidence"),
                "p_pick": entry.get("p_pick"),
                # Tier membership as FROZEN at lock time. sanity_check requires it
                # on every settled entry, because without it a settled pick would
                # fall back to comparing against the LIVE bar -- so raising the bar
                # would retroactively evict past picks from the record. It was
                # never written here, and nothing caught that until Ath Madrid v
                # Malaga became the first Best Pick ever to settle: every earlier
                # graded pick sat below 0.65 and so never reached this list.
                "board": entry.get("board"),
            }
            if bool(row["played"]):
                g = picks.grade(entry, {"home": row["home"], "away": row["away"],
                                        "home_goals": row["home_goals"],
                                        "away_goals": row["away_goals"]})
                item["result"] = {"home_goals": int(row["home_goals"]),
                                  "away_goals": int(row["away_goals"])}
                item["graded"] = g["graded"]
                item["void"] = g["void"]
                settled.append(item)
            elif (league, mid) not in seen_upcoming:
                # Inside the lock window BOTH sources describe the same fixture --
                # the payload copy (with scoreline and provisional flag) and this
                # one. Publishing both put a duplicate card, blank-scored, on the
                # board for every locked pick on matchday.
                upcoming.append(item)

    upcoming.sort(key=lambda x: (-(x["p_pick"] or 0), x["date"]))
    settled.sort(key=lambda x: x["date"], reverse=True)
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "_incomplete": incomplete,     # non-empty -> caller must NOT publish
        "min_probability": BEST_PICK_MIN_PROB,
        "record": picks.record(settled),
        "upcoming": upcoming,
        "settled": settled[:60],
        # Backtested expectation for this tier, so the page can state what the
        # board is worth rather than implying certainty. READ from the gate's
        # pooled walk-forward, not pasted: these were the literals 77.4 and 53.2,
        # which verified when written and had nothing keeping them true across a
        # refit.
        **_backtested_tier_stats(BEST_PICK_MIN_PROB),
    }


def append_history(best: dict, players: dict) -> list:
    """One row per publish: the record so far, so drift becomes visible.

    The Grades tab shows where the record STANDS. It cannot show which way it is
    moving, and the warning sign that matters is not a bad weekend -- a 65% pick
    loses one time in three, so three straight losses is normal -- but stated
    confidence drifting away from observed results over dozens of picks. Without a
    time series that is unanswerable, which makes "is it still working?" a matter
    of feel. This makes it a matter of record.

    Append-only, one row per day: re-running on the same day replaces that day's
    row rather than inflating the series.
    """
    path = PICKS_DIR / "record_history.json"
    try:
        hist = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        hist = []

    today = datetime.now(timezone.utc).date().isoformat()
    br, pr = best.get("record", {}), players.get("record", {})
    # Stated vs observed on the SETTLED best picks: the calibration question.
    settled = best.get("settled", [])
    graded = [s for s in settled if s.get("graded") in ("correct", "wrong")]
    stated = (sum(s.get("p_pick") or 0 for s in graded) / len(graded)) if graded else None
    actual = (sum(s.get("graded") == "correct" for s in graded) / len(graded)) if graded else None

    row = {
        "date": today,
        "best": {"correct": br.get("correct", 0), "wrong": br.get("wrong", 0),
                 "total": br.get("total", 0)},
        "players": {"correct": pr.get("correct", 0), "wrong": pr.get("wrong", 0),
                    "total": pr.get("total", 0)},
        "stated_pct": None if stated is None else round(100 * stated, 1),
        "actual_pct": None if actual is None else round(100 * actual, 1),
    }
    hist = [h for h in hist if h.get("date") != today] + [row]
    hist.sort(key=lambda h: h["date"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    tmp.replace(path)
    return hist


def build_player_picks() -> dict:
    """The cross-league player board: goalscorer, shot attempts, shots on target.

    Same discipline as build_best_picks -- upcoming is read from the freshly
    published payloads (so the board is populated outside the lock window, flagged
    provisional), settled comes ONLY from the frozen log and is graded against the
    player's actual match line.

    Graded per market as well as overall, because the three markets are not
    comparable: a 45% goalscorer pick and a 78% shots pick are both "high
    confidence" for their market, and pooling them would produce a headline number
    that describes neither.
    """
    upcoming, settled, ungradeable, incomplete = [], [], [], []
    # Frozen picks on PLAYED fixtures the shot feed does not cover yet.
    # Not settled (no evidence) and not upcoming (already kicked off), so
    # they need their own bucket -- otherwise they vanish from the page
    # entirely and a reader cannot tell a pick is still owed a result.
    uncovered = []
    for league, fname in FILE_FOR.items():
        lg = config.get(league)
        season_tag = lg.fixture_slug.rsplit("-", 1)[-1]
        log = picks.load_log(PICKS_DIR / lg.key.lower() / "player_picks_log.json")

        try:
            payload = json.loads((OUT / fname).read_text(encoding="utf-8"))
        except Exception:
            payload = {"matches": []}
        for m in payload.get("matches", []):
            for pp in m.get("player_picks", []) or []:
                upcoming.append({**pp, "league": lg.name, "league_key": league,
                                 "id": m["id"], "date": m["date"],
                                 "home": m["home"], "away": m["away"]})

        if not log:
            continue
        try:
            fx = fixtures.fetch_fixtures(league)
        except Exception as exc:
            print(f"  player-picks: CANNOT grade {league} ({exc}) -- refusing to "
                  f"publish a board that would drop its record")
            incomplete.append(lg.name)
            continue
        by_id = {int(r["match_id"]): r for _, r in fx.iterrows()}

        # Actual per-match player lines. Empty when shot events are unreadable
        # (upstream Bundesliga crash) -- then every pick in that league stays
        # PENDING rather than being graded wrong against missing data.
        primary_failed = False
        try:
            actuals = players.match_player_stats(league)
        except Exception as exc:
            print(f"  player-picks: no actuals for {league} ({exc})")
            actuals = pd.DataFrame()
            primary_failed = True
        # Second source, for fixtures Understat has not filed. Understat stays
        # authoritative wherever it HAS the match (it is shot-event derived and
        # identifies penalties); this only fills silence.
        try:
            api_actuals, api_covered = players.api_match_stats(league)
        except Exception as exc:
            print(f"  player-picks: fallback stats unreadable for {league} ({exc})")
            api_actuals, api_covered = pd.DataFrame(), set()
        have_actuals = not actuals.empty or not api_actuals.empty
        if primary_failed and have_actuals:
            # The fallback must never paper over an OUTAGE of the primary feed.
            # It covers only the fixtures it was asked to fetch, so continuing on
            # it alone would quietly move every Understat-covered pick into
            # awaiting_data -- shrinking the published record without saying why,
            # which is the same partial-record dishonesty the check below exists
            # to prevent. A feed that normally works and did not answer this run
            # is a reason to publish nothing, not a reason to publish less.
            print(f"  player-picks: {league} primary feed failed; refusing to "
                  f"publish a record backed only by the fallback")
            incomplete.append(lg.name)
            continue
        if not have_actuals:
            # Distinguish a PERMANENT missing feed from a transient one. Bundesliga
            # genuinely has no shot events (upstream crash) and its picks can never
            # be graded -- that is worth stating on the page. A one-off timeout on a
            # league that normally grades fine is a different thing entirely, and
            # treating it as permanent would silently delete that league's record
            # AND print a flatly false claim that it has no shot feed.
            try:
                # Permanent means NOTHING can ever settle a pick here -- so it
                # must ask about both feeds, not just the shot events. Asking
                # only Understat would brand Bundesliga permanently ungradeable
                # on a run where the fallback simply had nothing to fetch yet.
                permanent = not players.grading_feed_available(league)
            except Exception:
                permanent = False
            if permanent:
                ungradeable.append(lg.name)
            else:
                print(f"  player-picks: {league} actuals unavailable but its feed "
                      f"normally works -- refusing to publish a partial record")
                incomplete.append(lg.name)
            continue
        if have_actuals:
            # WHICH FIXTURES THE FEEDS ACTUALLY COVER.
            #
            # grade_prop treats a missing player row as WRONG on purpose: he
            # either did not play or played and never shot, and a shot feed
            # cannot tell those apart, so we take the harsher reading. That
            # reasoning holds only where the feed HAS the match. Where it does
            # not, "wrong" is not a harsh reading of the evidence -- it is an
            # answer invented in the absence of any.
            #
            # This is not hypothetical. Understat published NOTHING for 2026-27
            # while 26 fixtures were played: its newest row was 2026-05-24, the
            # previous May. The frame was far from empty (26,401 rows for the PL
            # alone) so the have_actuals guard above passed happily, every lookup
            # missed, and the board published 0 correct / 18 wrong -- including
            # 0-for-4 on picks stated at 60-70%, odds near one in ten million had
            # those picks been graded against real data.
            #
            # A side that played always registers shots, so its presence on a
            # date is a reliable signal that a feed covers that half of that
            # fixture. Keyed by the PLAYER'S OWN TEAM rather than the fixture, so
            # a pick is only graded against a side some feed can speak about.
            idx = {}
            covered = set(api_covered)
            if not api_actuals.empty:
                api_actuals = api_actuals.assign(
                    day=pd.to_datetime(api_actuals["date"]).dt.date)
                idx = {(r["player"], r["day"]): r
                       for _, r in api_actuals.iterrows()}
            if not actuals.empty:
                actuals = actuals.assign(
                    day=pd.to_datetime(actuals["date"]).dt.date)
                # Understat last, so it OVERWRITES the fallback on any fixture
                # both feeds hold.
                idx.update({(r["player"], r["day"]): r
                            for _, r in actuals.iterrows()})
                covered |= _covered_sides(actuals)
        for key, entry in log.items():
            parts = str(key).split(":")
            if len(parts) < 4 or parts[0] != season_tag:
                continue
            mid = int(parts[1])
            row = by_id.get(mid)
            if row is None:
                continue
            item = {**entry, "league": lg.name, "league_key": league, "id": mid,
                    "line": picks.PROP_MARKETS[entry["market"]][2],
                    "date": pd.Timestamp(row["date"]).isoformat(),
                    "home": row["home"], "away": row["away"]}
            if not bool(row["played"]):
                continue                  # already covered by the payload read above
            day = pd.Timestamp(row["date"]).date()
            if (entry.get("team"), day) not in covered:
                # The feed has no line for this player's side in this fixture, so
                # there is no evidence either way. Stays PENDING: it will grade
                # itself the moment the data lands, and an ungraded pick is an
                # honest gap where a fabricated loss is a lie in an append-only
                # record.
                uncovered.append(item)
                continue
            actual = idx.get((entry["player"], day))
            settled.append(picks.grade_prop(entry, None if actual is None
                                            else dict(actual)) | item)

    upcoming.sort(key=lambda x: (-(x["p_pick"] or 0), x["date"]))
    settled.sort(key=lambda x: x["date"], reverse=True)
    by_market = {mk: picks.record([s for s in settled if s.get("market") == mk])
                 for mk in picks.PROP_MARKETS}
    # The split that makes the headline number interpretable. A pick frozen without
    # knowing the XI is a different product from one frozen with it, and pooling
    # them means a poor hit rate cannot be attributed to either the model or the
    # missing team news.
    by_lineup = {
        "confirmed": picks.record([s for s in settled
                                   if s.get("lineup_confirmed") is True]),
        "unconfirmed": picks.record([s for s in settled
                                     if s.get("lineup_confirmed") is not True]),
    }
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "min_probability": PLAYER_PICK_MIN_PROB,
        "markets": {k: v[2] for k, v in picks.PROP_MARKETS.items()},
        "record": picks.record(settled),
        "record_by_market": by_market,
        "record_by_lineup": by_lineup,
        # Union of two sources, because the settled path alone is too late.
        # `ungradeable` is only appended when a league FAILS TO GRADE a settled
        # pick, so on 2026-08-21 six Bundesliga picks sat on the board flagged
        # gradeable=False -- Kane's 0.660 shots among them, kicking off that
        # evening -- while this list was empty and the page's "not graded" note
        # never rendered. The warning arrived only after a pick had already
        # silently failed to reach the record. An UPCOMING pick that cannot be
        # graded is exactly as worth warning about as a settled one.
        # Played, frozen, and still waiting on the shot feed. Surfaced so the
        # page can say so out loud: silence here is what let 18 fabricated
        # losses look like a real 0% hit rate.
        "awaiting_data": sorted(
            ({"player": u.get("player"), "team": u.get("team"),
              "market": u.get("market"), "league": u.get("league"),
              "date": u.get("date"), "fixture": f"{u.get('home')} v {u.get('away')}"}
             for u in uncovered),
            key=lambda u: (u["date"] or "", u["player"] or "")),
        "ungradeable_leagues": sorted(set(ungradeable) | {
            u["league"] for u in upcoming if u.get("gradeable") is False}),
        "_incomplete": incomplete,     # non-empty -> caller must NOT publish
        "upcoming": upcoming,
        "settled": settled[:120],
    }


def main(argv=None):
    """Publish all four leagues, or just the ones named on the command line
    (e.g. `python -m leagues.publish PL` for quick iteration)."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    leagues = [a.upper() for a in argv] or list(FILE_FOR)
    OUT.mkdir(parents=True, exist_ok=True)
    attempted = ok = 0
    known = []
    for league in leagues:
        if league not in FILE_FOR:
            print(f"skip {league!r}: unknown league; known {list(FILE_FOR)}")
            continue
        attempted += 1
        known.append(league)
    workers = max(1, min(int(os.environ.get("PUBLISH_WORKERS", "1")), len(known)))
    if workers > 1 and len(known) > 1:
        # The four leagues use disjoint source/cache paths and output files. Two
        # workers substantially reduce a cold-cache run without opening four
        # simultaneous scrapers against a free upstream service.
        from concurrent.futures import ThreadPoolExecutor
        print(f"publishing {len(known)} leagues with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_publish_one, lg, FILE_FOR[lg]) for lg in known]
            ok = sum(f.result() for f in futures)
    else:
        ok = sum(_publish_one(lg, FILE_FOR[lg]) for lg in known)
    # Cross-league high-confidence board, built from the frozen picks of every
    # league that just published.
    # Cross-league boards must represent one complete four-league refresh. If a
    # league failed, reading its previous output here would give stale picks a new
    # board timestamp and defeat the browser's staleness warning.
    full_refresh = set(leagues) == set(FILE_FOR) and attempted == len(FILE_FOR)
    boards_safe = full_refresh and ok == attempted
    if boards_safe:
        best = build_best_picks()
        bp = OUT / "best.json"
        if best["_incomplete"]:
            # Refuse rather than publish a record with a league's graded history
            # missing. The previous file stays -- stale by a run, but TRUE, which
            # is the right way round for a scoreboard.
            print(f"  SKIPPED best.json: could not grade {best['_incomplete']}; "
                  f"keeping the last complete board")
            bp = None
        if bp is not None:
            tmp = bp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
            tmp.replace(bp)
            r = best["record"]
            print(f"wrote {bp} - {len(best['upcoming'])} upcoming high-confidence "
                  f"picks, record {r['correct']}-{r['wrong']}")

        pp = build_player_picks()
        ppath = OUT / "player_picks.json"
        if pp["_incomplete"]:
            print(f"  SKIPPED player_picks.json: could not grade "
                  f"{pp['_incomplete']}; keeping the last complete board")
        else:
            tmp = ppath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(pp, indent=2, default=str), encoding="utf-8")
            tmp.replace(ppath)
            pr = pp["record"]
            counts = {mk: sum(1 for u in pp["upcoming"] if u["market"] == mk)
                      for mk in picks.PROP_MARKETS}
            print(f"wrote {ppath} - {len(pp['upcoming'])} upcoming player picks "
                  f"{counts}, record {pr['correct']}-{pr['wrong']}")

        # Model parlays -- built from the two boards just published, frozen and
        # graded with the same discipline. Only when BOTH boards are complete, so
        # a parlay can never stack a leg from a league whose record we could not
        # grade this run.
        if not best["_incomplete"] and not pp["_incomplete"]:
            par = parlays.build_parlays(best, pp, PICKS_DIR / "parlays_log.json")
            parpath = OUT / "parlays.json"
            tmp = parpath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(par, indent=2, default=str), encoding="utf-8")
            tmp.replace(parpath)
            pr2 = par["record"]
            n_par = sum(len(s["parlays"]) for s in par["sections"])
            print(f"wrote {parpath} - {n_par} model parlays, record "
                  f"{pr2['correct']}-{pr2['wrong']}")

        # bet365 6 Scores Challenge -- the model's six Premier League scorelines,
        # frozen and graded like everything else. Independent of the two boards
        # above: it reads the PL payload directly, so a thin player-props run can
        # never suppress it. Publishes an empty board (with a reason) when the
        # week's six have not been announced, rather than inventing a selection.
        try:
            pl_payload = json.loads((OUT / "pl.json").read_text(encoding="utf-8"))
            six_log_path = PICKS_DIR / "pl" / "six_scores_log.json"
            six_log = picks.load_log(six_log_path)
            six = six_scores.build(pl_payload, six_log,
                                   pd.Timestamp.now("UTC"))
            picks.save_log(six_log, six_log_path)
            spath = OUT / "six_scores.json"
            tmp = spath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(six, indent=2, default=str), encoding="utf-8")
            tmp.replace(spath)
            rec = six["record"]
            print(f"wrote {spath} - {len(six['picks'])} scorelines, record "
                  f"{rec['correct']}/{rec['total']}"
                  + (f", MISSING {six['missing_fixtures']}" if six["missing_fixtures"] else ""))
        except Exception as exc:
            # Never let a supplementary board take down a publish.
            print(f"WARNING: 6 Scores board not written ({exc})")

        # Record history -- only when BOTH boards are complete, or a refused board
        # would write a row understating the record and permanently distort the
        # series. A gap in the history is honest; a wrong point is not.
        if not best["_incomplete"] and not pp["_incomplete"]:
            hist = append_history(best, pp)
            hpath = OUT / "record_history.json"
            tmp = hpath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
            tmp.replace(hpath)
            print(f"wrote {hpath} - {len(hist)} snapshots")

    elif ok:
        print("SKIPPED cross-league boards: they require a complete successful "
              "four-league refresh")

    # A scheduled full refresh is atomic at the deployment boundary: individual
    # files may have been written locally, but callers must not commit or deploy
    # them unless all four builds succeeded.
    if attempted and (ok == 0 or (full_refresh and ok != attempted)):
        raise RuntimeError(
            f"only {ok}/{attempted} league publish(es) succeeded; refusing deployment")


if __name__ == "__main__":
    main()
