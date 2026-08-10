"""Player props: anytime scorer, shots, shots on target.

Pipeline: rates -> shrinkage -> expected minutes -> penalties -> RESCALE to the
match model's team lambda -> Poisson props.

The rescale (match_props) is load-bearing: it is the channel by which opponent
strength and home advantage reach the GOALSCORER number -- without it a striker's
number is identical against Brentford and Liverpool, a season-long rate table
wearing a costume. Shots and shots-on-target are NOT rescaled the same way (they
are not bounded by any team-lambda target), so their venue-awareness comes from a
second, separate channel: HOME_SHOT_FACTOR below.

All constants below are PROVISIONAL: they come from the design spec's research,
not from our own data. props_backtest.py is what tunes them.
"""
import numpy as np
import pandas as pd

# non-penalty goals per 90, by primary position
GOAL_PRIORS = {"FW": 0.45, "AM": 0.20, "MF": 0.10, "DF": 0.05, "GK": 0.001}
# total shots per 90, by primary position
SHOT_PRIORS = {"FW": 2.5, "AM": 1.4, "MF": 0.9, "DF": 0.4, "GK": 0.02}
SOT_RATIO_PRIOR = 0.35
SOT_PRIOR_SHOTS = 10.0    # shrinkage strength for the on-target ratio, in shots
# MEASURED (scripts/calibrate_home_shot_factor.py), not guessed. Home sides
# average 13.888 shots per match vs 11.309 away across 7,007 matches -- 5 seasons,
# all four leagues, from the same football-data.co.uk source leagues/history.py
# already fetches (HS/AS columns, previously unused). That is a ratio of 1.228;
# applied SYMMETRICALLY to a player's shot budget (home x, away 1/x) so a
# fixture's total shot volume barely moves but home/away players stop being
# projected identically, the symmetric factor is sqrt(1.228) = 1.108. Per-league
# figures (1.09-1.13, see data-raw/leagues/home_shot_factor_calibration.json) are
# close enough to pool into one constant, the same call already made for
# ABSENCE_GOAL_COST. An earlier version of this shipped as a plausible-sounding
# 1.08 before this was measured -- close, but a guess is still a guess.
HOME_SHOT_FACTOR = 1.108

SEASON_DECAY = 0.7        # alpha ^ (seasons ago)
K_NINETIES = 7.0          # empirical-Bayes strength, in 90s
W_REALIZED_HIGH = 0.6     # weight on actual goals for high-minute players
W_REALIZED_LOW = 0.4      # ...and for low-sample players (trust xG more)
HIGH_MINUTE_90S = 10.0
PEN_CONVERSION = 0.76
# A player flagged doubtful is assumed to play about half a match: he may start
# and be withdrawn, or come off the bench. Deliberately blunt -- we have a status
# word, not a minutes forecast, so pretending to more precision would be false.
DOUBT_MINUTES_FACTOR = 0.5
CONFIRMED_BENCH_APPEARANCE = 0.35
CONFIRMED_BENCH_MINUTES = 25.0

# MEASURED OVERCONFIDENCE CORRECTION, p -> p**k.
#
# scripts/props_pick_calibration.py asked the question the rate gate never did:
# of the picks the board actually PUBLISHED, what fraction won? Trained through
# 2024-25, scored on 2025-26, graded by the same harsh rule production uses live
# (no shot row = wrong). Every published probability already multiplies by the
# appearance probability, so a player who did not feature is priced in and the
# harsh grading is consistent with the claim -- the shortfall is real, not an
# artefact. In the two leagues with clean data (0% of picks on players who had
# left) every market came in short:
#     PL      goal 42.1 -> 15.6   shots 75.0 -> 50.0   sot 66.8 -> 55.6
#     LALIGA  goal 42.8 -> 29.5   shots 77.0 -> 60.2   sot 69.8 -> 61.1
#
# A power transform is used because it is monotone (it can never reorder the
# board), fixes 0 and 1, and has one parameter -- with ~100-150 picks per market
# anything richer fits noise. Each k was fitted on one league and tested on the
# other; the value kept here is the SMALLER (weakest) of the two, which is
# strictly better than no correction in BOTH leagues while being unable to
# over-deflate the league it was not fitted on.
#
# Confidence differs by market and is not hidden: sot cross-validates tightly
# (residual +/-1.8pp), shots is directionally solid but its two estimates
# disagree (1.95 vs 2.44), and goal does NOT generalise (2.15 vs 1.44, residual
# +/-13pp) -- so goal gets the most conservative treatment of the three and its
# number should be re-fitted once the live record has real volume.
#
# This corrects the PUBLISHED PROBABILITY only. lambda_goals still sums to the
# match model's team lambda (that invariant is untouched), and exp_shots/exp_sot
# remain the raw expected counts, so a calibrated probability and its expected
# count are deliberately not forced into agreement.
PROP_CALIBRATION = {"goal": 1.437, "shots": 1.954, "sot": 1.376}


def _calibrated(p: float, market: str) -> float:
    """Apply the measured overconfidence correction to a 0-1 probability."""
    k = PROP_CALIBRATION.get(market, 1.0)
    return float(np.clip(p, 0.0, 1.0)) ** k


def _prior(pos: str, table: dict, default_key: str = "MF") -> float:
    return table.get(pos, table[default_key])


def player_rates(logs: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """Decay-weighted, shrunk per-90 rates for every player in the logs.

    One row per player: team, pos, nineties, rate90 (non-penalty goals),
    shots90, sot_ratio.
    """
    df = logs.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Clipping the age at 0 would give a log dated AFTER ref the weight 0.7**0 = 1,
    # i.e. the MAXIMUM -- the exact lookahead leak weights.py documents fixing for
    # the match model. player_rates never got the same treatment. Not reachable
    # today (publish uses ref=now against season-end rows) but it is a live trap
    # for any future per-match player feed, so close it here.
    age_years = (ref - df["date"]).dt.days / 365.25
    df["w"] = np.where(age_years >= 0, SEASON_DECAY ** age_years.clip(lower=0), 0.0)
    df["w90"] = df["w"] * df["minutes"] / 90.0

    out = []
    for (team, player), g in df.groupby(["team", "player"], sort=False):
        n90 = float(g["w90"].sum())
        if n90 <= 0:
            continue
        pos = g["pos"].mode().iat[0] if not g["pos"].mode().empty else "MF"

        # 1. blend realized goals with xG; trust xG more when the sample is thin,
        #    because goals are near-pure noise at low volume.
        w_real = W_REALIZED_HIGH if n90 >= HIGH_MINUTE_90S else W_REALIZED_LOW
        g_per90 = float((g["np_goals"] * g["w"]).sum()) / n90
        x_per90 = float((g["npxg"] * g["w"]).sum()) / n90
        obs_goal = w_real * g_per90 + (1 - w_real) * x_per90
        obs_shot = float((g["shots"] * g["w"]).sum()) / n90

        shots_tot = float((g["shots"] * g["w"]).sum())
        sot_tot = float((g["sot"] * g["w"]).sum())

        # 2. empirical-Bayes shrinkage toward the position prior
        k = K_NINETIES
        rate90 = (n90 * obs_goal + k * _prior(pos, GOAL_PRIORS)) / (n90 + k)
        shots90 = (n90 * obs_shot + k * _prior(pos, SHOT_PRIORS)) / (n90 + k)
        # the on-target ratio shrinks on SHOTS taken, not on nineties
        sot_ratio = ((sot_tot + SOT_PRIOR_SHOTS * SOT_RATIO_PRIOR)
                     / (shots_tot + SOT_PRIOR_SHOTS))

        out.append({"team": team, "player": player, "pos": pos, "nineties": n90,
                    "rate90": rate90, "shots90": shots90, "sot_ratio": sot_ratio})

    # An empty logs frame otherwise yields a 0x0 frame with NO columns, so the
    # caller's rates["player"] raises a bare KeyError that reads like a schema
    # change. Return the real shape instead and let it degrade like every other
    # feed here.
    return pd.DataFrame(out, columns=["team", "player", "pos", "nineties",
                                      "rate90", "shots90", "sot_ratio"])


def match_props(rates: pd.DataFrame, home: str, away: str,
                lam_home: float, lam_away: float,
                minutes: dict | None = None,
                pen_taker: dict | None = None,
                opp_shot_factor: dict | None = None,
                exp_pens: dict | None = None,
                unavailable: set | None = None,
                doubtful: set | None = None,
                playing_time: dict | None = None,
                confirmed_starters: set | None = None,
                confirmed_bench: set | None = None) -> list[dict]:
    """Per-player props for ONE fixture.

    lam_home/lam_away are the match model's fitted team goal expectations
    (LeagueModel.predict -> lambda_home/lambda_away). Every player's goal lambda
    is rescaled so that each team's players sum to exactly that number.

    minutes:         player -> expected minutes (default 90)
    pen_taker:       team -> the player who takes penalties
    opp_shot_factor: team -> multiplier for how many shots the OPPONENT concedes
                     relative to league average (1.0 = average)
    exp_pens:        team -> expected penalties awarded in this match
    """
    minutes = minutes or {}
    pen_taker = pen_taker or {}
    opp_shot_factor = opp_shot_factor or {}
    exp_pens = exp_pens or {}

    unavailable = unavailable or set()
    doubtful = doubtful or set()
    playing_time = playing_time or {}
    confirmed_starters = confirmed_starters or set()
    confirmed_bench = confirmed_bench or set()

    out = []
    for team, lam_team in ((home, lam_home), (away, lam_away)):
        squad = rates[rates["team"] == team].copy()
        # Players confirmed OUT are dropped BEFORE the rescale, so their expected
        # goals are redistributed across the team-mates who are actually playing
        # rather than vanishing: the team's lambdas still sum to lam_team, which is
        # the match model's figure and must not change because a striker is injured.
        if len(unavailable):
            squad = squad[~squad["player"].isin(unavailable)]
        if squad.empty:
            continue

        # A doubtful player still features, but at reduced expected minutes -- the
        # rescale then hands the difference to the rest of the squad.
        appearance, conditional = [], []
        for player in squad["player"]:
            pt = playing_time.get(player) or {}
            expected = float(minutes.get(player, 90.0))
            p_app = float(pt.get("appearance_prob", 1.0))
            mins_if = float(pt.get("minutes_if_playing",
                                   expected / p_app if p_app > 0 else 0.0))
            if player in confirmed_starters:
                p_app = 1.0
                mins_if = max(mins_if, 70.0)
            elif player in confirmed_bench:
                p_app = CONFIRMED_BENCH_APPEARANCE
                mins_if = min(mins_if, CONFIRMED_BENCH_MINUTES)
            if player in doubtful and player not in confirmed_starters:
                p_app *= DOUBT_MINUTES_FACTOR
            appearance.append(float(np.clip(p_app, 0.0, 1.0)))
            conditional.append(float(np.clip(mins_if, 0.0, 90.0)))
        squad["appearance_prob"] = appearance
        squad["minutes_if_playing"] = conditional
        squad["exp_min"] = squad["appearance_prob"] * squad["minutes_if_playing"]
        squad["raw"] = squad["rate90"] * squad["exp_min"] / 90.0

        # Penalties are a TEAM property: take them out of the open-play budget
        # and hand them to exactly one player, rather than smearing them across
        # every attacker. Only reserve the penalty mass if the named taker is
        # actually in this squad -- otherwise it would be subtracted from open
        # play and handed to nobody, so the team's players would sum below
        # lam_team and every scorer would be understated.
        taker = pen_taker.get(team)
        has_taker = bool((squad["player"] == taker).any()) if taker else False
        taker_p = (float(squad.loc[squad["player"] == taker, "appearance_prob"].iloc[0])
                   if has_taker else 0.0)
        # Reserve only the penalty mass attributable to the primary taker actually
        # appearing. If he misses out, an unknown deputy takes it; that residual
        # stays in the open team budget instead of being falsely assigned.
        lam_pen = (float(exp_pens.get(team, 0.0)) * PEN_CONVERSION * taker_p
                   if has_taker else 0.0)
        lam_pen = min(lam_pen, max(lam_team - 1e-6, 0.0))
        lam_open = max(lam_team - lam_pen, 0.0)

        total_raw = float(squad["raw"].sum())
        # total_raw > 0 whenever any player has minutes (rate90 is shrunk toward a
        # strictly positive prior), but guard the division regardless.
        scale = (lam_open / total_raw) if total_raw > 0 else 0.0
        # Home sides take meaningfully more shots than away sides in the same
        # matchup. The goalscorer market inherits that venue effect through the
        # match lambda (which IS venue-aware), but the shot / shot-on-target
        # markets are built from season shot rates and only saw opp_shot_factor --
        # an opponent AGGREGATE, not venue-split -- so without this they were
        # identical home or away, systematically understating home players and
        # overstating away ones. Fold a modest, symmetric venue multiplier into
        # the shot budget so the two shot markets are venue-aware too. Conservative
        # and provisional (see HOME_SHOT_FACTOR); goals are untouched.
        venue = HOME_SHOT_FACTOR if team == home else 1.0 / HOME_SHOT_FACTOR
        factor = float(opp_shot_factor.get(team, 1.0)) * venue

        for _, r in squad.iterrows():
            lam_goals = float(r["raw"]) * scale
            is_taker = bool(r["player"] == taker)
            if is_taker:
                lam_goals += lam_pen

            p_app = float(r["appearance_prob"])
            mins_if = float(r["minutes_if_playing"])
            lam_goals_if_playing = lam_goals / p_app if p_app > 0 else 0.0
            anytime = p_app * (1.0 - np.exp(-lam_goals_if_playing))
            s_if_playing = float(r["shots90"]) * mins_if / 90.0 * factor
            sot_if_playing = s_if_playing * float(r["sot_ratio"])
            exp_shots = p_app * s_if_playing
            exp_sot = p_app * sot_if_playing
            p_shots = p_app * (1.0 - np.exp(-s_if_playing) * (1.0 + s_if_playing))
            p_sot = p_app * (1.0 - np.exp(-sot_if_playing))
            # Measured correction, applied last so it acts on the finished
            # probability rather than distorting any upstream expectation.
            anytime = _calibrated(anytime, "goal")
            p_shots = _calibrated(p_shots, "shots")
            p_sot = _calibrated(p_sot, "sot")

            out.append({
                "team": team,
                "player": r["player"],
                "position": r["pos"],
                "lambda_goals": lam_goals,
                "anytime_pct": round(100.0 * anytime, 1),
                "exp_shots": round(exp_shots, 2),
                "p_shots_2plus": round(100.0 * p_shots, 1),
                "exp_sot": round(exp_sot, 2),
                "p_sot_1plus": round(100.0 * p_sot, 1),
                "appearance_pct": round(100.0 * p_app, 1),
                "expected_minutes": round(p_app * mins_if, 1),
                "penalty_taker": is_taker,
                "doubt": bool(r["player"] in doubtful),
            })
    return out


# Goals a team loses per unit of missing SHOT SHARE when a player is genuinely
# absent. Measured, not guessed: scripts/absence_impact.py regresses the model's
# goal residual on the share of a team's shooting that was missing, across 2,888
# Premier League team-matches. Point estimate -0.77 (95% CI -1.33 to -0.22).
#
# We deliberately use a value well BELOW the point estimate, for two reasons.
# First, the confidence interval is wide -- the pessimistic end is six times the
# optimistic one. Second, the absence proxy is "took no shot", which even after
# filtering to runs of consecutive misses still catches some players dropped for
# bad form; that residual confounding inflates the estimate, so 0.77 is an upper
# bound on the causal effect rather than a central one.
#
# Under-reacting costs a little accuracy. Over-reacting would swing a win
# probability by ten points on one absence and make us worse than the market we
# already trail. When the direction of the error is asymmetric, take the cautious
# side.
#
# NOT SPLIT BY POSITION. scripts/absence_impact.run_by_position() tested FW/AM/MF
# as separate coefficients across PL, La Liga and Ligue 1 (see
# data-raw/leagues/absence_impact_by_position.json). Results did not replicate:
# each bucket was statistically distinguishable from zero in at most one of the
# three leagues, with confidence intervals that cross zero and change sign
# between leagues, and the AM bucket had essentially no identifiable data at all
# (Understat rarely tags a player "AM", so missing_AM is ~0 for almost every
# team-match and the coefficient is unidentified, not zero). Splitting one modest
# pooled sample three ways produced noise, not three real numbers -- same
# rejection logic as weibull_experiment.json. Kept as ONE pooled constant.
#
# NOT EXTENDED TO DEFENSIVE/GK ABSENCES, for a different reason: it can't be
# measured at all with this data. The proxy this whole mechanism rests on --
# "took no shot" -- is structurally uninformative for a position that takes ~0
# shots in a normal match anyway. A missing center-back or keeper is invisible to
# this method, not just imprecisely measured, and there is no historical lineup
# archive to fall back on (news.json is live-only; see scripts/sync_lineups.py).
# unmodeled_absentee_positions() below surfaces this explicitly rather than
# letting a confirmed defensive absence silently move nothing.
ABSENCE_GOAL_COST = 0.45

# A player flagged doubtful (not confirmed out) is weighted the same way the
# props pipeline already treats him (see DOUBT_MINUTES_FACTOR): about a 50%
# chance of featuring, rather than either the full penalty (treating "doubtful"
# as "out") or none at all (treating him as fully fit). Before this, a doubtful
# key attacker moved his own player-prop numbers but left the match-outcome
# model rating his team at full strength -- an inconsistency between the two
# boards' handling of the exact same team-news fact, not a new measured effect.
DOUBTFUL_ABSENCE_WEIGHT = DOUBT_MINUTES_FACTOR

# Positions absence_penalty's shot-share proxy cannot see, per the note above.
UNMODELED_ABSENCE_POSITIONS = {"DF", "GK"}


def absence_penalty(rates: pd.DataFrame, team: str, unavailable,
                    doubtful=None, cost: float = ABSENCE_GOAL_COST,
                    doubt_weight: float = DOUBTFUL_ABSENCE_WEIGHT) -> float:
    """Goals to subtract from a team's lambda for confirmed and doubtful absences.

    Scaled by the absent players' share of the team's SHOTS, not by their goals:
    shot share is the more stable quantity and is what the measurement above was
    fitted on. Confirmed-out players count fully; doubtful players count at
    `doubt_weight`, matching how the props pipeline already treats the same fact.
    Returns 0.0 whenever we cannot compute it, so a data gap never silently moves
    a published prediction.
    """
    if rates.empty or not (unavailable or doubtful):
        return 0.0
    squad = rates[rates["team"] == team]
    total = float(squad["shots90"].sum())
    if total <= 0:
        return 0.0
    out = set(unavailable or set())
    doubt = set(doubtful or set()) - out       # confirmed-out takes precedence
    missing_out = float(squad[squad["player"].isin(out)]["shots90"].sum())
    missing_doubt = float(squad[squad["player"].isin(doubt)]["shots90"].sum())
    weighted_share = (missing_out + doubt_weight * missing_doubt) / total
    return cost * weighted_share


def unmodeled_absentee_positions(rates: pd.DataFrame, team: str, unavailable) -> list:
    """Confirmed-out players this team is missing whose position absence_penalty
    cannot price in (see UNMODELED_ABSENCE_POSITIONS above). Purely informational
    -- callers can surface it as a warning rather than let a defensive or
    goalkeeper absence look accounted for when it silently isn't."""
    if rates.empty or not unavailable:
        return []
    squad = rates[(rates["team"] == team) & (rates["player"].isin(set(unavailable)))]
    hits = squad[squad["pos"].isin(UNMODELED_ABSENCE_POSITIONS)]
    return sorted(hits["player"].tolist())


def thin_squads(rates: pd.DataFrame, teams, min_players: int) -> list:
    """Teams with SOME player data but too little to share out a team's goals.

    The rescale in match_props forces a team's players to sum to the match model's
    lambda. With a handful of players that is arithmetically fine and factually
    absurd: a promoted club with one player of top-flight history had the whole
    team's 1.30 expected goals land on him, publishing a 72.8% anytime scorer when
    nothing else in four leagues exceeded 50.8%.

    Teams with NO data are excluded elsewhere and are not the danger -- an empty
    card is visibly missing, whereas one inflated name looks like the best pick on
    the board. Returned sorted so the caller can report them.
    """
    if rates.empty:
        # No player data at all for ANY team. That is the missing_squads case, not
        # the thin case -- returning every team here inverted the function's own
        # rule (0 < count < min) and would have flagged a full league as thin.
        return []
    counts = rates.groupby("team").size()
    return sorted(t for t in teams if 0 < int(counts.get(t, 0)) < min_players)


def top_props(props: list[dict], team: str, n: int = 3) -> list[dict]:
    """Top-n players for a team by anytime probability."""
    squad = [p for p in props if p["team"] == team]
    return sorted(squad, key=lambda p: p["anytime_pct"], reverse=True)[:n]
