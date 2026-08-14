"""Promoted-club prior from the club's SECOND-TIER season.

A promoted club has no top-flight history to fit on. ClubElo used to supply the
prior, but it is a single third-party point of failure (down for days at a time).
football-data.co.uk publishes the second divisions (E1/SP2/D2/F2) from the SAME
feed as our top-flight results, so we derive the prior from the club's actual
promotion-season goal form.

HOW MUCH DOES SECOND-TIER FORM TRANSLATE? Calibrated (scripts/calibrate_level_gap.py)
against every promoted club in the last five seasons, mapping each club's second-tier
deviation to its ACTUAL top-flight deviation:

  ATTACK :  tf_dev = 0.40 * st_dev - 0.39   (r=+0.29)  -- a mild signal, but a big
            negative intercept: every promoted club starts well below the top-flight
            scoring average, and scoring a lot in the 2nd tier only slightly offsets it.
  DEFENCE:  tf_dev = +0.19  (constant)       (r=-0.04)  -- NO signal at all. How well a
            club defended in the 2nd tier says nothing about the top flight; promoted
            clubs concede ~0.19 above average across the board. So the slope is zeroed
            and the prior is just the promoted-club mean.

This is strictly more honest than a single "gap" damping factor (which ignored the
level shift and produced nonsense), and than the weakest-club fallback (which ignored
the club's form entirely).
"""
import functools
import io
import urllib.request

import numpy as np
import pandas as pd

from leagues import config
from leagues.names import canonical, UnknownTeam

FEED = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# Fitted top-flight deviation = slope * second-tier deviation + intercept.
ATTACK_MAP = (0.40, -0.39)
DEFENCE_MAP = (0.0, 0.19)     # slope zeroed: 2nd-tier defence does not translate


def table_strengths(buf, league: str) -> dict:
    """Per-team goals-for / goals-against per game from a second-tier CSV, plus
    the league's average goals/game (the strength baseline). Names left raw."""
    d = pd.read_csv(buf).dropna(subset=["FTHG", "FTAG"])
    agg = {}
    for _, r in d.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        for t in (h, a):
            agg.setdefault(t, {"gf": 0, "ga": 0, "p": 0})
        agg[h]["gf"] += hg; agg[h]["ga"] += ag; agg[h]["p"] += 1
        agg[a]["gf"] += ag; agg[a]["ga"] += hg; agg[a]["p"] += 1
    total_p = sum(t["p"] for t in agg.values())
    lg = sum(t["gf"] for t in agg.values()) / max(total_p, 1)
    return {t: {"gf_pg": v["gf"] / v["p"], "ga_pg": v["ga"] / v["p"], "lg": lg}
            for t, v in agg.items() if v["p"] > 0}


def promoted_deviations(strengths: dict, teams: list) -> dict:
    """team -> (attack_dev, defence_dev): predicted TOP-FLIGHT deviations from the
    fitted map. A club absent from `strengths` is omitted (never guessed)."""
    out = {}
    for t in teams:
        s = strengths.get(t)
        if not s:
            continue
        st_att = np.log(max(s["gf_pg"], 0.3) / s["lg"])
        st_def = np.log(max(s["ga_pg"], 0.3) / s["lg"])
        att = ATTACK_MAP[0] * st_att + ATTACK_MAP[1]
        dfn = DEFENCE_MAP[0] * st_def + DEFENCE_MAP[1]
        out[t] = (float(att), float(dfn))
    return out


def season_code(start_year: int) -> str:
    """1995 -> '9596', 2025 -> '2526'. football-data.co.uk's season directory."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_code_start_year(code: str) -> int:
    """'2526' -> 2025. Inverse of season_code for the 2000s; football-data has
    used two-digit codes since 1993, so a pre-2000 code would need a century
    rule. This repo's history_seasons never reach back that far."""
    return 2000 + int(str(code)[:2])


def season_start_year(date) -> int:
    """The year a European season starting in `date`'s campaign began. July is the
    boundary: 2026-08-15 and 2027-05-01 both belong to the season starting 2026."""
    d = pd.Timestamp(date)
    return d.year if d.month >= 7 else d.year - 1


def feeder_season(date) -> str:
    """Second-tier season that fed the clubs promoted INTO the top-flight season
    containing `date`. A club playing its first top-flight game in August 2026 was
    in the second tier during 2025-26, so the feeder season is one behind."""
    return season_code(season_start_year(date) - 1)


def promotion_season(league: str) -> str:
    """The second-tier season feeding the CURRENT top-flight campaign.

    Derived from the league's last completed history season rather than hardcoded.
    A literal (it was "2526") is right for exactly one year and then silently seeds
    promoted clubs from a two-year-old table -- nothing fails, the priors are just
    quietly wrong. `history_seasons` has to be rolled forward each summer for the
    model to fit on current results at all, so tying the two together means the
    promotion season can no longer be forgotten on its own.
    """
    return config.get(league).history_seasons[-1]


@functools.lru_cache(maxsize=None)
def _fetch_strengths_cached(league: str, season: str) -> tuple:
    code = config.get(league).fd_code2
    req = urllib.request.Request(FEED.format(season=season, code=code),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = table_strengths(io.StringIO(resp.read().decode("utf-8", "replace")), league)
    out = {}
    for team, s in raw.items():
        try:
            out[canonical(team, league)] = s
        except UnknownTeam:
            continue
    # Cached as an immutable tuple of pairs so a caller cannot mutate the shared
    # entry; fetch_strengths rebuilds a fresh dict per call.
    return tuple((t, tuple(sorted(s.items()))) for t, s in out.items())


def fetch_strengths(league: str, season: str = None) -> dict:
    """Download one league's second-tier table, keyed by CANONICAL club name.
    Unmapped second-tier spellings are dropped (the caller warns on a promoted
    club that fails to resolve).

    MEMOIZED per (league, season). The walk-forward backtest asks for priors at
    every weekly cutoff -- hundreds per league per config -- but there are only a
    handful of distinct seasons, so without the cache the gate would spend its
    entire runtime re-downloading the same five CSVs.
    """
    season = season or promotion_season(league)
    return {t: dict(s) for t, s in _fetch_strengths_cached(league, season)}


def second_tier_priors(base_model, league: str, teams: list,
                       season: str = None) -> dict:
    """team -> (attack, defence) prior on the fitted model's strength scale.

    The fitted map gives DEVIATIONS from the top-flight average; `base_model`
    supplies that average (mean of its fitted strengths), matching how elo_priors
    and promoted_priors place a prior on the same scale. Returns only the teams it
    could resolve in the second-tier feed; the caller falls back for the rest.

    `season` defaults to the league's CURRENT promotion season (see
    promotion_season). The backtest passes an explicit season instead, so a
    2022 cutoff is seeded from the 2021-22 second tier rather than from a table
    that had not been played yet.
    """
    if not teams or not base_model.attack:
        return {}
    a_mean = float(np.mean(list(base_model.attack.values())))
    d_mean = float(np.mean(list(base_model.defence.values())))
    strengths = fetch_strengths(league, season)
    return {t: (a_mean + da, d_mean + dd)
            for t, (da, dd) in promoted_deviations(strengths, teams).items()}
