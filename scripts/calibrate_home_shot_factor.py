"""How much more does a home side actually shoot than an away side?

props.py's HOME_SHOT_FACTOR was originally shipped as a plausible-sounding
constant (1.08) rather than a measurement from this project's own data --
the one real gap a self-audit found: unlike ABSENCE_GOAL_COST (measured by
absence_impact.py) or the second-tier priors (measured by
calibrate_level_gap.py), the shot-side home advantage had never actually been
checked against the historical record this pipeline already fetches.

football-data.co.uk's own CSVs carry HS/AS (full-time home/away shots) columns
alongside the results and odds leagues/history.py already pulls -- so this is
free to measure properly: same source, same seasons, no new dependency.

Symmetric framing, matching how the factor is USED in props.py (home team's
shot budget x HOME_SHOT_FACTOR, away team's / HOME_SHOT_FACTOR): the measured
overall home/away ratio is R, and the symmetric multiplier that reproduces it
is sqrt(R).
"""
import io
import math
import urllib.request

import pandas as pd

from leagues import config

URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"


def fetch_shots(league: str, season: str) -> pd.DataFrame | None:
    lg = config.get(league)
    req = urllib.request.Request(URL.format(season=season, div=lg.fd_code),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        buf = io.StringIO(resp.read().decode("latin-1"))
    df = pd.read_csv(buf)
    if "HS" not in df.columns or "AS" not in df.columns:
        return None
    return df.dropna(subset=["HS", "AS"])[["HS", "AS"]]


def run() -> dict:
    frames = []
    for lg in config.LEAGUES:
        for season in config.get(lg).history_seasons:
            shots = fetch_shots(lg, season)
            if shots is not None:
                frames.append(shots.assign(league=lg))
    all_shots = pd.concat(frames, ignore_index=True)

    overall_ratio = float(all_shots["HS"].mean() / all_shots["AS"].mean())
    per_league = {}
    for lg in config.LEAGUES:
        sub = all_shots[all_shots["league"] == lg]
        r = float(sub["HS"].mean() / sub["AS"].mean())
        per_league[lg] = {"n": int(len(sub)), "ratio": round(r, 4),
                          "symmetric_factor": round(math.sqrt(r), 4)}

    result = {
        "n_matches": int(len(all_shots)),
        "mean_home_shots": round(float(all_shots["HS"].mean()), 3),
        "mean_away_shots": round(float(all_shots["AS"].mean()), 3),
        "overall_ratio": round(overall_ratio, 4),
        "symmetric_factor": round(math.sqrt(overall_ratio), 4),
        "per_league": per_league,
    }
    print(f"n={result['n_matches']}  home {result['mean_home_shots']} vs "
          f"away {result['mean_away_shots']}  ratio {result['overall_ratio']}  "
          f"-> HOME_SHOT_FACTOR {result['symmetric_factor']}")
    for lg, d in per_league.items():
        print(f"  {lg}: n={d['n']}  ratio={d['ratio']}  factor={d['symmetric_factor']}")
    return result


if __name__ == "__main__":
    import json
    out = run()
    json.dump(out, open("data-raw/leagues/home_shot_factor_calibration.json", "w"), indent=2)
    print("\nwrote data-raw/leagues/home_shot_factor_calibration.json")
