"""The real league table, and the fixtures that never made it into the record.

BOTH LIVE HERE RATHER THAN IN publish.py SO A LIGHT JOB CAN USE THEM. Neither
needs a model -- one counts goals, the other compares two lists -- but importing
them from `leagues.publish` drags in penaltyblog, scipy, sklearn and soccerdata,
which is minutes of install for two pure functions. `scripts/refresh_results.py`
runs in the fast lock job on a pandas-only environment and calls both.

Same reasoning that put LOCK_WINDOW_HOURS in config.py: the fast path must not
have to import the slow path to learn something simple. publish.py re-exports
these, so there is still exactly one implementation.
"""
import pandas as pd


def unrecorded_fixtures(played, log, key_for) -> list[dict]:
    """Played fixtures that carry NO frozen pick, so are absent from the record.

    A pick enters the record only if a locking run happened before kickoff. When
    the scheduler drops runs, a fixture can be shown on the board with a pick,
    be played, and then leave no trace at all -- not graded, not void, simply
    gone. On 2026-08-28 that took four matches across four leagues, including
    Bayern Munich v Stuttgart and Lille v Paris SG.

    A record that silently omits them is not merely incomplete, it is a BIASED
    SAMPLE: it describes the fixtures that happened to kick off near a workflow
    run. `leagues.lockwindow` narrows how often this happens; publishing the list
    is what stops the remainder being invisible. A reader can see 5-4 and also
    see that two played fixtures are not in it.
    """
    out = []
    for _, m in played.iterrows():
        if key_for(m["match_id"]) in log:
            continue
        out.append({
            "date": str(m["date"]),
            "home": m["home"], "away": m["away"],
            "score": (f"{int(m['home_goals'])}-{int(m['away_goals'])}"
                      if pd.notna(m.get("home_goals")) else None),
            "reason": "no locking run happened before kickoff",
        })
    return sorted(out, key=lambda r: r["date"])


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
