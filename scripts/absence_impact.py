"""How much does a team actually lose when a key attacker is absent?

The naive answer -- his whole share of the team's expected goals -- is wrong,
because someone replaces him. What matters is the gap between him and his
replacement, and that gap is what this measures.

METHOD. For every historical match, the fitted model gives an expected goals
figure that knows NOTHING about who played. The residual (actual - expected) is
therefore the part attributable to team news. Regressing that residual on the
share of the team's shooting that was missing estimates the marginal cost of an
absence, controlling for opponent and home advantage through the model itself.

ABSENCE PROXY, and its honest limitation: our only per-match player feed is shot
events, so "absent" means "took no shot". For a high-volume shooter that is a
decent proxy; for anyone else it is mostly noise, which is why the sample is
restricted to players averaging >=1.5 shots per 90. A quiet game still looks like
an absence, so any true effect here is ATTENUATED -- the measured slope is a lower
bound on the real one.
"""
import numpy as np, pandas as pd
from leagues import dataset, players
from leagues.model import LeagueModel

MIN_SHOTS90 = 1.5          # below this, "took no shot" carries no information


def run(league="PL"):
    matches = dataset.build_matches(league)
    matches["date"] = pd.to_datetime(matches["date"])
    ev = players.match_player_stats(league)
    if ev.empty:
        print(f"{league}: no per-match player feed; cannot measure")
        return None
    ev["day"] = pd.to_datetime(ev["date"]).dt.normalize()

    # per player: how often he shoots, and his share of his team's shots
    tot = ev.groupby(["team", "player"])["shots"].sum()
    games = ev.groupby(["team", "player"])["game_id"].nunique()
    team_games = ev.groupby("team")["game_id"].nunique()
    rate = (tot / games.clip(lower=1)).rename("shots_per_app")
    appearance = (games / team_games.reindex(games.index.get_level_values(0)).values)
    share = (tot / tot.groupby(level=0).sum()).rename("shot_share")
    prof = pd.concat([rate, share], axis=1)
    prof["appearance"] = appearance.values
    # regulars who shoot enough that silence is informative
    prof = prof[(prof["shots_per_app"] >= MIN_SHOTS90) & (prof["appearance"] >= 0.5)]

    model = LeagueModel().fit(matches, ref=matches["date"].max())
    rows = []
    played_days = ev.groupby("team")["day"].apply(set).to_dict()
    seen = ev.groupby(["team", "day"])["player"].apply(set).to_dict()

    # CONSECUTIVE-ABSENCE FILTER. "Took no shot" conflates two very different
    # things: a man who was injured, and a man who played badly. The second is a
    # SYMPTOM of the team performing poorly, so regressing goals on it measures
    # reverse causation and would make the model overreact to every absence.
    # A real absence (injury, suspension) spans a RUN of matches; a quiet game is
    # isolated. Requiring two consecutive misses keeps mostly the former.
    order = {t: sorted(days) for t, days in played_days.items()}
    genuine = {}          # (team, day) -> set of players absent for >=2 in a row
    for team, days in order.items():
        regs = prof.loc[team].index if team in prof.index.get_level_values(0) else []
        for i, day in enumerate(days):
            here = seen.get((team, day), set())
            prev = seen.get((team, days[i - 1]), set()) if i else set()
            nxt = seen.get((team, days[i + 1]), set()) if i + 1 < len(days) else set()
            run = {p for p in regs if p not in here
                   and (i and p not in prev or (i + 1 < len(days) and p not in nxt))}
            genuine[(team, day)] = run

    for _, m in matches.iterrows():
        day = m["date"].normalize()
        for side, opp, goals in (("home", "away", m["home_goals"]),
                                 ("away", "home", m["away_goals"])):
            team = m[side]
            if team not in model.attack or m[opp] not in model.attack:
                continue
            if day not in played_days.get(team, set()):
                continue                       # no shot data for this fixture
            lam = model.expected_goals(team, m[opp], home=(side == "home")) \
                  if hasattr(model, "expected_goals") else None
            if lam is None:
                a, d = model.attack[team], model.defence[m[opp]]
                lam = float(np.exp(a + d + (model.home_adv if side == "home" else 0.0)))
            regs = prof.loc[team] if team in prof.index.get_level_values(0) else None
            if regs is None or regs.empty:
                continue
            absent = genuine.get((team, day), set())
            missing = regs[regs.index.isin(absent)]
            rows.append({"missing_share": float(missing["shot_share"].sum()),
                         "resid": float(goals) - lam, "lam": lam})

    d = pd.DataFrame(rows)
    if len(d) < 200:
        print(f"{league}: only {len(d)} usable team-matches; too few")
        return None
    # slope of residual on missing share
    x, y = d["missing_share"].to_numpy(), d["resid"].to_numpy()
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    # bootstrap CI
    rng = np.random.default_rng(7)
    bs = [np.polyfit(x[i], y[i], 1)[0]
          for i in (rng.integers(0, n, n) for _ in range(400))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{league}: n={n}  mean missing share={x.mean():.3f}")
    print(f"   slope = {slope:+.3f} goals per unit of missing shot share "
          f"(95% CI {lo:+.3f} to {hi:+.3f})")
    print(f"   -> losing a player worth 25% of the team's shots costs "
          f"{-slope*0.25:+.3f} goals" if slope < 0 else
          f"   -> no cost detected")
    signif = hi < 0
    print(f"   statistically distinguishable from zero: {signif}")
    return {"league": league, "n": n, "slope": float(slope),
            "ci_low": float(lo), "ci_high": float(hi), "significant": bool(signif)}


if __name__ == "__main__":
    import json, sys
    out = [r for r in (run(l) for l in (sys.argv[1:] or ["PL", "LALIGA", "LIGUE1"])) if r]
    json.dump(out, open("data-raw/leagues/absence_impact.json", "w"), indent=2)
