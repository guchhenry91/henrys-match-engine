"""The rebuilt soccer player-prop model: anytime scorer, 2+ shots, 1+ on target.

docs/REBUILD_PLAN.md. The old props (leagues/props.py) multiplied season-long per-90
rates by an assumed 90 minutes, rescaled to the match model's lambda, then needed a
large measured "overconfidence" power correction (goal 42% stated -> 16-30% landed).
Its two weak points were exactly what Understat's per-MATCH data can now supply:
WHO PLAYS, and HOW LONG.

This model works per player per match, from entering features only:

  1. ROLE -- P(starts), P(comes off the bench), P(does not appear), a gradient-boosted
     classifier on his recent starts and appearances for THIS team (a candidate is
     anyone who appeared for the team in its last CANDIDATE_WINDOW matches). Live, a
     confirmed XI replaces it: starters 1, named bench his sub rate.
  2. OUTCOME given the role -- a classifier per market on his recent per-90 shots,
     shots on target, non-penalty xG, goals, xA, minutes, his share of the team's
     PENALTIES, his team's and the opponent's xG for/against, and home/away, with
     "started" as an input so a substitute's shorter game is learned, not assumed.
  3. P(market) = P(start) * P(market | start) + P(sub) * P(market | sub). Not
     appearing is a loss, exactly as the record grades it.

The bookmaker's price is never an input.
Data: Understat's per-match rosters and shots (scripts/export_understat_matches.py),
trimmed to one row per player per match (trim()).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MATCHES_DIR = ROOT / "data-raw" / "leagues" / "understat_matches"
LEAGUES = ("PL", "LALIGA", "BUNDESLIGA", "SERIEA", "LIGUE1")
MARKETS = {"goal": "scored", "shots": "shots2", "sot": "sot1"}
CANDIDATE_WINDOW = 6
TRAIN_SEASONS = 4
POS = {"GK": 0, "D": 1, "DM": 2, "M": 3, "AM": 4, "F": 5}


def _pos_group(p: str) -> str:
    p = str(p or "")
    if p == "GK":
        return "GK"
    if p.startswith("FW"):
        return "F"
    if p.startswith("AM"):
        return "AM"
    if p.startswith("DM"):
        return "DM"
    if p.startswith("D"):
        return "D"
    if p == "Sub":
        return ""
    return "M"


def trim(players: pd.DataFrame, shots: pd.DataFrame, league: str, season: int) -> pd.DataFrame:
    """One row per player per match: who, for whom, how long, and what he did."""
    p = players.copy()
    p["team"] = np.where(p["h_a"] == "h", p["home"], p["away"])
    p["opponent"] = np.where(p["h_a"] == "h", p["away"], p["home"])
    p["is_home"] = (p["h_a"] == "h").astype(int)
    p["team_goals"] = np.where(p["h_a"] == "h", p["home_goals"], p["away_goals"])
    p["opp_goals"] = np.where(p["h_a"] == "h", p["away_goals"], p["home_goals"])
    p["team_xg"] = np.where(p["h_a"] == "h", p["home_xg"], p["away_xg"])
    p["opp_xg"] = np.where(p["h_a"] == "h", p["away_xg"], p["home_xg"])
    s = shots.copy()
    s = s[s["result"] != "OwnGoal"]
    s["on_target"] = s["result"].isin(["Goal", "SavedShot"]).astype(int)
    s["pen"] = (s["situation"] == "Penalty").astype(int)
    s["pen_goal"] = ((s["situation"] == "Penalty") & (s["result"] == "Goal")).astype(int)
    s["npxg"] = np.where(s["situation"] == "Penalty", 0.0, s["xG"].astype(float))
    agg = s.groupby(["match_id", "player_id"], as_index=False).agg(
        sot=("on_target", "sum"), pens=("pen", "sum"), pen_goals=("pen_goal", "sum"),
        npxg=("npxg", "sum"))
    p = p.merge(agg, on=["match_id", "player_id"], how="left")
    for c in ("sot", "pens", "pen_goals", "npxg"):
        p[c] = p[c].fillna(0.0)
    p["started"] = (p["position"] != "Sub").astype(int)
    p["league"], p["season"] = league, int(season)
    p["date"] = pd.to_datetime(p["datetime"]).dt.strftime("%Y-%m-%d")
    keep = ["league", "season", "match_id", "date", "team", "opponent", "is_home",
            "player_id", "player", "position", "started", "time", "goals", "shots", "sot",
            "pens", "pen_goals", "npxg", "xA", "key_passes", "team_goals", "opp_goals",
            "team_xg", "opp_xg"]
    return p[keep].rename(columns={"time": "minutes"})


def load(seasons, leagues=LEAGUES, base: Path | None = None) -> pd.DataFrame:
    base = Path(base or MATCHES_DIR)
    frames = []
    for lg in leagues:
        for s in seasons:
            f = base / f"{lg}_{s}.csv.gz"
            if f.exists():
                frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10])
    d["player_id"] = d["player_id"].astype(int)
    return d


def _team_matches(d: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, with ENTERING xG form for the team and opponent."""
    t = (d.groupby(["league", "season", "match_id", "date", "team", "opponent", "is_home"],
                   as_index=False)
         .agg(team_xg=("team_xg", "first"), opp_xg=("opp_xg", "first"),
              team_pens=("pens", "sum"), team_shots=("shots", "sum")))
    t = t.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    g = t.groupby("team", sort=False)
    for c, span in (("team_xg", 10), ("opp_xg", 10), ("team_pens", 38), ("team_shots", 10)):
        t[f"{c}_in"] = g[c].transform(lambda x, k=span: x.shift(1).ewm(span=k, min_periods=1).mean())
    t["team_seq"] = g.cumcount()
    opp = t[["match_id", "team", "team_xg_in", "opp_xg_in"]].rename(
        columns={"team": "opponent", "team_xg_in": "opp_attack_in", "opp_xg_in": "opp_defence_in"})
    return t.merge(opp, on=["match_id", "opponent"], how="left")


def build(d: pd.DataFrame, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candidate rows (player x team-match) with entering features and outcomes.

    `upcoming`: rows [league, season, match_id, date, team, opponent, is_home] for
    matches not yet played; their candidates get features and NaN outcomes."""
    d = d.copy()
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        up["date"] = pd.to_datetime(up["date"])
        teams = _team_matches(pd.concat([d, up.assign(team_xg=np.nan, opp_xg=np.nan, pens=0, shots=0)],
                                        ignore_index=True))
    else:
        teams = _team_matches(d)
    # STATUS GRID: every player who ever appeared for a team, across that team's
    # matches -- 2 started, 1 came on, 0 did not appear.
    d["status"] = np.where(d["started"] == 1, 2, 1)
    tm = teams[["team", "match_id", "team_seq", "date"]]
    first = d.groupby(["team", "player_id"], as_index=False).agg(first_date=("date", "min"))
    grid = tm.merge(first, on="team")
    grid = grid[grid["date"] > grid["first_date"]]              # only after his first game there
    stat_cols = ["status", "minutes", "goals", "shots", "sot", "npxg", "xA", "pens", "pen_goals",
                 "position", "player", "started"]
    grid = grid.merge(d[["team", "match_id", "player_id"] + stat_cols], how="left",
                      on=["team", "match_id", "player_id"])
    own = d.merge(tm[["team", "match_id", "team_seq"]], on=["team", "match_id"])
    own = own.merge(first, on=["team", "player_id"])
    own = own[own["date"] <= own["first_date"]]                 # his first game there
    grid = pd.concat([grid, own[grid.columns.intersection(own.columns)]], ignore_index=True)
    grid["status"] = grid["status"].fillna(0)
    grid = grid.sort_values(["team", "player_id", "team_seq"]).reset_index(drop=True)
    g = grid.groupby(["team", "player_id"], sort=False)
    app = (grid["status"] > 0).astype(float)
    st = (grid["status"] == 2).astype(float)
    grid["app_e5"] = app.groupby([grid["team"], grid["player_id"]]).transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    grid["start_e5"] = st.groupby([grid["team"], grid["player_id"]]).transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    grid["start_e15"] = st.groupby([grid["team"], grid["player_id"]]).transform(
        lambda x: x.shift(1).ewm(span=15, min_periods=1).mean())
    grid["last_status"] = g["status"].shift(1)
    grid["recent_app"] = app.groupby([grid["team"], grid["player_id"]]).transform(
        lambda x: x.shift(1).rolling(CANDIDATE_WINDOW, min_periods=1).max())
    pens = grid["pens"].fillna(0.0)
    grid["pens_e38"] = pens.groupby([grid["team"], grid["player_id"]]).transform(
        lambda x: x.shift(1).ewm(span=38, min_periods=1).mean())
    grid = grid[grid["recent_app"] == 1].copy()
    # PLAYER FORM across all his appearances (any team): per-90 rates, entering.
    a = d.sort_values(["player_id", "date", "match_id"]).copy()
    ga = a.groupby("player_id", sort=False)
    for span in (6, 20):
        mins = ga["minutes"].transform(lambda x, k=span: x.shift(1).ewm(span=k, min_periods=1).mean())
        for c in ("shots", "sot", "npxg", "goals", "xA", "key_passes"):
            num = ga[c].transform(lambda x, k=span: x.shift(1).ewm(span=k, min_periods=1).mean())
            a[f"{c}90_e{span}"] = 90.0 * num / mins.clip(lower=10)
        a[f"min_e{span}"] = mins
    a["min_start_e10"] = (a["minutes"].where(a["started"] == 1)
                          .groupby(a["player_id"]).transform(
                              lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()))
    a["min_start_e10"] = a.groupby("player_id")["min_start_e10"].ffill()
    a["apps_before"] = ga.cumcount()
    a["pos_g"] = a["position"].map(_pos_group)
    a["pos_g"] = a["pos_g"].replace("", np.nan)
    a["pos_g"] = a.groupby("player_id")["pos_g"].transform(lambda x: x.ffill().shift(1).ffill())
    form_cols = [c for c in a.columns if "90_e" in c or c.startswith("min_")] + ["apps_before", "pos_g"]
    form = a[["player_id", "date", "match_id"] + form_cols]
    # For a candidate row, his form is from his last appearance BEFORE this match
    # (the row itself when he appeared there: those features are already entering).
    grid = grid.sort_values("date")
    form = form.sort_values("date")
    played_here = grid["status"] > 0
    here = grid[played_here].merge(form.drop(columns=["date"]), on=["player_id", "match_id"], how="left")
    away = grid[~played_here].drop(columns=[c for c in form_cols if c in grid.columns])
    post = a.sort_values("date").copy()
    # Form AFTER his last appearance (that game included) for rows he missed.
    for span in (6, 20):
        mins = post.groupby("player_id")["minutes"].transform(lambda x, k=span: x.ewm(span=k, min_periods=1).mean())
        for c in ("shots", "sot", "npxg", "goals", "xA", "key_passes"):
            num = post.groupby("player_id")[c].transform(lambda x, k=span: x.ewm(span=k, min_periods=1).mean())
            post[f"{c}90_e{span}"] = 90.0 * num / mins.clip(lower=10)
        post[f"min_e{span}"] = mins
    post["min_start_e10"] = (post["minutes"].where(post["started"] == 1).groupby(post["player_id"])
                             .transform(lambda x: x.ewm(span=10, min_periods=1).mean()))
    post["min_start_e10"] = post.groupby("player_id")["min_start_e10"].ffill()
    post["apps_before"] = post.groupby("player_id").cumcount() + 1
    post["pos_g"] = post["position"].map(_pos_group).replace("", np.nan)
    post["pos_g"] = post.groupby("player_id")["pos_g"].ffill()
    away = pd.merge_asof(away.sort_values("date"), post[["player_id", "date"] + form_cols].sort_values("date"),
                         on="date", by="player_id", allow_exact_matches=False)
    grid = pd.concat([here, away], ignore_index=True)
    grid = grid.merge(teams.drop(columns=["team_xg", "opp_xg", "team_pens", "team_shots", "date"]),
                      on=["team", "match_id", "team_seq"], how="left")
    latest_name = d.sort_values("date").groupby("player_id")["player"].last()
    grid["player"] = grid["player_id"].map(latest_name).fillna(grid["player"])
    grid["pen_share"] = (grid["pens_e38"] / grid["team_pens_in"].clip(lower=0.02)).clip(upper=1).fillna(0)
    grid["pos"] = grid["pos_g"].map(POS).astype(float)
    grid["days_since"] = np.nan
    # Outcomes (NaN for upcoming rows): did not appear counts as a loss.
    scored = grid["goals"].fillna(0) > 0
    grid["scored"] = scored.astype(float)
    grid["shots2"] = (grid["shots"].fillna(0) >= 2).astype(float)
    grid["sot1"] = (grid["sot"].fillna(0) >= 1).astype(float)
    grid["role"] = grid["status"].astype(int)
    if upcoming is not None and len(upcoming):
        nxt = grid["match_id"].isin(set(upcoming["match_id"]))
        grid.loc[nxt, ["scored", "shots2", "sot1", "role"]] = np.nan
        grid["is_next"] = nxt
    else:
        grid["is_next"] = False
    return grid.sort_values(["date", "match_id", "team", "player_id"]).reset_index(drop=True)


ROLE_COLS = ["app_e5", "start_e5", "start_e15", "last_status", "apps_before", "pos", "min_e6",
             "min_e20", "min_start_e10", "is_home", "team_xg_in", "opp_xg_in"]
OUT_COLS = (["started", "pos", "is_home", "pen_share", "apps_before", "min_e6", "min_e20",
             "min_start_e10", "team_xg_in", "opp_xg_in", "opp_attack_in", "opp_defence_in",
             "team_shots_in"]
            + [f"{c}90_e{k}" for c in ("shots", "sot", "npxg", "goals", "xA", "key_passes")
               for k in (6, 20)])


def _gbm(classes=2, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=seed)


class PropForecast:
    """Role model + one outcome model per market."""

    def fit(self, train: pd.DataFrame):
        train = train[~train["is_next"]]
        self.role = _gbm().fit(train[ROLE_COLS], train["role"].astype(int))
        played = train[train["role"] > 0].copy()
        played["started"] = (played["role"] == 2).astype(float)
        self.out = {m: _gbm().fit(played[OUT_COLS], played[col].astype(int))
                    for m, col in MARKETS.items()}
        return self

    def roles(self, rows: pd.DataFrame) -> np.ndarray:
        """[P(none), P(sub), P(start)] per row."""
        p = self.role.predict_proba(rows[ROLE_COLS])
        full = np.zeros((len(rows), 3))
        for j, c in enumerate(self.role.classes_):
            full[:, int(c)] = p[:, j]
        return full

    def conditional(self, rows: pd.DataFrame, market: str, started: bool) -> np.ndarray:
        x = rows[OUT_COLS].copy()
        x["started"] = 1.0 if started else 0.0
        return self.out[market].predict_proba(x)[:, 1]

    def predict(self, rows: pd.DataFrame, roles: np.ndarray | None = None) -> dict:
        """{market: P(market | he PLAYS)} plus 'p_start', 'p_sub'. A player who does
        not play is VOID, as bet365 settles, so the chance is conditional on taking
        the field: starting and coming on, weighted by how likely each is. `roles`
        overrides the role model (a confirmed XI)."""
        r = self.roles(rows) if roles is None else roles
        play = np.clip(r[:, 1] + r[:, 2], 1e-6, None)
        out = {"p_start": r[:, 2], "p_sub": r[:, 1], "p_play": r[:, 1] + r[:, 2]}
        for m in MARKETS:
            out[m] = (r[:, 2] * self.conditional(rows, m, True)
                      + r[:, 1] * self.conditional(rows, m, False)) / play
        return out


def appearances(league: str, season: int, base: Path | None = None) -> tuple[set, set]:
    """({(norm player, date)} who APPEARED, {(canonical team, date)} whose lineup we
    hold) for one league-season, from Understat's per-match rosters -- which list
    every player who took the field, shots or not. What decides a VOID: a player
    missing from a lineup we hold did not play."""
    from leagues import names
    from nfl import odds as odds_mod
    f = Path(base or MATCHES_DIR) / f"{league}_{season}.csv.gz"
    if not f.exists():
        return set(), set()
    d = pd.read_csv(f, usecols=["date", "team", "player"])
    day = pd.to_datetime(d["date"].astype(str).str[:10]).dt.date
    appeared = set(zip(d["player"].map(lambda n: odds_mod.norm_name(str(n))), day))
    sides = {(names.canonical(t, league), dd) for t, dd in zip(d["team"], day)}
    return appeared, sides

