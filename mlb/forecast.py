"""The rebuilt MLB prop model: forecast the player's game, never read a line.

docs/REBUILD_PLAN.md. The old MLB props (mlb/features.augment_lines + PropModel) were
trained on lines set from each player's own median with the line as an input. This
model forecasts the WHOLE DISTRIBUTION of the stat for the game ahead -- a
gradient-boosted classifier over the count itself (0, 1, 2, ... hits; 0 ... 12+
strikeouts) -- so the chance of beating ANY line is read straight off it, with no
assumed shape (small counts like hits are under-dispersed; a Poisson or negative
binomial would overstate their tails).

Inputs, all ENTERING (built from earlier games) except what is known before first
pitch -- home/away, the park, and the opposing starter:
  batters  -- per-PA rates of hits, HR, RBI, runs (short and long memory), plate
              appearances per game and usual lineup slot (role), his team's scoring,
              the opposing starter's strikeout rate and depth, the opponent's runs
              allowed, park factors for runs and home runs;
  pitchers -- strikeouts per batter faced, batters faced and outs per start (how deep
              he goes), the opponent lineup's strikeout rate against starters, park;
  teams    -- runs scored and allowed form, the opposing starter, park.
Data: Retrosheet history (mlb.data) + this season's box scores (mlb.current), the
same columns the live board has every day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb import config, current, data
from mlb import features as old_features

# market -> (kind, outcome column, highest count class; the top class is "this or more")
MARKETS = {
    "hits": ("batter", "H", 4),
    "hrr": ("batter", "HRR", 7),
    "hr": ("batter", "HR", 1),
    "rbi": ("batter", "RBI", 1),
    "strikeouts": ("pitcher", "K", 13),
    "team_runs": ("team", "RUNS", 13),
}
TRAIN_SEASONS = 5
MIN_GAMES = {"batter": 10, "pitcher": 3, "team": 5}


def _ewm(frame, by, col, span):
    return frame.groupby(by, sort=False)[col].transform(
        lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())


def load(seasons, with_current: bool = True) -> dict:
    """{"batter", "pitcher", "team"} frames, history + this season's box scores."""
    out = {"batter": data.batter_games(seasons), "pitcher": data.pitcher_games(seasons),
           "team": data.team_games(seasons)}
    if with_current:
        for kind, name in (("batter", "batting"), ("pitcher", "pitching"), ("team", "team")):
            live = current.load(name)
            if live.empty:
                continue
            live = live[[c for c in out[kind].columns if c in live.columns]]
            out[kind] = pd.concat([out[kind], live], ignore_index=True)
    for k in out:
        out[k]["game_date"] = pd.to_datetime(out[k]["game_date"])
        out[k]["is_next"] = False
    return out


def build(frames: dict) -> dict:
    """Entering features for every row of the three kinds. Upcoming games may be
    appended (is_next=True, NaN outcomes) to each kind's frame before calling."""
    b, p, t = (frames[k].copy() for k in ("batter", "pitcher", "team"))
    for f in (b, p, t):
        f["is_next"] = f["is_next"].fillna(False).astype(bool)
        f.sort_values(["game_date", "GAME_ID"], inplace=True)
        f.reset_index(drop=True, inplace=True)
        f["home_team"] = np.where(f["is_home"] >= 0.5, f["TEAM"], f["opponent"])
    # --- starting pitchers: K rate and depth, entering --------------------------------
    gp = p.groupby("PLAYER_ID", sort=False)
    p["games_before"] = gp.cumcount()
    p["rest"] = (p["game_date"] - gp["game_date"].shift(1)).dt.days.clip(upper=30).fillna(30)
    for span in (5, 20):
        p[f"k_e{span}"] = _ewm(p, "PLAYER_ID", "K", span)
        p[f"bfp_e{span}"] = _ewm(p, "PLAYER_ID", "BFP", span)
        p[f"ip_e{span}"] = _ewm(p, "PLAYER_ID", "IPOUTS", span)
        p[f"krate_e{span}"] = p[f"k_e{span}"] / p[f"bfp_e{span}"].clip(lower=1)
    # Each lineup's strikeout rate against STARTERS, entering (from the pitcher rows,
    # the only per-game K source the live box scores carry).
    vs = p[~p["is_next"]].groupby(["GAME_ID", "game_date", "opponent"], as_index=False)[["K", "BFP"]].sum()
    vs = vs.sort_values("game_date")
    for c in ("K", "BFP"):
        vs[f"{c}_e"] = vs.groupby("opponent")[c].transform(lambda x: x.shift(1).ewm(span=40, min_periods=1).mean())
    vs["lineup_krate"] = vs["K_e"] / vs["BFP_e"].clip(lower=1)
    last_lk = vs.groupby("opponent")[["K", "BFP"]].apply(
        lambda g: g["K"].ewm(span=40).mean().iloc[-1] / max(g["BFP"].ewm(span=40).mean().iloc[-1], 1))
    # --- teams: runs scored and allowed, entering ---------------------------------------
    gt = t.groupby("TEAM", sort=False)
    t["games_before"] = gt.cumcount()
    tr = t[~t["is_next"]][["GAME_ID", "TEAM", "RUNS"]].rename(columns={"TEAM": "opponent", "RUNS": "ALLOWED"})
    t = t.merge(tr, on=["GAME_ID", "opponent"], how="left")
    t = t.sort_values(["game_date", "GAME_ID"]).reset_index(drop=True)
    for span in (10, 40):
        t[f"runs_e{span}"] = _ewm(t, "TEAM", "RUNS", span)
        t[f"allowed_e{span}"] = _ewm(t, "TEAM", "ALLOWED", span)
    # Live rows have no team-game row yet -> carry each team's latest form.
    team_form = {}
    for span in (10, 40):
        for c in (f"runs_e{span}", f"allowed_e{span}"):
            src = "RUNS" if c.startswith("runs") else "ALLOWED"
            team_form[c] = t[~t["is_next"]].groupby("TEAM")[src].apply(
                lambda x, k=span: x.ewm(span=k).mean().iloc[-1])
    # --- park factors from PRIOR seasons (mlb.features.park_factors) ---------------------
    hist_t = t[~t["is_next"]]
    pf_runs = old_features.park_factors(hist_t, "RUNS").rename(columns={"park_pf": "park_runs"})
    hist_b = b[~b["is_next"]]
    pf_hr = old_features.park_factors(hist_b, "HR").rename(columns={"park_pf": "park_hr"})

    # Starter of each side in each game (actual for history; probable for upcoming rows,
    # which carry it in OPP_STARTER / STARTER columns set by the caller).
    starters = p[["GAME_ID", "TEAM", "PLAYER_ID"]].rename(columns={"PLAYER_ID": "starter"})
    sp_cols = ["krate_e5", "krate_e20", "bfp_e20", "ip_e5", "ip_e20", "games_before"]
    sp_feat = p[["GAME_ID", "PLAYER_ID"] + sp_cols].rename(
        columns={c: f"opp_sp_{c}" for c in sp_cols} | {"PLAYER_ID": "opp_starter"})

    def attach_context(f, own_starter_col=None):
        f = f.merge(pf_runs, on=["home_team", "season"], how="left").merge(
            pf_hr, on=["home_team", "season"], how="left")
        f[["park_runs", "park_hr"]] = f[["park_runs", "park_hr"]].fillna(1.0)
        # opposing starter: actual (history) or probable (upcoming, column OPP_STARTER)
        opp = starters.rename(columns={"TEAM": "opponent", "starter": "opp_starter"})
        f = f.merge(opp, on=["GAME_ID", "opponent"], how="left")
        if "OPP_STARTER" in f:
            f["opp_starter"] = f["opp_starter"].fillna(f["OPP_STARTER"])
        f = f.merge(sp_feat, on=["GAME_ID", "opp_starter"], how="left")
        # an upcoming starter has no row in this game: use his latest entering-after form
        miss = f["opp_sp_krate_e20"].isna() & f["opp_starter"].notna()
        if miss.any():
            latest = p[~p["is_next"]].sort_values("game_date").groupby("PLAYER_ID")
            post = pd.DataFrame({
                "opp_sp_krate_e5": latest.apply(lambda g: g["K"].ewm(span=5).mean().iloc[-1] / max(g["BFP"].ewm(span=5).mean().iloc[-1], 1)),
                "opp_sp_krate_e20": latest.apply(lambda g: g["K"].ewm(span=20).mean().iloc[-1] / max(g["BFP"].ewm(span=20).mean().iloc[-1], 1)),
                "opp_sp_bfp_e20": latest["BFP"].apply(lambda x: x.ewm(span=20).mean().iloc[-1]),
                "opp_sp_ip_e5": latest["IPOUTS"].apply(lambda x: x.ewm(span=5).mean().iloc[-1]),
                "opp_sp_ip_e20": latest["IPOUTS"].apply(lambda x: x.ewm(span=20).mean().iloc[-1]),
                "opp_sp_games_before": latest.size()})
            for c in post.columns:
                f.loc[miss, c] = f.loc[miss, "opp_starter"].map(post[c])
        # team and opponent run form (entering), carried for upcoming rows
        tf = t[["GAME_ID", "TEAM"] + [c for c in t.columns if c.startswith(("runs_e", "allowed_e"))]]
        f = f.merge(tf, on=["GAME_ID", "TEAM"], how="left", suffixes=("", "_t"))
        of = tf.rename(columns={"TEAM": "opponent"} | {c: f"opp_{c}" for c in tf.columns if c.startswith(("runs_e", "allowed_e"))})
        f = f.merge(of, on=["GAME_ID", "opponent"], how="left")
        for c, last in team_form.items():
            if c not in f:
                continue
            m = f[c].isna()
            f.loc[m, c] = f.loc[m, "TEAM"].map(last)
            m = f[f"opp_{c}"].isna()
            f.loc[m, f"opp_{c}"] = f.loc[m, "opponent"].map(last)
        # the opponent lineup's K rate (for pitchers) / own lineup's (for batters)
        lk = vs[["GAME_ID", "opponent", "lineup_krate"]]
        return f, lk

    # --- batters -------------------------------------------------------------------------
    gb = b.groupby("PLAYER_ID", sort=False)
    b["games_before"] = gb.cumcount()
    for span in (15, 60):
        b[f"pa_e{span}"] = _ewm(b, "PLAYER_ID", "PA", span)
        for c in ("H", "HR", "RBI", "R"):
            b[f"{c.lower()}_rate_e{span}"] = _ewm(b, "PLAYER_ID", c, span) / b[f"pa_e{span}"].clip(lower=0.5)
    b["slot_e10"] = _ewm(b, "PLAYER_ID", "lineup", 10)
    b, _ = attach_context(b)
    # --- pitchers ------------------------------------------------------------------------
    p2, lk = attach_context(p)
    p2 = p2.merge(lk.rename(columns={"opponent": "opponent"}), on=["GAME_ID", "opponent"], how="left")
    m = p2["lineup_krate"].isna()
    p2.loc[m, "lineup_krate"] = p2.loc[m, "opponent"].map(last_lk)
    # --- teams ---------------------------------------------------------------------------
    t2, _ = attach_context(t)
    return {"batter": b, "pitcher": p2, "team": t2}


CONTEXT = ["is_home", "park_runs", "park_hr", "opp_sp_krate_e5", "opp_sp_krate_e20",
           "opp_sp_bfp_e20", "opp_sp_ip_e5", "opp_sp_ip_e20", "opp_sp_games_before",
           "runs_e10", "runs_e40", "allowed_e10", "allowed_e40",
           "opp_runs_e10", "opp_runs_e40", "opp_allowed_e10", "opp_allowed_e40"]
FEATURES = {
    "batter": CONTEXT + ["games_before", "slot_e10"]
    + [f"pa_e{s}" for s in (15, 60)]
    + [f"{c}_rate_e{s}" for c in ("h", "hr", "rbi", "r") for s in (15, 60)],
    "pitcher": ["is_home", "park_runs", "games_before", "rest", "lineup_krate",
                "opp_runs_e10", "opp_runs_e40"]
    + [f"{c}_e{s}" for c in ("k", "bfp", "ip", "krate") for s in (5, 20)],
    "team": CONTEXT + ["games_before"],
}


def eligible(frame: pd.DataFrame, kind: str) -> pd.Series:
    return frame["games_before"] >= MIN_GAMES[kind]


class CountModel:
    """P(count = k) for k = 0 .. top (top = "top or more"), one gradient-boosted
    multiclass classifier; P(over line) is the sum of the classes above it."""

    def __init__(self, market: str, seed: int = 0):
        self.market = market
        self.kind, self.stat, self.top = MARKETS[market]
        self.cols = FEATURES[self.kind]
        self.seed = seed
        self.model = None

    def fit(self, train: pd.DataFrame):
        from sklearn.ensemble import HistGradientBoostingClassifier
        train = train[eligible(train, self.kind) & ~train["is_next"] & train[self.stat].notna()]
        y = train[self.stat].clip(upper=self.top).astype(int)
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, random_state=self.seed).fit(train[self.cols], y)
        return self

    def pmf(self, rows: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(rows[self.cols])
        out = np.zeros((len(rows), self.top + 1))
        for j, c in enumerate(self.model.classes_):
            out[:, int(c)] = raw[:, j]
        return out

    def p_over(self, rows: pd.DataFrame, lines) -> np.ndarray:
        pmf = self.pmf(rows)
        lines = np.broadcast_to(np.asarray(lines, float), (len(rows),))
        ks = np.arange(self.top + 1)
        return np.clip((pmf * (ks[None, :] > lines[:, None])).sum(axis=1), 1e-4, 1 - 1e-4)

    def mean(self, rows: pd.DataFrame) -> np.ndarray:
        return self.pmf(rows) @ np.arange(self.top + 1)
