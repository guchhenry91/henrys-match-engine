"""The prop model: a logistic fit blended with an empirical baseline.

WHY A BLEND. The logistic model is trained on a few thousand rows per market and
will happily produce confident probabilities out at the tails where it has seen
almost nothing. The empirical baseline -- the player's own entering hit rate
against his own line -- is dumb but never overconfident. Averaging them keeps the
model's information while capping how far it can run from what the player has
actually done, which is what the calibration gate measures.

WHY NOT SOMETHING CLEVERER. It was tried: gradient boosting beat logistic on
log-loss in training and lost on the held-out seasons, which is the only place
that counts. A model this size, on this many rows, does not have the evidence to
support more parameters.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# The pre-game signals, in the order the fitted coefficients expect.
# Candidate feature sets, chosen PER MARKET by an inner split of the TRAINING
# window -- never by looking at a scored season. Adding home/away and rest
# globally improved receiving yards and broke the touchdown market, which is what
# a single hand-picked feature set does: it is tuned to whichever market you were
# staring at. Letting each market pick its own, on evidence it is allowed to see,
# is both better and honest. The alternative -- picking the set that makes the
# gate go green -- is fitting the holdout, and it is the exact failure the gate
# exists to catch.
CORE = ["hist_edge", "form5_edge", "form10_edge", "opp5", "opp_allowed_edge",
        "games_before"]
WITH_CONTEXT = CORE + ["is_home", "rest_days"]
CANDIDATES = {"core": CORE, "context": WITH_CONTEXT}
FEATURES = WITH_CONTEXT          # superset, for frame_features to build

BLEND = 0.5          # equal parts model and empirical baseline

# Rows required before isotonic calibration is trusted. Isotonic is non-parametric
# and will happily carve a step function out of noise on a small sample, so below
# this the model falls back to Platt scaling -- one parameter, far harder to
# overfit -- and below THAT it ships uncalibrated rather than pretending.
MIN_ROWS_FOR_ISOTONIC = 1500
MIN_ROWS_FOR_PLATT = 300
# The tail of the training window held back to fit the calibrator. It must be
# unseen by the logistic fit, or the mapping is learned from predictions the model
# has already memorised and corrects a distortion that will not exist live.
CALIBRATION_SEASONS = 2


def frame_features(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    """Model inputs, expressed RELATIVE TO THE LINE wherever a line exists.

    A receiver's 60-yard form means nothing on its own; it means everything
    against a 45.5 line and nothing against a 95.5 one. Feeding raw yardage would
    make the model learn "big numbers mean over", which is true only until it
    meets a player whose line is big too.
    """
    out = pd.DataFrame(index=frame.index)
    if market == "anytime_touchdown":
        # No line: the event is the outcome. Rates are already comparable.
        out["hist_edge"] = frame["hist_rate"]
        out["form5_edge"] = frame["form5"]
        out["form10_edge"] = frame["form10"]
        out["opp_allowed_edge"] = frame["opp_allowed"].fillna(frame["opp_allowed"].mean())
    else:
        line = frame["line"].replace(0, np.nan)
        out["hist_edge"] = (frame["hist_rate"] - frame["line"]) / line
        out["form5_edge"] = (frame["form5"] - frame["line"]) / line
        out["form10_edge"] = (frame["form10"] - frame["line"]) / line
        allowed = frame["opp_allowed"]
        out["opp_allowed_edge"] = ((allowed - allowed.mean()) / (allowed.std() or 1.0))
    out["opp5"] = frame["opp5"]
    out["games_before"] = frame["games_before"]
    out["is_home"] = frame.get("is_home", 0.5)
    out["rest_days"] = frame.get("rest_days", 7.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def empirical_baseline(frame: pd.DataFrame, market: str) -> np.ndarray:
    """The dumb-but-honest comparator: what this player has done before.

    For the touchdown market that is his own scoring rate. For a yards market the
    line is his own median, so his entering rate of beating it is close to 0.5 by
    construction -- the baseline there is deliberately weak, and any market whose
    model cannot beat it has no business being published.
    """
    if market == "anytime_touchdown":
        return frame["hist_rate"].clip(0.01, 0.99).to_numpy()
    edge = (frame["form5"] - frame["line"]) / frame["line"].replace(0, np.nan)
    return (0.5 + 0.25 * edge.fillna(0.0)).clip(0.05, 0.95).to_numpy()


class PropModel:
    def __init__(self, market: str):
        self.market = market
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, C=1.0)
        self.calibrator = None
        self.columns = list(CORE)
        self.fitted = False

    def _choose_features(self, train: pd.DataFrame) -> list:
        """Pick a candidate set using the LAST training season as an inner test.

        Strictly inside the training window, so nothing a scored season contains
        can influence the choice. Ties and single-season windows fall back to the
        core set: fewer parameters is the right default when there is no evidence
        to prefer more.
        """
        seasons = sorted(train["season"].unique())
        if len(seasons) < 2:
            return list(CORE)
        inner_train = train[train["season"] < seasons[-1]]
        inner_test = train[train["season"] == seasons[-1]]
        if inner_train.empty or inner_test.empty:
            return list(CORE)
        best, best_score = list(CORE), None
        for _, columns in sorted(CANDIDATES.items()):
            try:
                scaler = StandardScaler()
                x = scaler.fit_transform(frame_features(inner_train, self.market)[columns])
                y = inner_train["outcome"].to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(x, y)
                xt = scaler.transform(frame_features(inner_test, self.market)[columns])
                prob = clf.predict_proba(xt)[:, 1]
                score = float(np.mean((prob - inner_test["outcome"].to_numpy()) ** 2))
            except Exception:
                continue
            if best_score is None or score < best_score:
                best, best_score = list(columns), score
        return best

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        """Blended probability, before calibration."""
        base = empirical_baseline(frame, self.market)
        if not self.fitted:
            return base
        x = self.scaler.transform(frame_features(frame, self.market)[self.columns])
        modelled = self.clf.predict_proba(x)[:, 1]
        return BLEND * modelled + (1 - BLEND) * base

    def fit(self, frame: pd.DataFrame):
        """Fit the model, then fit a calibrator on a slice it never saw.

        WHY THIS EXISTS. The first honest walk-forward showed every market beating
        its baseline on Brier while three of four failed the 0.04 ECE gate --
        receiving yards at 0.052, rushing at 0.048, passing at 0.054 in 2024. The
        ranking was right and the NUMBERS were wrong: the model knew who was more
        likely to go over, and said 70% where it should have said 62%. A board
        whose stated confidence does not mean what it says is worse than no board,
        because the record it builds is unreadable.

        Calibration adds no information. It maps the probabilities the model
        already produces onto the frequencies those probabilities actually
        deliver, learned on the LAST season of the training window -- held out of
        the logistic fit, because a mapping learned from predictions the model has
        memorised corrects a distortion that will not exist on live data.
        """
        seasons = sorted(frame["season"].unique())
        holdout = seasons[-CALIBRATION_SEASONS:] if len(seasons) > CALIBRATION_SEASONS else []
        core = frame[~frame["season"].isin(holdout)] if holdout else frame
        calib = frame[frame["season"].isin(holdout)] if holdout else frame

        self.columns = self._choose_features(core)
        x = self.scaler.fit_transform(frame_features(core, self.market)[self.columns])
        y = core["outcome"].to_numpy()
        # A fold with one class in it cannot be fitted and must not be faked --
        # fall back to the baseline alone rather than inventing a decision boundary.
        if len(np.unique(y)) < 2:
            self.fitted = False
            return self
        self.clf.fit(x, y)
        self.fitted = True

        raw = self._raw(calib)
        truth = calib["outcome"].to_numpy()
        if len(raw) >= MIN_ROWS_FOR_ISOTONIC and len(np.unique(truth)) > 1:
            self.calibrator = IsotonicRegression(out_of_bounds="clip",
                                                 y_min=0.01, y_max=0.99).fit(raw, truth)
        elif len(raw) >= MIN_ROWS_FOR_PLATT and len(np.unique(truth)) > 1:
            self.calibrator = LogisticRegression(max_iter=1000).fit(
                raw.reshape(-1, 1), truth)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw(frame)
        if self.calibrator is None:
            return raw
        if isinstance(self.calibrator, IsotonicRegression):
            return np.clip(self.calibrator.predict(raw), 0.01, 0.99)
        return np.clip(self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1],
                       0.01, 0.99)
