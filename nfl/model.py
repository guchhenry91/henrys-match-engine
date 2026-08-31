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
from sklearn.model_selection import KFold
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
WITH_SHARE = CORE + ["share5"]
WITH_BOTH = CORE + ["is_home", "rest_days", "share5"]
WITH_EFF = CORE + ["share5", "eff5"]
WITH_ALL = CORE + ["is_home", "rest_days", "share5", "eff5"]
SCRIPT = ["team_scored5", "team_allowed5", "opp_scored5", "opp_allowed5"]
WITH_SCRIPT = WITH_BOTH + SCRIPT
# SCRIPT is built and available but NOT a candidate. Points scored and allowed
# were a reasonable hypothesis for passing volume -- a team behind throws more --
# and it was tested and rejected: it left passing yards unchanged and pushed
# rushing back out of release. Every extra candidate also costs something even
# when it is never chosen, because the selector picks between more options on the
# same finite evidence and sometimes picks wrong. A feature set has to earn its
# place; this one did not.
CANDIDATES = {"core": CORE, "context": WITH_CONTEXT, "share": WITH_SHARE,
              "both": WITH_BOTH, "eff": WITH_EFF, "all": WITH_ALL}
FEATURES = WITH_ALL + SCRIPT     # superset, for frame_features to build

BLEND = 0.5          # equal parts model and empirical baseline

# Rows required before isotonic calibration is trusted. Isotonic is non-parametric
# and will happily carve a step function out of noise on a small sample, so below
# this the model falls back to Platt scaling -- one parameter, far harder to
# overfit -- and below THAT it ships uncalibrated rather than pretending.
MIN_ROWS_FOR_ISOTONIC = 5000
MIN_ROWS_FOR_PLATT = 300
# The tail of the training window held back to fit the calibrator, as a FRACTION
# of rows in date order rather than a whole number of seasons.
#
# Seasons degenerate at the edge. With a two-season calibration window and a
# two-season training window -- which is exactly the first scored fold -- the
# holdout came out empty, the guard silently fell back to using everything, and
# the calibrator was fitted on the very rows the logistic had just memorised. It
# then corrected a distortion that does not exist on unseen data. A fraction
# cannot degenerate: both parts are always non-empty and always disjoint.
# CROSS-FITTED, not held out. A single held-back tail costs the logistic 30% of
# its training data -- which the small markets cannot afford, rushing yards having
# only ~1,000 rows a season -- and fits the calibrator on one slice of one period.
# Cross-fitting gives every training row an out-of-fold prediction, so the
# calibrator sees the whole window and the logistic is then refitted on all of it.
# Both stages use 100% of the data and neither ever sees its own prediction.
CALIBRATION_FOLDS = 5
MIN_CALIBRATION_ROWS = 200


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
    out["share5"] = frame.get("share5", 0.0)
    out["eff5"] = frame.get("eff5", 0.0)
    for column in SCRIPT:
        out[column] = frame.get(column, 0.0)
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
        self.sets, self.scalers, self.models = [list(CORE)], [], []
        self.fitted = False

    # NO SELECTION. Every candidate set is fitted and their probabilities are
    # AVERAGED.
    #
    # Selecting the best candidate on an inner split was the right instinct and
    # the wrong mechanism. Watching markets flip in and out of release as
    # candidates were added -- receiving yards passing, then failing by 0.0003,
    # while passing yards did the reverse -- made it plain that the selector's
    # own variance had become the dominant error. It was choosing between six
    # options on a few hundred quarterback-games and sometimes choosing wrong,
    # and every extra candidate made that worse rather than better.
    #
    # Averaging removes the choice. It cannot pick the wrong set because it does
    # not pick, it is the standard variance reducer for exactly this situation,
    # and it is decided on principle rather than by trying arrangements until
    # the gate went green.

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        """Blended probability, before calibration."""
        base = empirical_baseline(frame, self.market)
        if not self.fitted:
            return base
        built = frame_features(frame, self.market)
        modelled = np.mean([
            clf.predict_proba(scaler.transform(built[columns].to_numpy()))[:, 1]
            for columns, scaler, clf in zip(self.sets, self.scalers, self.models)
        ], axis=0)
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
        # CHRONOLOGICAL ORDER, by whatever column the sport has for it. The
        # cross-fitted calibrator needs rows in time order; NFL numbers its games
        # by week, the NBA by date. Generalised rather than requiring a fake
        # "week" column, which would have been a misnamed field in the NBA frame
        # for the sake of one sort. NFL frames still carry `week`, so their
        # ordering is byte-identical to before.
        order = ["season", "week"] if "week" in frame.columns else ["season", "game_date"]
        ordered = frame.sort_values(order).reset_index(drop=True)
        built = frame_features(ordered, self.market)
        self.sets = [list(c) for _, c in sorted(CANDIDATES.items())]
        y = ordered["outcome"].to_numpy()
        # A fold with one class in it cannot be fitted and must not be faked --
        # fall back to the baseline alone rather than inventing a decision boundary.
        if len(np.unique(y)) < 2:
            self.fitted = False
            return self

        # Out-of-fold predictions for every training row: each row is scored by a
        # model that never saw it, so the calibrator learns the distortion the
        # model will actually show on unseen data.
        oof = np.full(len(ordered), np.nan)
        if len(ordered) >= MIN_CALIBRATION_ROWS:
            stacked = np.zeros((len(self.sets), len(ordered)))
            covered = np.zeros(len(ordered), dtype=bool)
            for s_i, columns in enumerate(self.sets):
                matrix = built[columns].to_numpy()
                for tr, te in KFold(n_splits=CALIBRATION_FOLDS, shuffle=True,
                                    random_state=20260826).split(matrix):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    scaler = StandardScaler().fit(matrix[tr])
                    clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                        scaler.transform(matrix[tr]), y[tr])
                    stacked[s_i, te] = clf.predict_proba(scaler.transform(matrix[te]))[:, 1]
                    covered[te] = True
            base = empirical_baseline(ordered, self.market)
            oof = np.where(covered, BLEND * stacked.mean(axis=0) + (1 - BLEND) * base,
                           np.nan)

        # Refit each set on EVERYTHING -- the folds existed to produce honest
        # predictions for the calibrator, not to throw data away.
        self.scalers, self.models = [], []
        for columns in self.sets:
            matrix = built[columns].to_numpy()
            scaler = StandardScaler().fit(matrix)
            self.scalers.append(scaler)
            self.models.append(LogisticRegression(max_iter=2000, C=1.0).fit(
                scaler.transform(matrix), y))
        self.fitted = True

        usable = ~np.isnan(oof)
        if usable.sum() < MIN_CALIBRATION_ROWS:
            return self
        raw = oof[usable]
        truth = y[usable]
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
