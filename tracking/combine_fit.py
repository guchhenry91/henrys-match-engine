"""Learn, from REAL lines, how much to trust the forecast against the bookmaker.

docs/REBUILD_PLAN.md. A rebuilt board's final chance is
    sigmoid(a * logit(forecast) + b * logit(book) + c)
(by default b = 1 - a and c = 0: the book moved part of the way to the forecast),
and its first weights come from the gate's SIMULATED informed book. This module
replaces them with weights fitted on what actually happened:

  * every priced line (not only the published picks -- fitting on the published
    ones alone would learn only from the cases the old weights already liked) is
    logged with the forecast's and the book's chance of the OVER, the last quote
    seen before tip-off kept;
  * once graded, a stat with at least MIN_N results is re-fitted, and the new
    weights are ADOPTED only if they beat the current ones on 5-fold out-of-fold
    log-loss. Weights that merely fit noise never replace working ones.

Pure numpy + sklearn; one log and one weights file per sport.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MIN_N = 300
# TRUST THE BOOK BY DEFAULT (four-reviewer audit, 2026-10-12). Until a market's weights
# are re-fitted on REAL graded lines and adopted, the book's chance counts in full
# (b = 1), the forecast may only nudge it (a <= A_CAP), and no constant is applied --
# the gates' simulated books were weaker than real ones, so their weights trusted the
# forecast far more than any real line has earned.
A_CAP = 0.2
# SHRINK, DON'T STACK (third review, 2026-10-12). The prior used to be
# sigmoid(a*logit(forecast) + 1*logit(book)): the forecast's confidence ADDED on top of
# the book's full confidence, so a forecast that merely agreed with a 72% book read
# 75.6% -- every favourite looked like value at bet365 with no disagreement at all.
# Weights are now (a, 1 - a, c): the book's logit moved a fraction a of the way toward
# the forecast's. A forecast equal to the book leaves the book's chance unchanged.


def prior(w) -> tuple:
    """The default weights for a market not yet proven on real lines."""
    a = round(min(max(float(w[0]) if w else 0.0, 0.0), A_CAP), 4)
    return (a, round(1.0 - a, 4), 0.0)


def shrink(p_model: float, p_book: float, a: float = A_CAP) -> float:
    """The book's chance moved a fraction `a` (in logit) toward the forecast's."""
    lm, lb = float(_logit(p_model)), float(_logit(p_book))
    return float(1 / (1 + np.exp(-(lb + a * (lm - lb)))))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _load(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _over(pick: dict):
    """(forecast P(over), book P(over)) from a published-shape pick of either side."""
    pm, pb = pick.get("p_model"), pick.get("book_p")
    if pm is None or pb is None:
        return None
    if pick.get("side") == "under":
        return 1 - float(pm), 1 - float(pb)
    return float(pm), float(pb)


def record(path: Path, priced: list, now: datetime) -> int:
    """Log every priced line; the last quote before tip-off wins. Returns rows kept."""
    log = _load(path)
    n = 0
    for p in priced:
        over = _over(p)
        if over is None or p.get("tipoff") is None:
            continue
        key = f"{p['game_id']}:{p['player_id']}:{p['market']}"
        old = log.get(key) or {}
        if old.get("y") is not None:
            continue                                  # already graded
        tip = datetime.fromisoformat(str(p["tipoff"]).replace("Z", "+00:00"))
        if tip.tzinfo is None:
            tip = tip.replace(tzinfo=timezone.utc)
        if now >= tip:
            continue                                  # never a quote from after tip-off
        pid = p["player_id"]
        # NBA/MLB ids are numbers; NFL's are gsis strings ("00-0039875").
        pid = int(pid) if str(pid).lstrip("-").isdigit() else str(pid)
        log[key] = {"game_id": str(p["game_id"]), "player_id": pid,
                    "market": p["market"], "line": float(p["line"]), "tipoff": str(p["tipoff"]),
                    "p_model": round(over[0], 5), "p_book": round(over[1], 5),
                    "seen_at": now.isoformat(timespec="seconds"), "y": None}
        n += 1
    _save(path, log)
    return n


def grade(path: Path, actual) -> int:
    """`actual(entry)` -> the stat, or None while unknown. Exact hits are pushes."""
    log = _load(path)
    n = 0
    for e in log.values():
        if e.get("y") is not None:
            continue
        a = actual(e)
        if a is None:
            continue
        if a == "void":                               # did not play: never a bet
            e["y"] = "void"
            continue
        if float(a) == e["line"]:
            e["y"] = "push"
        else:
            e["y"] = 1 if float(a) > e["line"] else 0
            n += 1
    _save(path, log)
    return n


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _apply(w, x):
    a, b, c = w
    return 1 / (1 + np.exp(-(a * x[:, 0] + b * x[:, 1] + c)))


PUBLISH_BAR = 0.62
# would-publish picks needed before the tail can be judged: at n=30 one standard
# error is ~8.5 points, at 100 ~4.6 (third review, 2026-10-12)
TAIL_MIN = 100
TAIL_SLACK = 0.02       # landed may trail stated by at most this
BOOTSTRAPS = 400
# Re-test only after this many NEW graded lines since the last fit: re-testing every run
# on a growing log is repeated peeking at the same data (third review, 2026-10-12).
REFIT_STEP = 50
BOOK_ALONE = (0.0, 1.0, 0.0)


def _folds(es) -> np.ndarray:
    """Five folds by GAME, so lines from one game never sit on both sides of a split."""
    import zlib
    return np.array([zlib.crc32(str(e.get("game_id")).encode()) % 5 for e in es])


def _significant_gain(p_old, p_new, y, groups, seed=0) -> tuple:
    """(mean log-loss gain, lower 95% bound) of the candidate over the weights in force,
    bootstrapped over GAMES -- a gain inside the noise is not a gain."""
    eps = 1e-6
    ll = lambda p: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    d = ll(p_old) - ll(p_new)
    keys, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(0, len(keys), len(keys))
        boots.append(sums[pick].sum() / max(counts[pick].sum(), 1))
    return float(d.mean()), float(np.percentile(boots, 2.5))


def _tail(p, y, pm=None, pb=None) -> dict:
    """What the PUBLISHED picks would have done. With the forecast's and the book's
    chances (pm, pb) that is what tracking/pricecheck would let through -- the side at
    PUBLISH_BAR or more, +MIN_EV at bet365's estimated price, the forecast within
    MAX_GAP of the book; scoring everything over the bar instead would judge picks the
    board never shows (third review). Without them, the side at PUBLISH_BAR or more."""
    side = np.where(p >= 0.5, p, 1 - p)
    won = np.where(p >= 0.5, y, 1 - y)
    m = side >= PUBLISH_BAR
    if pm is not None and pb is not None:
        from tracking import pricecheck
        pm, pb = np.asarray(pm, float), np.asarray(pb, float)
        fair = np.where(p >= 0.5, pb, 1 - pb)
        mod = np.where(p >= 0.5, pm, 1 - pm)
        ev = side / (np.clip(fair, 1e-6, 1) * (1 + pricecheck.PROP_MARGIN)) - 1
        m &= (ev >= pricecheck.MIN_EV) & (np.abs(mod - fair) <= pricecheck.MAX_GAP) & (mod >= 0.5)
    return {"n": int(m.sum()), "said": round(float(side[m].mean()), 4) if m.any() else None,
            "landed": round(float(won[m].mean()), 4) if m.any() else None}


def refit(log_path: Path, weights_path: Path, current: dict, now: datetime,
          min_n: int = MIN_N, offset: bool = False) -> dict:
    """Re-fit each market with min_n graded lines. ADOPT ONLY IF (audit 2026-10-12):
      * the out-of-fold gain in log-loss over the weights IN FORCE is real -- its 95%
        lower bound, bootstrapped by game, is above zero;
      * the candidate's would-publish tail (>= PUBLISH_BAR) landed at least its stated
        rate less TAIL_SLACK, on TAIL_MIN+ picks;
    with folds grouped by game. `current` is {market: (a, b, c)} actually in force.
    `offset` holds the book's weight at 1 (fit_offset): True, or a set of markets.
    Adoption changes weights only -- never a board's publishing rule."""
    from sklearn.linear_model import LogisticRegression

    def fit(x, y, market=None):
        if offset is True or (offset and market in offset):
            ex = lambda v: 1 / (1 + np.exp(-v))
            return fit_offset(ex(x[:, 0]), ex(x[:, 1]), y)
        lr = LogisticRegression(C=1.0).fit(x, y)
        return (round(float(lr.coef_[0][0]), 4), round(float(lr.coef_[0][1]), 4),
                round(float(lr.intercept_[0]), 4))
    log = _load(log_path)
    out = _load(weights_path)
    by_market = {}
    for e in log.values():
        if e.get("y") in (0, 1):
            by_market.setdefault(e["market"], []).append(e)
    for market, es in by_market.items():
        if len(es) < min_n:
            out.setdefault("_waiting", {})[market] = len(es)
            continue
        last = out.get(market) or {}
        if last.get("n") and len(es) < int(last["n"]) + REFIT_STEP:
            continue                                  # too few new lines to look again
        x = np.column_stack([_logit([e["p_model"] for e in es]), _logit([e["p_book"] for e in es])])
        y = np.array([e["y"] for e in es], float)
        folds = _folds(es)
        oof = np.zeros(len(y))
        for k in range(5):
            tr, te = folds != k, folds == k
            if not te.any() or tr.sum() < 20:
                oof[te] = _apply(current.get(market) or (0.0, 1.0, 0.0), x[te])
                continue
            oof[te] = _apply(fit(x[tr], y[tr], market), x[te])
        cur = current.get(market) or (0.0, 1.0, 0.0)
        p_cur = _apply(cur, x)
        gain, lower = _significant_gain(p_cur, oof, y, np.array([str(e.get("game_id")) for e in es]))
        pm_, pb_ = 1 / (1 + np.exp(-x[:, 0])), 1 / (1 + np.exp(-x[:, 1]))
        tail = _tail(oof, y, pm_, pb_)
        # No would-publish picks: nothing on the board can lose, so the log-loss gain
        # alone decides. Some, but too few to judge: wait. Enough: they must land.
        tail_ok = tail["n"] == 0 or (tail["n"] >= TAIL_MIN and tail["landed"] is not None
                                     and tail["landed"] >= tail["said"] - TAIL_SLACK)
        fitted = fit(x, y, market)
        adopt = lower > 0 and tail_ok
        entry = {"n": len(es), "fitted_at": now.isoformat(timespec="seconds"),
                 "candidate": list(fitted), "in_force": list(cur),
                 "logloss_current": round(_logloss(p_cur, y), 5),
                 "logloss_candidate_oof": round(_logloss(oof, y), 5),
                 "gain": round(gain, 5), "gain_lower95": round(lower, 5),
                 "tail_candidate_oof": tail, "tail_current": _tail(p_cur, y, pm_, pb_),
                 "adopted": adopt,
                 "why": ("adopted" if adopt else
                         "gain inside the noise" if lower <= 0 else
                         f"too few would-publish picks to judge ({tail['n']} of {TAIL_MIN})"
                         if tail["n"] < TAIL_MIN else
                         "would-publish picks landed below their stated rate")}
        if adopt:
            entry["w"] = list(fitted)
        elif (out.get(market) or {}).get("w"):
            # REVOCATION (third review): adopted weights are re-tested each time against
            # the book alone; if the book is now significantly better, they are dropped
            # and the market returns to the default nudge.
            kept = out[market]["w"]
            g_b, lo_b = _significant_gain(_apply(kept, x), _apply(BOOK_ALONE, x), y,
                                          np.array([str(e.get("game_id")) for e in es]))
            if lo_b > 0:
                entry["revoked"] = {"w": kept, "book_gain": round(g_b, 5), "gain_lower95": round(lo_b, 5)}
                entry["why"] = "adopted weights revoked: the book alone is now better"
            else:
                entry["w"] = kept                   # keep previously adopted weights
        out[market] = entry
        (out.get("_waiting") or {}).pop(market, None)
    out["_note"] = ("Forecast-vs-book combination weights fitted on REAL graded lines "
                    "(tracking/combine_fit.py). Adopted only when better out-of-fold.")
    _save(weights_path, out)
    return out


def adopted(weights_path: Path) -> dict:
    """{market: (a, b, c)} for markets with adopted weights."""
    return {m: tuple(v["w"]) for m, v in _load(weights_path).items()
            if isinstance(v, dict) and v.get("w")}


def fit_offset(p_model, p_book, y, l2: float = 1.0):
    """(a, 1 - a, c): the book's logit taken AS GIVEN, learning only how far toward the
    forecast to move it -- z = logit(book) + a*(logit(forecast) - logit(book)) + c.
    Used where the book's chance barely varies (lines set at the median sit near
    50/50): a free book weight then has nothing to fit and explodes. A mild L2 on a
    and c keeps a thin sample from overreaching, and 0 <= a <= 1: a forecast that only
    ever pulls AGAINST its own reading is given no say rather than a reversed one."""
    from scipy.optimize import minimize
    off = _logit(p_book)
    xm = _logit(p_model) - off
    y = np.asarray(y, float)

    def loss(w):
        z = w[0] * xm + off + w[1]
        p = 1 / (1 + np.exp(-z))
        ll = -np.sum(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        return ll + l2 * (w[0] ** 2 + w[1] ** 2)
    w = minimize(loss, np.zeros(2), method="L-BFGS-B", bounds=[(0.0, 1.0), (-3.0, 3.0)]).x
    a = round(float(w[0]), 4)
    return a, round(1.0 - a, 4), round(float(w[1]), 4)


def learning(log_path: Path, weights_path: Path) -> dict:
    """{market: {graded, adopted, why}} for the board: how far each market is toward
    earning more than the default nudge, from EVERY priced line (third review)."""
    log, w = _load(log_path), _load(weights_path)
    out = {}
    for e in log.values():
        if isinstance(e, dict) and e.get("market"):
            m = out.setdefault(e["market"], {"graded": 0, "adopted": False, "why": None})
            m["graded"] += e.get("y") in (0, 1)
    for m, v in w.items():
        if isinstance(v, dict) and m in out:
            out[m]["adopted"] = bool(v.get("w"))
            out[m]["why"] = None if v.get("w") else v.get("why")
    return out
