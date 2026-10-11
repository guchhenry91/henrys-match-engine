"""Learn, from REAL lines, how much to trust the forecast against the bookmaker.

docs/REBUILD_PLAN.md. A rebuilt board's final chance is
    sigmoid(a * logit(forecast) + b * logit(book) + c)
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
        log[key] = {"game_id": str(p["game_id"]), "player_id": int(p["player_id"]),
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


def refit(log_path: Path, weights_path: Path, current: dict, now: datetime,
          min_n: int = MIN_N) -> dict:
    """Re-fit each market with min_n graded lines; adopt only on better OOF log-loss.
    `current` is {market: (a, b, c)} in force now. Returns the weights file content."""
    from sklearn.linear_model import LogisticRegression
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
        x = np.column_stack([_logit([e["p_model"] for e in es]), _logit([e["p_book"] for e in es])])
        y = np.array([e["y"] for e in es], float)
        folds = np.arange(len(y)) % 5
        oof = np.zeros(len(y))
        for k in range(5):
            tr, te = folds != k, folds == k
            lr = LogisticRegression(C=1.0).fit(x[tr], y[tr])
            oof[te] = lr.predict_proba(x[te])[:, 1]
        cur = current.get(market) or (0.3, 0.7, 0.0)
        old_ll, new_ll = _logloss(_apply(cur, x), y), _logloss(oof, y)
        lr = LogisticRegression(C=1.0).fit(x, y)
        fitted = (round(float(lr.coef_[0][0]), 4), round(float(lr.coef_[0][1]), 4),
                  round(float(lr.intercept_[0]), 4))
        adopt = new_ll < old_ll
        entry = {"n": len(es), "fitted_at": now.isoformat(timespec="seconds"),
                 "candidate": list(fitted), "logloss_current": round(old_ll, 5),
                 "logloss_candidate_oof": round(new_ll, 5), "adopted": adopt}
        if adopt:
            entry["w"] = list(fitted)
        elif (out.get(market) or {}).get("w"):
            entry["w"] = out[market]["w"]           # keep previously adopted weights
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
