"""How far to trust the model over the bookmaker -- fitted on the LIVE record.

THE AUDIT THAT FORCED THIS (2026-10-01). Across the first 417 graded picks that
carried both the model's probability and the book's fair one, the book was the
better forecast (log-loss 0.652 vs 0.674), and the more the model disagreed with
the book, the worse its picks did: picks with a 6pt+ "edge" said 65%, the book
said 53%, and 49% landed. Blending as p = book + w * (model - book) and fitting w
by maximum likelihood gave w = 0.00 for MLB props, NFL props, overs and unders
alike (90% bootstrap ranges topping out at 0.2-0.35). The model's disagreements
were, on that record, noise -- and every "value" tag built on them was too.

So every published probability is now pulled toward the book by the weight the
record supports, re-fitted on every run from the picks graded so far:

  w = 0   the board shows the book's own chance; nothing reads as value
  w = 1   the board shows the raw model

The raw model is kept on each pick as `p_model`, so the record keeps measuring it
and w can rise if the corrected models (Phase 3, the 2026-data NFL fix) start
beating the book. A sport with fewer than MIN_PICKS graded uses the pooled weight
of the others; with nothing pooled either, DEFAULT_W.
"""
from __future__ import annotations

import json
from pathlib import Path

from nfl import market_blend

ROOT = Path(__file__).resolve().parent.parent
# ONE-SIDED markets carry the book's RAW price (margin included) as book_p: there is
# no other side to de-vig against. Pulling a probability toward it would publish the
# bookmaker's margin as a probability, and fitting on it would mis-measure the model
# (review, 2026-10-04: NFL anytime-TD 0.6897 was exactly 1/1.45). Left unblended.
ONE_SIDED = {"anytime_touchdown"}
MIN_PICKS = 100
DEFAULT_W = 0.25
LOGS = {"mlb": ROOT / "data-raw" / "mlb" / "picks_log.json",
        "nba": ROOT / "data-raw" / "nba" / "picks_log.json",
        "nfl": ROOT / "data-raw" / "nfl" / "picks_log.json"}


def _entries(sport: str) -> list:
    try:
        raw = json.loads(LOGS[sport].read_text(encoding="utf-8"))
    except Exception:
        return []
    if sport == "nfl":
        out = []
        for section, kind in (("games", "winner"), ("props", "prop")):
            for k, e in (raw.get(section) or {}).items():
                if not k.startswith("_") and isinstance(e, dict):
                    out.append(dict(e, kind=kind))
        return out
    return [e for e in raw.values() if isinstance(e, dict)]


def rows(entries, kind: str) -> list:
    """(raw model, book fair chance, won) for graded picks of `kind` with both."""
    out = []
    for e in entries:
        if e.get("graded") not in ("correct", "wrong") or e.get("kind", "prop") != kind:
            continue
        if e.get("market") in ONE_SIDED:
            continue
        if kind == "prop":
            p, b = e.get("p_model", e.get("probability", e.get("p_pick"))), e.get("book_p")
        else:
            p, b = e.get("p_model", e.get("p_pick")), e.get("book_p_pick")
        if p is None or b is None:
            continue
        out.append((float(p), float(b), 1.0 if e["graded"] == "correct" else 0.0))
    return out


def weights() -> dict:
    """{sport: {"prop": {w, n, basis}, "winner": {...}}} from the live record."""
    data = {s: {k: rows(_entries(s), k) for k in ("prop", "winner")} for s in LOGS}
    out = {}
    for kind in ("prop", "winner"):
        pooled = [r for s in data for r in data[s][kind]]
        pool_fit = market_blend.fit(pooled) if len(pooled) >= MIN_PICKS else None
        for s in LOGS:
            own = data[s][kind]
            if len(own) >= MIN_PICKS:
                f = market_blend.fit(own)
                entry = {"w": f["w"], "n": len(own), "basis": "own record"}
            elif pool_fit:
                entry = {"w": pool_fit["w"], "n": len(pooled), "basis": "other sports pooled"}
            else:
                entry = {"w": DEFAULT_W, "n": len(pooled), "basis": "default (too few graded)"}
            out.setdefault(s, {})[kind] = entry
    return out


def apply_line_board(payload: dict, sport_weights: dict) -> None:
    """Pull every pick toward the book by the fitted weight; keep the raw model."""
    wp = float(sport_weights["prop"]["w"])
    ww = float(sport_weights["winner"]["w"])
    for g in payload.get("games") or []:
        b = g.get("book_p_pick")
        if g.get("p_pick") is None or b is None or g.get("locked"):
            continue
        raw = float(g.get("p_model", g["p_pick"]))
        g["p_model"] = round(raw, 4)
        g["p_pick"] = round(market_blend.blend(raw, float(b), ww), 4)
        g["edge"] = round(g["p_pick"] - float(b), 4)
        g["trust_w"] = ww
    for market, block in (payload.get("props") or {}).items():
        for p in (block or {}).get("picks") or []:
            b = p.get("book_p")
            if p.get("probability") is None or b is None or p.get("locked"):
                continue
            if market in ONE_SIDED or p.get("market") in ONE_SIDED:
                continue
            raw = float(p.get("p_model", p["probability"]))
            p["p_model"] = round(raw, 4)
            p["probability"] = round(market_blend.blend(raw, float(b), wp), 4)
            p["edge"] = round(p["probability"] - float(b), 4)
            p["trust_w"] = wp
    payload["model_trust"] = sport_weights


def apply_game_props(game_props: dict, sport_weights: dict) -> None:
    """The same pull toward the book for the game pages' full lists. Idempotent:
    a pick also on the main board is the same object and re-blends from p_model."""
    for markets in (game_props or {}).values():
        apply_line_board({"props": {m: {"picks": ps} for m, ps in (markets or {}).items()}},
                         sport_weights)


# ---- soccer --------------------------------------------------------------------------
# THE SAME AUDIT, FOR SOCCER (review, 2026-10-04). On the 215 graded match picks,
# bet365's pre-match line was the better forecast (log-loss 0.6591 vs the model's
# 0.6696), and where the model disagreed by 6pt+ it said 46.7%, bet365 34.9%, and
# 37.5% landed. So the soccer VALUE verdict (and its stake) is judged on the model
# pulled toward bet365 by the weight the record supports. Unlike the US boards, the
# board's p_pick is left as the model's own number: the Best Picks tier and its
# backtested hit rate describe the model, and they stay comparable.
SOCCER_DIR = ROOT / "data-raw" / "leagues"
_SOCCER = None


def soccer_entries(kind: str) -> list:
    name = "picks_log.json" if kind == "winner" else "player_picks_log.json"
    out = []
    for path in sorted(SOCCER_DIR.glob(f"*/{name}")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out += [dict(e, kind="winner" if kind == "winner" else "prop")
                for k, e in raw.items() if not str(k).startswith("_") and isinstance(e, dict)]
    return out


def soccer_rows(kind: str) -> list:
    """(model, bet365's fair chance, won) for graded soccer picks carrying both.
    A match pick's fair chance is the one frozen at lock (book_p_pick) or, for picks
    stamped after the fact, bet365's pre-match line (open.p, leagues/closing.py)."""
    out = []
    for e in soccer_entries(kind):
        if e.get("graded") not in ("correct", "wrong") or e.get("void"):
            continue
        p = e.get("p_model", e.get("p_pick"))
        b = e.get("book_p_pick") or (e.get("open") or {}).get("p") if kind == "winner" else e.get("book_p")
        if p is None or b is None:
            continue
        out.append((float(p), float(b), 1.0 if e["graded"] == "correct" else 0.0))
    return out


def soccer_weights() -> dict:
    """{"winner": {w, n, basis}, "prop": {...}}; cached for the process."""
    global _SOCCER
    if _SOCCER is None:
        _SOCCER = {}
        for kind in ("winner", "prop"):
            own = soccer_rows(kind)
            if len(own) >= MIN_PICKS:
                _SOCCER[kind] = {"w": market_blend.fit(own)["w"], "n": len(own), "basis": "own record"}
            else:
                _SOCCER[kind] = {"w": DEFAULT_W, "n": len(own), "basis": "default (too few graded)"}
    return _SOCCER


def soccer_value_p(p, book_p, kind: str):
    """The probability a soccer value verdict is judged on."""
    if p is None or book_p is None:
        return p
    return round(market_blend.blend(float(p), float(book_p), float(soccer_weights()[kind]["w"])), 4)
