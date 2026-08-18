"""bet365's 6 Scores Challenge: the model's six Premier League scorelines.

WHAT THIS IS HONEST ABOUT. Correct score is the hardest common football market,
and in the Premier League this engine does not beat the trivial strategy. On the
held-out season it hit 11.84% per fixture against 12.37% for writing 1-1 on every
line, because PL's fitted Dixon-Coles rho of -0.105 lifts the 1-1 cell enough to
make it the likeliest score on 82% of fixtures (scripts/correct_score_check.py,
and scripts/correct_score_fix.py for why better selection cannot rescue it). So
this board publishes its own expected hit rate next to the picks. It exists to be
measured, not to imply a jackpot.

THE CARD SHOWS TWO SCORES, not one. The model's own top score carries only
10-14% in the Premier League, so printing it alone reads as a confident call
when the model is saying the opposite: with lambdas around 1.3 goals a side, one
goal each is simply the least unlikely of forty-odd outcomes. Showing the top two
together makes the flatness visible instead of hiding it behind a single number.

The scoreline is GRID MODE -- the likeliest score across all outcomes -- not the
pick-conditional one. Grid mode is measurably better at actually hitting the
score (11.84% vs 9.74% in PL), which is the only thing that scores in this game.
It will therefore sometimes disagree with the match-winner pick, and the board
says so rather than quietly showing two different numbers.
"""
import json
from pathlib import Path

import pandas as pd

from leagues import picks

ROOT = Path(__file__).resolve().parent.parent
SELECTION = ROOT / "data-raw" / "leagues" / "six_scores.json"

# Measured on the held-out season, quoted so the page never overstates itself.
PL_HIT_RATE_PCT = 11.84
PL_BASELINE_PCT = 12.37
EXPECTED_OF_SIX = round(6 * PL_HIT_RATE_PCT / 100.0, 2)


def load_selection() -> dict:
    if not SELECTION.exists():
        return {"fixtures": [], "matchweek": None}
    try:
        return json.loads(SELECTION.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: could not read six_scores.json ({exc}); no board this run")
        return {"fixtures": [], "matchweek": None}


def _entry(m: dict) -> dict | None:
    """One fixture's scoreline card, from the PL payload's own published grid."""
    pred = m.get("prediction") or {}
    tops = pred.get("top_scores") or []
    if not tops:
        return None
    best = tops[0]
    return {
        "id": m["id"], "date": m["date"], "home": m["home"], "away": m["away"],
        "score": best["score"],
        "score_pct": best["pct"],
        # The SECOND likeliest score, shown next to the first rather than hidden.
        # A single scoreline implies a confidence the model never claimed: its top
        # score sits at 10-14%, so the honest reading is "these two are the least
        # unlikely", not "this is the result". Two, not three -- enough to show the
        # field is flat without turning the card into a table.
        "alternatives": [{"score": t["score"], "pct": t["pct"]} for t in tops[1:2]],
        "match_pick": pred.get("pick"),
        # Grid mode answers "what is the likeliest score", the match pick answers
        # "who wins". They disagree often and legitimately; flag it rather than
        # letting the card look self-contradictory.
        "agrees_with_match_pick": bool(best.get("agrees_with_pick")),
        "provisional": bool(pred.get("provisional", True)),
    }


def build(pl_payload: dict, log: dict, now: pd.Timestamp) -> dict:
    """The published board. `log` is the frozen-score log, mutated in place."""
    sel = load_selection()
    wanted = list(sel.get("fixtures") or [])
    by_key = {f"{m['home']}|{m['away']}": m for m in pl_payload.get("matches", [])}
    season = str(pd.Timestamp(now).year)

    picks_out, missing = [], []
    for key in wanted:
        m = by_key.get(key)
        if not m:
            missing.append(key)
            continue
        e = _entry(m)
        if not e:
            missing.append(key)
            continue
        # Freeze the scoreline on the same terms as every other pick: inside the
        # lock window it is committed and never rewritten, so the record grades
        # the genuine pre-match call.
        hours_out = (pd.Timestamp(e["date"]) - now).total_seconds() / 3600.0
        if hours_out <= 0.75 and not e["provisional"]:
            frozen = picks.lock_pick(log, f"{season}:six:{e['id']}",
                                     pick=e["score"], confidence=1,
                                     kickoff=e["date"], now=now,
                                     p_pick=e["score_pct"] / 100.0, board=True)
            e["score"] = frozen["pick"]
            e["provisional"] = False
        picks_out.append(e)

    settled, correct = [], 0
    for m in pl_payload.get("season", []):
        entry = log.get(f"{season}:six:{m.get('id')}")
        if not entry or not m.get("result"):
            continue
        r = m["result"]
        actual = f"{r['home_goals']}-{r['away_goals']}"
        hit = entry["pick"] == actual
        correct += hit
        settled.append({"home": m["home"], "away": m["away"], "date": m["date"],
                        "predicted": entry["pick"], "actual": actual,
                        "graded": "correct" if hit else "wrong"})
    settled.sort(key=lambda x: x["date"], reverse=True)

    return {
        "updated": pd.Timestamp(now).isoformat(),
        "matchweek": sel.get("matchweek"),
        "selection_verified_on": sel.get("_verified_on"),
        "picks": picks_out,
        "missing_fixtures": missing,
        "settled": settled[:24],
        "record": {"correct": correct, "total": len(settled)},
        # Published WITH the picks so the claim and its evidence travel together.
        "expected_of_six": EXPECTED_OF_SIX,
        "hit_rate_pct": PL_HIT_RATE_PCT,
        "baseline_pct": PL_BASELINE_PCT,
    }
