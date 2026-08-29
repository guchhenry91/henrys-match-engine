"""Build the published Champions League board: data/ucl/board.json.

Fixtures come from the API's own schedule for the current season; strengths come
from sixteen seasons of European results. Two clubs of the 36 -- Real Betis and
Como -- have no Champions League history at all in that window, so they are SEEDED
at the strength of the competition's weakest sides and the card says so. Seven
more have fewer than twenty matches and carry a thin-history flag.

That flagging is the whole point. A number fitted from 182 matches and a number
seeded from none render identically unless the board distinguishes them, and the
second is the one a reader would most want to discount.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from leagues import picks
from leagues.config import LOCK_WINDOW_HOURS
from leagues.model import promoted_priors
from ucl import backtest, config, data

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ucl"
REPORT = ROOT / "data-raw" / "ucl" / "backtest_report.json"
FIXTURES = ROOT / "data-raw" / "ucl" / "fixtures.json"
PICKS_LOG = ROOT / "data-raw" / "ucl" / "picks_log.json"

BEST_PICK_MIN_PROB = 0.65      # same bar the soccer board uses


def _confidence(p: float) -> int:
    if p >= 0.70: return 5
    if p >= 0.60: return 4
    if p >= 0.50: return 3
    if p >= 0.40: return 2
    return 1


def evidence() -> dict:
    try:
        raw = json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        return {}
    overall = raw.get("overall") or {}
    return {"released": raw.get("released"), "n": overall.get("n"),
            "rps": overall.get("rps"), "rps_baseline": overall.get("rps_baseline"),
            "accuracy": overall.get("accuracy"),
            "seasons_scored": sorted((raw.get("per_season") or {}).keys()),
            "eras": raw.get("eras")}


def upcoming_fixtures() -> list:
    """The drawn league-phase fixtures still to be played."""
    try:
        raw = json.loads(FIXTURES.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [f for f in (raw.get("fixtures") or []) if not f.get("played")]


def freeze_and_grade(matches: list, now=None) -> tuple[dict, dict]:
    """Freeze the picks due to start, grade the ones already played.

    Reuses leagues.picks wholesale rather than writing a second implementation.
    The three rules it enforces -- lock before kickoff, grade the FROZEN pick, and
    void a pick first locked after kickoff -- are not football-specific, and a
    parallel copy of them would be a second place for the record to drift.

    A fixture with no id or no kickoff is NOT locked. The API supplies both, so
    their absence means the feed is incomplete, and a pick frozen against a
    guessed kickoff is either frozen too early to see team news or marked tainted
    and voided out of the record. Left unlocked, it simply locks on a later run.

    Returns (log, record). Mutates each match dict in place so the board DISPLAYS
    the frozen pick: showing a freshly computed pick beside a record that grades a
    different one is how a board quietly stops describing itself.
    """
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now("UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    log = picks.load_log(PICKS_LOG)
    results = data.results_by_id()

    for match in matches:
        key = match.get("id")
        kickoff = match.get("kickoff")
        if key is None or not kickoff:
            match["lockable"] = False
            continue
        key = str(key)
        match["lockable"] = True
        kickoff = pd.Timestamp(kickoff)
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")

        # A kickoff that moved after the pick froze -- correction or postponement.
        # Only ever released while the new time is still in the future.
        picks.release_moved_lock(log, key, kickoff, now=now)

        hours_out = (kickoff - now).total_seconds() / 3600.0
        if key not in log and hours_out <= LOCK_WINDOW_HOURS:
            picks.lock_pick(log, key, match["pick"], match["confidence"],
                            kickoff, now=now, p_pick=match["p_pick"],
                            board=bool(match.get("best_pick")))

        entry = log.get(key)
        if not entry:
            continue
        # Show what was frozen, not what the model would say now.
        match["pick"] = entry["pick"]
        match["p_pick"] = entry.get("p_pick", match["p_pick"])
        match["confidence"] = entry.get("confidence", match["confidence"])
        match["locked"] = True
        match["locked_at"] = entry.get("locked_at")
        match["tainted"] = bool(entry.get("tainted"))

    # GRADING SWEEPS THE LOG, NOT THE BOARD. `matches` holds only fixtures still
    # to be played, so a match grades on the very run it disappears from the board
    # -- and every run after that would have skipped it, leaving finished picks
    # pending forever and a record permanently reading 0-0.
    for key, entry in list(log.items()):
        if key.startswith("_") or entry.get("graded") is not None:
            continue
        result = results.get(key)
        if result:
            log[key] = picks.grade(entry, result)

    # Reflect the verdict on any graded fixture still shown.
    for match in matches:
        entry = log.get(str(match.get("id")))
        if entry and entry.get("graded"):
            match["graded"] = entry["graded"]
            result = results.get(str(match.get("id")))
            if result:
                match["result"] = f"{result['home_goals']}-{result['away_goals']}"

    picks.save_log(log, PICKS_LOG)
    return log, record(log)


def record(log: dict) -> dict:
    """correct/wrong/void/pending over every frozen UCL pick."""
    counts = {"correct": 0, "wrong": 0, "void": 0, "pending": 0}
    by_confidence = {}
    for key, entry in log.items():
        if key.startswith("_"):
            continue                      # _released archive, never a pick
        verdict = entry.get("graded") or "pending"
        counts[verdict] = counts.get(verdict, 0) + 1
        tier = by_confidence.setdefault(str(entry.get("confidence")),
                                        {"correct": 0, "wrong": 0})
        if verdict in tier:
            tier[verdict] += 1
    settled = counts["correct"] + counts["wrong"]
    return {
        **counts,
        "total": sum(counts.values()),
        "settled": settled,
        "hit_rate": round(counts["correct"] / settled, 4) if settled else None,
        "by_confidence": by_confidence,
    }


def build() -> dict:
    history = data.matches()
    depth = data.history_depth(history)
    drawn = data.drawn_clubs()

    model = backtest.fit_for(history)
    # Clubs with no European record cannot be fitted. Seeded at the weakest-side
    # strength -- the same fallback the leagues engine uses for a promoted club it
    # cannot resolve -- rather than at the competition average, which in a field
    # containing Real Madrid would flatter them badly.
    unfittable = []
    for name in drawn:
        try:
            model.predict(name, "Real Madrid")
        except Exception:
            unfittable.append(name)
    if unfittable:
        model = backtest.fit_for(history)
        priors = promoted_priors(model, unfittable)
        model = model.fit(history[["date", "home", "away", "home_goals", "away_goals"]],
                          ref=history["date"].max(), priors=priors)

    fixtures, matches = upcoming_fixtures(), []
    for fixture in fixtures:
        home, away = fixture.get("home"), fixture.get("away")
        if not (home and away):
            continue
        try:
            pred = model.predict(home, away)
        except Exception:
            continue
        probs = {home: pred["p_home"], "Draw": pred["p_draw"], away: pred["p_away"]}
        pick = max(probs, key=probs.get)
        matches.append({
            "id": fixture.get("id"),
            "kickoff": fixture.get("kickoff"),
            "date": fixture.get("date"), "matchday": fixture.get("matchday"),
            "home": home, "away": away,
            "home_pot": drawn.get(home), "away_pot": drawn.get(away),
            "p_home": round(pred["p_home"], 3),
            "p_draw": round(pred["p_draw"], 3),
            "p_away": round(pred["p_away"], 3),
            "pick": pick, "p_pick": round(probs[pick], 4),
            "confidence": _confidence(probs[pick]),
            "best_pick": probs[pick] >= BEST_PICK_MIN_PROB,
            # How much evidence stands behind each side's number, so a fit from
            # 182 matches is distinguishable from a seed from none.
            "home_matches": depth.get(home, 0),
            "away_matches": depth.get(away, 0),
            "thin": sorted({t for t in (home, away)
                            if depth.get(t, 0) < config.THIN_HISTORY}),
        })
    matches.sort(key=lambda m: (m.get("date") or "", -m["p_pick"]))
    _, rec = freeze_and_grade(matches)

    clubs = []
    for name, pot in sorted(drawn.items(), key=lambda kv: (kv[1], kv[0])):
        n = depth.get(name, 0)
        clubs.append({"club": name, "pot": pot, "matches": n,
                      "basis": ("seeded" if n == 0 else
                                "thin" if n < config.THIN_HISTORY else "fitted")})

    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "season": config.CURRENT_SEASON,
        "format": "36-team league phase, 8 matches each",
        "history": {"seasons": len(set(history["season"])),
                    "matches": int(len(history)),
                    "clubs": len(depth)},
        "clubs": clubs,
        "matches": matches,
        "record": rec,
        "evidence": evidence(),
        "caveats": [
            "Strengths come from Champions League and qualifying results only -- "
            "domestic form is not an input, so a club in poor league form still "
            "carries its European record.",
            f"{sum(1 for c in clubs if c['basis'] == 'seeded')} club(s) have no "
            f"Champions League history in the loaded window and are seeded at the "
            f"weakest sides' strength; "
            f"{sum(1 for c in clubs if c['basis'] == 'thin')} more have fewer than "
            f"{config.THIN_HISTORY} matches. Both are flagged per fixture.",
            "The competition changed format in 2024 from groups to a 36-team "
            "league phase; the backtest reports both eras separately.",
        ],
    }


def main():
    payload = build()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "board.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    rec = payload["record"]
    print(f"UCL board: {len(payload['matches'])} fixtures, "
          f"{len(payload['clubs'])} clubs, "
          f"history {payload['history']['matches']} matches")
    print(f"  picks: {rec['correct']}-{rec['wrong']} settled, "
          f"{rec['pending']} pending, {rec['void']} void")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
