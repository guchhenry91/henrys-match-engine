"""Find players the model may have at the wrong club. Run before every matchday.

TWO HALVES, because neither alone is sufficient:

  DATA SCAN (this script) -- every player the roster reconciliation DROPPED whose
  name also turns up at a different club. That is the only shape a stale
  attribution can take once the feed knows about the move.

  NEWS PASS (not automatable) -- a move announced in the last day or two is
  INVISIBLE here by construction: the feed still lists the old club, Understat
  agrees with it, the two are consistent, the player is never dropped, and no
  discrepancy exists for this scan to find. Bruno Guimaraes (Newcastle ->
  Arsenal, 2026-08-09) was missed exactly this way. Whatever this script prints,
  still search the news for the last ~72 hours.

The scan proposes CANDIDATES ONLY. The roster feed proves a NAME exists at a
club, never that it is the same PERSON: on the first run it confidently
proposed Alvaro Garcia and Alvaro Fernandez, both of which the pipeline already
flags as two different players at two clubs. Every write needs two independent
sources or an official club announcement.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from leagues import players, props
from leagues.players import _player_key as key, _name_tokens, _forenames_compatible, _PARTICLES

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["PL", "LALIGA", "BUNDESLIGA", "LIGUE1"]


def _index(roster: dict) -> list[tuple]:
    out = []
    for lg in LEAGUES:
        for club, entry in (roster.get(lg) or {}).items():
            for p in entry.get("players", []):
                out.append((lg, club, p["name"], _name_tokens(p["name"])))
    return out


def _candidates(uname: str, ulg: str, uclub: str, idx: list[tuple]) -> list[tuple]:
    """Strict: a SURNAME-position token of >=4 chars must be shared, and the
    remaining parts must be compatible. Mononyms are refused across clubs -- a
    bare forename matched three different Kevins to one 'Kevin' on the first run."""
    ut = _name_tokens(uname)
    if len(ut) < 2:
        return []
    surn = {t for t in ut[1:] if len(t) >= 4 and t not in _PARTICLES}
    if not surn:
        return []
    hits = []
    for lg, club, rn, rt in idx:
        if lg == ulg and club == uclub:
            continue
        if len(rt) < 2:
            continue
        shared = surn & {t for t in rt[1:] if len(t) >= 4 and t not in _PARTICLES}
        if not shared:
            continue
        if _forenames_compatible(set(ut) - shared, set(rt) - shared):
            hits.append((lg, club, rn))
    return hits


def run() -> dict:
    roster = json.load(open(ROOT / "data-raw/leagues/rosters.json", encoding="utf-8"))
    idx = _index(roster)
    report = {"same_league": [], "cross_league": [], "checked": 0}

    for lg in LEAGUES:
        overrides = set(players.load_transfers(lg) or {})
        logs = players.fetch_player_logs(lg, apply_transfers=True)
        squad = players.current_squad(logs)
        rates = props.player_rates(logs, ref=pd.Timestamp.now("UTC").tz_localize(None))
        rates = rates[rates["player"].isin(squad)]
        safe, _inc, unmatched, _amb = players.reconcile_rates_to_roster(rates, lg)
        report["checked"] += len(rates)

        for u in unmatched:
            club, name = u.split("/", 1)
            if name in overrides:
                continue                       # already decided by a human
            hits = _candidates(name, lg, club, idx)
            if len(hits) != 1:
                continue                       # 0 = gone; >1 = ambiguous, never guess
            tlg, tclub, tname = hits[0]
            row = rates[(rates["player"] == name) & (rates["team"] == club)]
            if row.empty:
                continue
            item = {
                "league": lg, "from": club, "player": name,
                "to_league": tlg, "to": tclub, "roster_name": tname,
                "exact_name": key(tname) == key(name),
                "nineties": round(float(row["nineties"].iloc[0]), 1),
                "rate90": round(float(row["rate90"].iloc[0]), 3),
                "pos": str(row["pos"].iloc[0]),
            }
            # A same-league move needs an override to REASSIGN him. A cross-league
            # move is already handled: he is correctly dropped from this league and
            # has no rate history in the new one, so an entry would change nothing.
            report["same_league" if tlg == lg else "cross_league"].append(item)

    for k in ("same_league", "cross_league"):
        report[k].sort(key=lambda x: -(x["nineties"] * x["rate90"] * 10 + x["nineties"]))
    return report


def main():
    # Player names are full of accents; a Windows console defaults to cp1252 and
    # raises on the first one it cannot encode. That killed a whole audit run
    # after the scan had finished but BEFORE the report was written, losing the
    # work to a printing detail.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rep = run()
    print(f"scanned {rep['checked']} player-rows\n")
    print(f"=== SAME-LEAGUE candidates ({len(rep['same_league'])}) -- an override would reassign these ===")
    for c in rep["same_league"]:
        tag = "EXACT" if c["exact_name"] else "fuzzy"
        print(f"  {tag} [{c['pos']}] {c['nineties']:5.1f}x90 r{c['rate90']:.2f}  "
              f"{c['league']}/{c['from']:<18} {c['player'][:24]:<24} -> {c['to']}  ({c['roster_name']})")
    print(f"\n=== CROSS-LEAGUE ({len(rep['cross_league'])}) -- already correctly dropped, no entry needed ===")
    for c in rep["cross_league"][:10]:
        print(f"        [{c['pos']}] {c['nineties']:5.1f}x90 r{c['rate90']:.2f}  "
              f"{c['league']}/{c['from']:<16} {c['player'][:22]:<22} -> {c['to_league']}/{c['to']}")
    if len(rep["cross_league"]) > 10:
        print(f"        ... and {len(rep['cross_league']) - 10} more")
    print("\nCANDIDATES ONLY -- verify each against two independent sources or an "
          "official announcement before writing. And run a news pass for the last "
          "~72h, which this scan cannot see.")
    path = ROOT / "data-raw" / "leagues" / "transfer_audit.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
