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

from leagues import config, players, props
from leagues.players import _player_key as key, _name_tokens, _forenames_compatible, _PARTICLES

ROOT = Path(__file__).resolve().parents[1]
# Driven by config, not a literal. Serie A was added as the fifth league and
# this list was not, so for its whole life it was audited by nothing at all --
# a hardcoded list silently opts a new league OUT of the check it needs most.
LEAGUES = list(config.LEAGUES)


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
    report = {"same_league": [], "cross_league": [], "ambiguous": [],
              "orphan_targets": [], "checked": 0}

    # An override naming a club that is not in the league sends the player
    # nowhere: he is dropped exactly as if the entry said null, so the mistake
    # costs nothing at runtime and shows up in no output -- which is why two of
    # them (West Ham, Nantes, both relegated) sat in the file unnoticed. Cheap to
    # check, and the only thing standing between a typo'd club and a silent drop.
    for lg in LEAGUES:
        clubs = set(roster.get(lg) or {})
        for player, club in (players.load_transfers(lg) or {}).items():
            if club and club not in clubs:
                report["orphan_targets"].append(f"{lg}: {player} -> {club!r} is not a {lg} club")

    for lg in LEAGUES:
        overrides = set(players.load_transfers(lg) or {})
        logs = players.fetch_player_logs(lg, apply_transfers=True)
        squad = players.current_squad(logs)
        rates = props.player_rates(logs, ref=pd.Timestamp.now("UTC").tz_localize(None))
        rates = rates[rates["player"].isin(squad)]
        safe, _inc, unmatched, ambiguous = players.reconcile_rates_to_roster(rates, lg)
        # THE RECONCILER ALREADY KNOWS which abbreviated names are shared by two
        # clubs, and refuses to guess between them -- that is why both men land in
        # `unmatched`. Reading only `unmatched` and ignoring `ambiguous` threw that
        # knowledge away and re-guessed: Lorenzo Pellegrini (Roma) and Luca
        # Pellegrini (Lazio) were proposed as transferring INTO each other's club,
        # in both directions, off one shared 'L. Pellegrini'. Same shape as the
        # Alvaro Garcia/Alvaro Fernandez pair in the docstring, but this time the
        # pipeline had already caught it and the audit un-caught it.
        # Keyed on the CLUB PAIR plus the shared surname, not on the name string:
        # the reconciler reports the roster spelling ("M. Pessina") while the scan
        # holds the Understat spelling ("Massimo Pessina"), so comparing the two
        # name keys directly never matches and the guard silently does nothing.
        ambiguous_pairs = set()
        for a in ambiguous:
            clubs, _, shared = a.partition(": ")
            if not shared or "/" not in clubs:
                continue
            c1, c2 = clubs.split("/", 1)
            surnames = {tok for tok in _name_tokens(shared)[1:]
                        if len(tok) >= 4 and tok not in _PARTICLES}
            for s in surnames:
                ambiguous_pairs.add((c1, c2, s))
                ambiguous_pairs.add((c2, c1, s))
        report["checked"] += len(rates)
        report["ambiguous"] += [f"{lg}: {a}" for a in ambiguous]

        for u in unmatched:
            club, name = u.split("/", 1)
            if name in overrides:
                continue                       # already decided by a human
            hits = _candidates(name, lg, club, idx)
            if len(hits) != 1:
                continue                       # 0 = gone; >1 = ambiguous, never guess
            tlg, tclub, tname = hits[0]
            surnames = {tok for tok in _name_tokens(name)[1:]
                        if len(tok) >= 4 and tok not in _PARTICLES}
            if any((club, tclub, s) in ambiguous_pairs for s in surnames):
                continue                       # two men share this name; never guess
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
    if rep.get("orphan_targets"):
        print(f"\n=== ORPHAN OVERRIDE TARGETS ({len(rep['orphan_targets'])}) -- "
              f"these silently drop the player ===")
        for o in rep["orphan_targets"]:
            print(f"        {o}")
    if rep.get("ambiguous"):
        print(f"\n=== REFUSED as ambiguous ({len(rep['ambiguous'])}) -- "
              f"two clubs share the name ===")
        for a in rep["ambiguous"]:
            print(f"        {a}")
    print("\nCANDIDATES ONLY -- verify each against two independent sources or an "
          "official announcement before writing. And run a news pass for the last "
          "~72h, which this scan cannot see.")
    path = ROOT / "data-raw" / "leagues" / "transfer_audit.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
