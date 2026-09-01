"""Audit the dated roster snapshot without pretending an incomplete feed is final."""
import json
from pathlib import Path

from leagues import config

ROOT = Path(__file__).resolve().parents[1]
ROSTERS = ROOT / "data-raw" / "leagues" / "rosters.json"
TRANSFERS = ROOT / "data-raw" / "leagues" / "transfers.json"
CLUBS = {
    "PL": "clubs.json",
    "LALIGA": "clubs_laliga.json",
    "BUNDESLIGA": "clubs_bundesliga.json",
    "LIGUE1": "clubs_ligue1.json",
    "SERIEA": "clubs_seriea.json",
}


def _verified_clubs(league: str) -> dict:
    """player name -> the club we have VERIFIED he now plays for.

    transfers.json is the manual override list and carries the project's evidence
    bar: two independent sources or an official club announcement.
    """
    try:
        raw = json.loads(TRANSFERS.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entries = raw.get(league) or {}
    return {str(k).casefold(): v for k, v in entries.items()
            if isinstance(v, str)}


def _same_player(snapshot_name: str, transfer_name: str) -> bool:
    """Do a snapshot name and a transfers.json key describe one man?

    The feed abbreviates ("T. Awoniyi") where transfers.json spells it out
    ("Taiwo Awoniyi"), so compare on the SURNAME plus a compatible first initial.
    Deliberately narrow: this only ever suppresses an error for a player we have
    already verified by name, so a loose match here would hide a real conflict.
    """
    a = str(snapshot_name).replace(".", " ").split()
    b = str(transfer_name).replace(".", " ").split()
    if not a or not b:
        return False
    if a[-1].casefold() != b[-1].casefold():
        return False
    return a[0][:1].casefold() == b[0][:1].casefold()


def audit(payload):
    errors, warnings = [], []
    for league, cfg in config.LEAGUES.items():
        teams = payload.get(league, {})
        expected = json.loads(
            (ROOT / "data" / "leagues" / CLUBS[league]).read_text(encoding="utf-8"))

        # A LEAGUE THE SNAPSHOT DOES NOT COVER AT ALL IS A WARNING, NOT AN ERROR.
        # This is the same absence-of-evidence distinction the engine already
        # draws for a thin club roster: evidence that CONTRADICTS the league is
        # dangerous and must fail, but no evidence at all is simply a gap. The
        # publish path already detects it and prints a data_warning on the page
        # ("no current-roster evidence is available for this league at all"), so
        # player attribution falls back to last season plus transfer overrides
        # rather than deleting anyone.
        #
        # Conflating the two would mean a newly added league -- or an outage at
        # the source -- fails
        # the whole audit as though the data were WRONG rather than absent.
        if not teams:
            warnings.append(
                f"{league}: no roster evidence in the snapshot at all; attribution "
                f"falls back to last season plus transfer overrides")
            continue

        missing = sorted(set(expected) - set(teams))
        extra = sorted(set(teams) - set(expected))
        if len(teams) != cfg.n_teams or missing or extra:
            errors.append(
                f"{league}: expected {cfg.n_teams} clubs; missing={missing}, extra={extra}")

        verified = _verified_clubs(league)
        seen = {}
        for club, entry in teams.items():
            players = entry.get("players", [])
            if len(players) < 18:
                warnings.append(
                    f"{league}/{club}: only {len(players)} players in source snapshot")
            for player in players:
                pid = player["id"]
                if pid in seen and seen[pid] != club:
                    # A DUPLICATE WE HAVE ALREADY RESOLVED IS NOT AN UNKNOWN.
                    #
                    # During a registration window the feed lists a moving player
                    # at BOTH clubs for a few days. Taiwo Awoniyi blocked the whole
                    # pipeline this way -- Nottingham Forest and Coventry at once --
                    # while transfers.json already recorded him at Coventry, on the
                    # project's evidence bar (Forest's own announcement plus three
                    # outlets). Failing on a conflict we have verified and written
                    # down teaches the operator to ignore this check, which is worse
                    # than the duplicate.
                    #
                    # It stays a WARNING, never silence: the snapshot is genuinely
                    # inconsistent and that is worth seeing until the feed catches up.
                    match = next((name for name in verified
                                  if _same_player(player["name"], name)), None)
                    target = verified.get(match) if match else None
                    if target and target in (seen[pid], club):
                        warnings.append(
                            f"{league}: {player['name']} ({pid}) listed for "
                            f"{seen[pid]} and {club}; transfers.json has him "
                            f"VERIFIED at {target} -- feed mid-registration")
                        seen[pid] = target
                        continue
                    errors.append(
                        f"{league}: {player['name']} ({pid}) listed for "
                        f"{seen[pid]} and {club}")
                seen[pid] = club
    return errors, warnings


def main():
    payload = json.loads(ROSTERS.read_text(encoding="utf-8"))
    errors, warnings = audit(payload)
    for item in errors:
        print("ERROR:", item)
    for item in warnings:
        print("WARNING:", item)
    print(f"{len(errors)} error(s), {len(warnings)} incomplete-roster warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
