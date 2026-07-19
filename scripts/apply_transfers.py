"""Resolve agent-verified transfer names to EXACT Understat spellings.

A transfers.json key that does not match the source spelling character-for-character
is a silent no-op -- the player keeps appearing at his old club and nothing warns
you. This resolves each verified name against the real player list, reports anything
it cannot match, and writes the file only from confirmed matches.
"""
import json
import sys
import unicodedata
from pathlib import Path

from leagues import players

ROOT = Path(__file__).resolve().parents[1]

# Verified by four per-league agents, each requiring two independent sources or an
# official club/league announcement. Rumours deliberately excluded.
VERIFIED = {
    "PL": {
        "Mohamed Salah": None,          # released by Liverpool
        "Anthony Gordon": None,         # -> Barcelona
        "Raul Jimenez": None,           # -> Wolves (relegated)
        "Leandro Trossard": None,       # -> Besiktas
        "Antoine Semenyo": "Manchester City",
        "Donyell Malen": None,
        "Tyrique George": "Everton",
        "Harry Wilson": "Leeds",
        "Eliezer Mayenda": None,        # -> Rennes
        "Bertrand Traore": None,
        "Brajan Gruda": None,           # -> RB Leipzig (loan w/ obligation)
        "Jack Grealish": "Manchester City",   # Everton loan expired
        "Lorenzo Lucca": None,          # -> back to Napoli
        "Samuel Chukwueze": None,       # stayed at Milan
        "Reiss Nelson": "Arsenal",      # Brentford loan expired
    },
    "LALIGA": {
        "Robert Lewandowski": None,     # -> Chicago Fire
        "Marcus Rashford": None,        # -> back to Man Utd
        "Alexis Sanchez": None,
        "Dodi Lukebakio": None,         # -> Benfica (Sept 2025)
        "Umar Sadiq": "Valencia",       # Jan 2026
        "Largie Ramazani": None,        # loan expired, back at Leeds
        "Morales": None,                # contract ended at Levante
    },
    "BUNDESLIGA": {
        "Samuel Essende": None,
        "Phillip Tietz": "Mainz",
        "Elias Saad": None,
        "Nicolas Jackson": None,
        "Karim Adeyemi": None,          # -> Barcelona
        "Arnaud Kalimuendo": None,
        "Junior Adamu": "Schalke 04",
        "Maximilian Philipp": None,
        "Robert Glatzel": None,         # -> Wolfsburg
        "Haris Tabakovic": None,        # -> RB Salzburg
        "Nelson Weiper": None,          # -> Sturm Graz (loan)
        "Armindo Sieb": "Bayern Munich",
        "Lois Openda": None,            # -> Juventus
        "Timo Werner": None,            # -> MLS
        "Nick Woltemade": None,         # at Newcastle since Aug 2025
        "Jovan Milosevic": "Stuttgart",
        "Victor Boniface": "Leverkusen",
    },
    "LIGUE1": {
        "Sidiki Cherif": None,          # -> Fenerbahce
        "Goduine Koyalipou": None,      # -> Kortrijk (loan)
        "Ibrahim Osman": None,
        "Remy Labeau Lascary": "Lens",
        "Eric Junior Dina Ebimbe": None,
        "Kenny Quetant": None,          # -> Werder Bremen
        "Wesley Said": None,
        "Ahmadou Bamba Dieng": None,
        "Sambou Soumano": None,
        "Pablo Pagis": "Paris FC",
        "Georges Mikautadze": None,     # at Villarreal since Sept 2025
        "Roman Yaremchuk": None,
        "Endrick": None,                # loan expired, back at Real Madrid
        "Pierre-Emerick Aubameyang": None,
        "Sepe Elye Wahi": None,
        "Goncalo Ramos": None,          # -> AC Milan
        "Esteban Lepaul": "Rennes",
        "Emanuel Emegha": None,         # -> Chelsea
        "David Datro Fofana": None,
        "Emersonn": None,               # -> Ipswich
        "Frank Magri": None,
    },
}

# Deliberately NOT applied: below the two-independent-source bar.
HELD_BACK = {
    "LIGUE1": {"Sekou Mara": "Strasbourg (Transfermarkt only -- two pages, one source)"},
}


def _key(s):
    """Accent- and case-insensitive comparison key."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def main():
    out, unmatched = {}, []
    for league, moves in VERIFIED.items():
        # Load the RAW player list with no overrides applied: players removed by a
        # previous run of this file would otherwise be unfindable, and would then be
        # dropped from the new file and silently reappear at their old club.
        logs = players.fetch_player_logs(league, apply_transfers=False)
        real = sorted(set(logs["player"]))
        index = {}
        for r in real:
            index.setdefault(_key(r), []).append(r)

        resolved = {}
        for name, dest in moves.items():
            k = _key(name)
            hits = index.get(k, [])
            if not hits:
                # Understat sometimes carries extra given/family names
                # ("Arnaud Kalimuendo" -> "Arnaud Kalimuendo Muinga").
                hits = [r for kk, names in index.items() for r in names
                        if kk.startswith(k + " ") or kk.endswith(" " + k)]
            if len(hits) == 1:
                resolved[hits[0]] = dest
            elif len(hits) > 1:
                unmatched.append(f"{league}: {name!r} ambiguous -> {hits}")
            else:
                unmatched.append(f"{league}: {name!r} NOT FOUND in {league} player data")
        out[league] = resolved
        print(f"{league}: matched {len(resolved)}/{len(moves)}")

    for u in unmatched:
        print("  UNMATCHED", u)

    import datetime as dt
    payload = {
        "_note": ("Manual current-window transfer overrides. Understat has no 2026-27 "
                  "data yet, so summer-window moves (and late 2025-26 ones) are invisible "
                  "without this file. Keys are EXACT Understat spellings, resolved by "
                  "scripts/apply_transfers.py. Value = new canonical club, or null if the "
                  "player left this league. Verified by per-league agents requiring two "
                  "independent sources or an official announcement; rumours excluded."),
        "_verified_on": dt.date.today().isoformat(),
        "_held_back": HELD_BACK,
        **out,
    }
    path = ROOT / "data-raw" / "leagues" / "transfers.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
