"""Player data from Understat: season totals + shot events.

WHY NOT FBREF: soccerdata's FBref player-match reader drives a headless Chrome
once per match page (~4/min), so five seasons of one league is ~8 hours and four
leagues is days. Understat gives the same signal in seconds:

  read_player_season_stats  -- minutes, non-penalty goals, npxG, shots, position
                               (one request per season)
  read_shot_events          -- every shot: player, result, situation
                               (one request per season; ~25s for 9.5k shots)

Shots on target and penalty attempts are not in the season totals, so they are
derived from the shot events. Penalty attempts also tell us empirically WHO takes
the penalties, rather than us hardcoding a taker per club.

Output is one row per player-SEASON (not per appearance). props.player_rates
sums over rows and weights them by age, so season rows and appearance rows carry
identical semantics -- the only thing lost is per-match granularity, which the
props gate works around (see props_backtest.py).
"""
import json
import os
from pathlib import Path
import unicodedata

import pandas as pd
import soccerdata as sd

from leagues import config
from leagues.names import canonical, UnknownTeam

# Understat "result" values that count as on target. A shot against the post is
# NOT on target, and an own goal is not the shooter's shot at all.
ON_TARGET = {"Goal", "Saved Shot"}
POSITION_MAP = {"F": "FW", "M": "MF", "D": "DF", "GK": "GK", "AM": "AM"}
MIN_COMPLETE_ROSTER = 18
MAX_ROSTER_AGE_HOURS = 72


# Letters NFKD does NOT decompose: they are distinct code points, not
# base+combining pairs, so the strip-combining-marks pass below leaves them
# unchanged. Without these, "Djordje Petrovic" (Understat) and "Đ. Petrović"
# (API-Football) key to different strings and a current player is dropped as
# departed. Same class of failure for Nordic o-slash and Polish l-stroke.
_FOLD = str.maketrans({
    "đ": "d", "Đ": "d", "ð": "d", "Ð": "d",
    "ø": "o", "Ø": "o", "ł": "l", "Ł": "l",
    "ı": "i", "İ": "i", "ŧ": "t", "Ŧ": "t",
    "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
    "ß": "ss", "ẞ": "ss", "þ": "th", "Þ": "th",
})


def _player_key(name: str) -> str:
    """Accent/case/punctuation-insensitive player identity key.

    This is deliberately stricter than fuzzy matching: a false match can assign a
    departed player to the wrong club and produce a confident-looking prop.
    """
    text = unicodedata.normalize("NFKD", str(name)).translate(_FOLD)
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return "".join(c for c in text if c.isalnum())


# Nobility/patronymic particles. They are >=3 characters so they survive the
# length filter, but they identify nobody -- "van" is shared by van Hecke and
# van de Ven at the same club, which made BOTH look like candidates and tripped
# the "exactly one match" guard, silently refusing to rescue either.
_PARTICLES = {"van", "von", "der", "den", "des", "del", "dos", "das", "della",
              "van't", "ten", "ter", "abu", "bin", "ibn", "mac", "the"}


def _name_tokens(name: str) -> list[str]:
    """Normalized name parts. Hyphens split as well as spaces: the roster feed
    abbreviates "Gian-Luca Waldschmidt" to "L. Waldschmidt", so the initial to
    compare against is the SECOND half of the hyphenated forename."""
    parts = str(name).replace("-", " ").split()
    return [k for k in (_player_key(p) for p in parts) if k]


def _forenames_compatible(ours: set[str], theirs: set[str]) -> bool:
    """Do two sets of non-surname name parts plausibly describe one person?

    Accepts: either side empty (mononym, or every part was the shared surname);
    a shared part; an initial matching the start of a full part; one part being
    a prefix of the other ("Josh"/"Joshua", "Ansu"/"Anssumane"); or a
    single-character typo ("Yeremi"/"Yeremy").

    Rejects two unrelated full forenames -- that is the guard which stops a
    DEPARTED player being kept alive by a namesake team-mate ("Joao Neves" must
    not match "Ruben Neves"), so it must stay strict enough to fail that.
    """
    if not ours or not theirs or (ours & theirs):
        return True
    for o in ours:
        for t in theirs:
            if len(o) == 1 or len(t) == 1:          # initial vs full name
                if t.startswith(o) or o.startswith(t):
                    return True
                continue
            if o.startswith(t) or t.startswith(o):   # nickname / short form
                return True
            # Same check with doubled letters collapsed: the short form drops a
            # doubling the formal name keeps ("Anssumane" -> "Ansu"), which a raw
            # prefix test misses on the second 's'.
            od, td = _dedouble(o), _dedouble(t)
            if od.startswith(td) or td.startswith(od):
                return True
            if abs(len(o) - len(t)) <= 1 and _within_one_edit(o, t):
                return True
    return False


def _dedouble(text: str) -> str:
    """Collapse runs of the same letter: "anssumane" -> "ansumane"."""
    out = []
    for ch in text:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _within_one_edit(a: str, b: str) -> bool:
    """True when a and b differ by at most one substitution/insertion/deletion."""
    if a == b:
        return True
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):
        if short == long[:i] + long[i + 1:]:
            return True
    return False


def load_roster_snapshot(league: str) -> dict:
    """Load the dated free-source roster snapshot for one league."""
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "rosters.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get(league, {})


def roster_snapshot_age_hours(league: str | None = None) -> float | None:
    """Age of the roster evidence, or None when absent/malformed."""
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "rosters.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp = ((payload.get("_league_verified_at") or {}).get(league)
                 if league else None) or payload["_verified_at"]
        checked = pd.Timestamp(stamp)
        checked = (checked.tz_localize("UTC") if checked.tzinfo is None
                   else checked.tz_convert("UTC"))
        return float((pd.Timestamp.now("UTC") - checked).total_seconds() / 3600)
    except (KeyError, TypeError, ValueError):
        return None


def roster_snapshot_status(league: str) -> tuple[str, float | None]:
    """("missing" | "stale" | "ok", age_hours).

    Split out so callers can word a warning correctly. Before this, "the roster
    source lists fewer than 18 players for these clubs" was the ONLY message the
    page could show -- even when the real cause was that no snapshot file existed
    at all, or that it had aged past the 72h limit. Those are three different
    problems (a specific club is under-listed vs. we have no current evidence for
    ANY club) and reporting them identically is the exact conflation a reader
    cannot act on.
    """
    snapshot = load_roster_snapshot(league)
    age = roster_snapshot_age_hours(league)
    if not snapshot:
        return "missing", None
    if age is None or age > MAX_ROSTER_AGE_HOURS:
        return "stale", age
    return "ok", age


def reconcile_rates_to_roster(rates: pd.DataFrame, league: str,
                              min_players: int = MIN_COMPLETE_ROSTER):
    """Return (safe rates, incomplete clubs, unmatched historical players,
    ambiguous identities).

    The snapshot CORROBORATES; it does not by itself convict. Where a club's roster
    is complete we trust it fully: it reassigns a player's club (catching the
    January-transfer class our season-attributed source gets wrong) and a player
    absent from the whole league is treated as departed. Where a club's roster is
    thin we fall back to the existing attribution and warn, because absence from
    incomplete evidence is not evidence of absence.

    That distinction is load-bearing. Deleting thin clubs outright removed Real
    Madrid, Barcelona, Atletico, PSG, Marseille and 14 of 18 Bundesliga sides --
    70% of La Liga and Ligue 1 players, Mbappe and Raphinha among them -- because
    the free roster feed happened to list fewer than 18 names for them. Failing
    closed on the source's completeness rather than on the player's status turned a
    feed-quality problem into silent deletion of the best players in Europe.
    """
    snapshot = load_roster_snapshot(league)
    age = roster_snapshot_age_hours(league)
    if not snapshot or age is None or age > MAX_ROSTER_AGE_HOURS:
        # No usable evidence at all -> keep the existing attribution untouched and
        # report every club as unverified. Withholding the whole player model on a
        # missing/stale snapshot punishes the reader for a feed problem.
        teams = sorted(snapshot) if snapshot else (
            sorted(set(rates["team"])) if not rates.empty else [])
        return rates.reset_index(drop=True), teams, [], []

    incomplete = sorted(
        club for club, entry in snapshot.items()
        if len(entry.get("players", [])) < min_players
    )
    complete_clubs = {c for c in snapshot if c not in incomplete}
    current = {}
    key_names = {}         # key -> {club: name}, used to report WHICH clubs collided
    duplicate_keys = set()
    # Roster names per COMPLETE club, used only to rescue a player already
    # attributed to that club. Never used to move a player between clubs, so it
    # cannot manufacture a transfer.
    by_club = {}
    for club, entry in snapshot.items():
        if club in incomplete:
            continue
        for player in entry.get("players", []):
            name = player.get("name", "")
            key = _player_key(name)
            if not key:
                continue
            key_names.setdefault(key, {})[club] = name
            if key in current and current[key] != club:
                duplicate_keys.add(key)
            current[key] = club
            by_club.setdefault(club, []).append(name)
    for key in duplicate_keys:
        current.pop(key, None)  # ambiguous identity -> withhold, never guess
    # Reported so an ambiguity is visible rather than silently discarded, e.g.
    # "Real Madrid/Ath Bilbao: Alex Garcia" -- exactly the two-club identity
    # collision the withholding logic exists to protect against.
    ambiguous = sorted(
        "/".join(sorted(key_names[key])) + ": " + next(iter(key_names[key].values()))
        for key in duplicate_keys)

    # MANUAL OVERRIDES OUTRANK THE FEED. transfers.json exists precisely for the
    # window the roster feed cannot cover: a move announced hours ago, where the
    # feed still lists the old club, Understat agrees with it, and nothing looks
    # inconsistent to any automated check. Before this, the reconciliation below
    # saw an exact name match at the old club and reassigned the player straight
    # back, silently undoing the override -- Bruno Guimaraes was moved to Arsenal
    # in transfers.json and still published at Newcastle. A human who has checked
    # two sources beats a feed that is up to a day behind, so an overridden player
    # keeps the club the override gives him.
    overridden = {_player_key(p) for p, club in (load_transfers(league) or {}).items()
                  if club}

    kept, unmatched = [], []
    for _, row in rates.iterrows():
        if _player_key(row["player"]) in overridden:
            kept.append(row.copy())          # already at the overridden club
            continue
        club = current.get(_player_key(row["player"]))
        if club is None:
            # Rescue pass: Understat's spelling routinely differs from the roster
            # feed's, and treating every difference as a departure deleted real
            # first-team players (Alisson, Ezri Konsa, Ansu Fati, Amad Diallo...).
            # Match on any SHARED NAME PART, not just the last token, because the
            # two feeds disagree about which part is the surname:
            #   "Ezri Konsa Ngoyo"  vs "E. Konsa"        (surname in the middle)
            #   "Alisson"           vs "Alisson Becker"  (mononym vs full name)
            #   "Woo-Yeong Jeong"   vs "Jeong Woo-Yeong" (name order reversed)
            #
            # Guards, because a shared name part alone is weak evidence:
            #  1. only within the club we already have him at -- cannot invent a
            #     transfer;
            #  2. exactly ONE roster name at that club may share a part, otherwise
            #     we cannot tell which team-mate we matched (Inaki vs Nico Williams);
            #  3. the remaining name parts must be compatible (_forenames_compatible),
            #     so a departed "Joao Neves" is not kept alive by "Ruben Neves";
            #  4. a shared part that is only BOTH names' first part is a forename
            #     collision, not identity evidence ("Marc Cucurella" must not match
            #     team-mate "Marc Guiu"). A mononym is exempt: its single part is
            #     the whole name, so "Alisson" matching "Alisson Becker" is real;
            #  5. the candidate's OWN full-name key must not be one of the globally
            #     ambiguous keys withheld above, or an identity we already decided
            #     we cannot trust would be let straight back in here.
            ours_all = _name_tokens(row["player"])
            strong = {t for t in ours_all if len(t) >= 3 and t not in _PARTICLES}
            cands = []
            for rname in by_club.get(row["team"], ()):
                theirs_all = _name_tokens(rname)
                shared = strong & {t for t in theirs_all
                                   if len(t) >= 3 and t not in _PARTICLES}
                if not shared:
                    continue
                forename_only = (len(ours_all) >= 2 and len(theirs_all) >= 2
                                 and shared == {ours_all[0]} == {theirs_all[0]})
                if forename_only:
                    continue
                # Compatibility is part of BEING a candidate, not a test applied
                # after picking one. Filtering afterwards let an incompatible
                # near-miss inflate the count and veto a genuine unique match.
                if not _forenames_compatible(set(ours_all) - shared,
                                             set(theirs_all) - shared):
                    continue
                cands.append(rname)
            if len(cands) == 1 and _player_key(cands[0]) not in duplicate_keys:
                club = row["team"]
        if club is None:
            if row["team"] in complete_clubs:
                # Complete roster for his club and he is nowhere in the league:
                # genuinely gone. Drop him.
                unmatched.append(f"{row['team']}/{row['player']}")
                continue
            club = row["team"]        # thin evidence -> keep existing attribution
        item = row.copy()
        item["team"] = club
        kept.append(item)
    safe = pd.DataFrame(kept, columns=rates.columns)
    return safe.reset_index(drop=True), incomplete, sorted(unmatched), ambiguous


def understat_position(pos: str) -> str:
    """Understat spells positions like "F M S" / "D S" / "GK"; "S" means
    substitute, so take the first token that is a real position."""
    for token in str(pos).split():
        if token in POSITION_MAP:
            return POSITION_MAP[token]
    return "MF"


def season_end(season: str) -> pd.Timestamp:
    """"2526" -> 2026-05-31. The decay in props.player_rates is measured from
    when the football was played, so each season row is dated at its end."""
    end_year = 2000 + int(str(season)[2:4])
    return pd.Timestamp(year=end_year, month=5, day=31)


def _assign_current_club(df: pd.DataFrame, transfers: dict | None) -> pd.DataFrame:
    """Attribute every player to his CURRENT club and drop players who left.

    A player's club is his most recent SEASON's club -- but Understat has no data
    for the in-progress season, so summer-window moves are invisible. `transfers`
    (player -> new canonical club, or None if he left the league) is a manual
    override applied on top: it re-attributes a moved player's whole history to
    his new club (his scoring rate follows him) and removes anyone who left."""
    latest = df.sort_values("season").groupby("player")["team"].last()
    for player, club in (transfers or {}).items():
        latest[player] = club                        # club may be None (departed)
    df = df.copy()
    df["team"] = df["player"].map(latest)
    return df[df["team"].notna()].reset_index(drop=True)   # drop departed players


def build_player_logs(season_stats: pd.DataFrame, shots: pd.DataFrame,
                      league: str, transfers: dict | None = None) -> pd.DataFrame:
    """Pure parser -- no network. One row per player-season."""
    df = season_stats.copy()
    df["team"] = [canonical(t, league) for t in df["team"]]
    df["pos"] = [understat_position(p) for p in df["position"]]
    for c in ("minutes", "np_goals", "np_xg", "shots"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Understat exposes appearances as `matches`. Keep them: event probability
    # needs to distinguish "50% chance of 80 minutes" from "certain to play 40".
    # They have the same expected minutes but very different chances of 2+ shots.
    if "matches" in df:
        df["appearances"] = pd.to_numeric(df["matches"], errors="coerce").fillna(0)
    else:
        # Compatibility with older cached frames. This conservative estimate
        # never claims more appearances than full-match equivalents observed.
        df["appearances"] = (df["minutes"] / 90.0).clip(lower=0)
    df = df.rename(columns={"np_xg": "npxg"})

    if shots is None or len(shots) == 0:
        # Shot-level data unavailable (an upstream soccerdata parser bug crashes
        # read_shot_events on some leagues -- one match returns its roster as a
        # list, not a dict). Degrade rather than lose the whole league: estimate
        # SOT from the league-average on-target ratio, and leave penalties unknown.
        from leagues.props import SOT_RATIO_PRIOR
        df["season"] = df["season"].astype(str)
        df["sot"] = (df["shots"] * SOT_RATIO_PRIOR).round().astype(int)
        df["pens_att"] = 0
        df = _assign_current_club(df, transfers)
        df["date"] = [season_end(s) for s in df["season"]]
        return df[["date", "season", "team", "player", "pos", "minutes", "appearances", "np_goals",
                   "shots", "sot", "npxg", "pens_att"]].reset_index(drop=True)

    # shots on target + penalty attempts, per player-season, from the shot events
    ev = shots.copy()
    ev["team"] = [canonical(t, league) for t in ev["team"]]
    ev["is_sot"] = ev["result"].isin(ON_TARGET)
    # PENALTIES ARE NOT LABELLED. soccerdata's Understat reader maps the source's
    # "Penalty" situation to NA rather than a value, so `situation` is null for
    # exactly the penalties and nothing else. Verified on 2025-26 PL: all 92
    # NA-situation shots fall in the 0.70-0.80 xG band (penalty xG is ~0.76) and
    # the players taking them are the actual PL penalty takers. Matching on NA is
    # therefore correct -- and matching on the string "penalty" silently yields
    # zero takers, which is the bug this comment exists to prevent.
    ev["is_pen"] = ev["situation"].isna()
    agg = (ev.groupby(["season", "player"], as_index=False)
             .agg(sot=("is_sot", "sum"), pens_att=("is_pen", "sum")))
    agg["season"] = agg["season"].astype(str)
    df["season"] = df["season"].astype(str)
    df = df.merge(agg, on=["season", "player"], how="left")
    df["sot"] = df["sot"].fillna(0).astype(int)
    df["pens_att"] = df["pens_att"].fillna(0).astype(int)

    # A player who changed clubs must be attributed to his CURRENT club, or
    # props.player_rates (which groups by team+player) would split him into two
    # half-players at two different clubs. Transfer overrides are applied here too.
    df = _assign_current_club(df, transfers)

    df["date"] = [season_end(s) for s in df["season"]]
    return df[["date", "season", "team", "player", "pos", "minutes", "appearances", "np_goals",
               "shots", "sot", "npxg", "pens_att"]].reset_index(drop=True)


def fetch_player_logs(league: str, apply_transfers: bool = True) -> pd.DataFrame:
    """Download (cached) five seasons of player season totals + shot events.

    apply_transfers=False returns the RAW attribution with no overrides -- used by
    scripts/apply_transfers.py so it can still find players a previous override
    removed (otherwise resolving names against an already-filtered list would drop
    them from the file and silently restore them to their old club)."""
    lg = config.get(league)
    us = sd.Understat(leagues=lg.understat, seasons=list(lg.history_seasons))

    stats = us.read_player_season_stats().reset_index()
    required = {"season", "team", "player", "position", "minutes", "np_goals",
                "np_xg", "shots"}
    missing = required - set(stats.columns)
    if missing:
        raise RuntimeError(f"Understat player schema changed; missing {sorted(missing)}. "
                           f"Got: {list(stats.columns)}")

    try:
        shots = us.read_shot_events().reset_index()
        for col in ("season", "team", "player", "result", "situation"):
            if col not in shots.columns:
                raise RuntimeError(f"Understat shot schema changed; missing {col!r}. "
                                   f"Got: {list(shots.columns)}")
    except Exception as exc:
        # Upstream soccerdata bug (e.g. GER-Bundesliga: a match roster comes back
        # as a list, not a dict, crashing read_shot_events). Don't sink the whole
        # league -- degrade to season stats only (see build_player_logs).
        print(f"WARNING: shot events unavailable for {league} ({type(exc).__name__}: "
              f"{exc}); shots-on-target use the league-average ratio and penalty "
              f"takers are not identified")
        shots = None

    tr = load_transfers(league) if apply_transfers else None
    return build_player_logs(stats, shots, league, transfers=tr)


def api_match_stats(league: str):
    """Per-match player lines fetched from API-Football -> (DataFrame, covered).

    The fallback feed, written by scripts/sync_player_stats.py and read here so
    grading has a second source when Understat has not filed a fixture. Same
    columns as match_player_stats so callers need no special case.

    `covered` is the set of (team, date) sides the fallback can speak about. It
    is derived from the SQUADS the feed returned, not from the matched rows: a
    fixture where every player is present but our particular man never shot is
    still covered, and his pick must grade wrong rather than hang forever.
    """
    path = (Path(__file__).resolve().parent.parent / "data-raw" / "leagues"
            / "player_stats.json")
    empty = pd.DataFrame(columns=["date", "team", "player", "goals", "shots", "sot"])
    if not path.exists():
        return empty, set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8")).get(league, {})
    except Exception as exc:
        print(f"WARNING: player_stats.json unreadable ({exc}); ignoring fallback")
        return empty, set()

    rows, covered = [], set()
    for entry in raw.values():
        try:
            day = pd.Timestamp(entry["date"]).date()
        except Exception:
            continue
        for team in (entry.get("api_squads") or {}):
            covered.add((team, day))
        for name, st in (entry.get("players") or {}).items():
            rows.append({"date": pd.Timestamp(entry["date"]), "team": st.get("team"),
                         "player": name, "goals": int(st.get("goals") or 0),
                         "shots": int(st.get("shots") or 0),
                         "sot": int(st.get("sot") or 0)})
    return (pd.DataFrame(rows) if rows else empty), covered


def resolve_squad_name(ours: str, candidates) -> str | None:
    """Find `ours` among a squad list written by a DIFFERENT feed, or None.

    API-Football spells players its own way ("Richarlison de Andrade" for
    Understat's "Richarlison", "K. Mbappe" for "Kylian Mbappe-Lottin"), so
    grading a pick against its match stats needs the two vocabularies joined.

    The guards are the ones the roster rescue already learned the hard way, and
    they matter more here, not less: a wrong join does not merely mislabel a
    player, it settles a bet against a stranger's shot count.

      1. Callers pass ONE FIXTURE'S ONE CLUB, so a match can never cross squads.
      2. An exact identity key wins outright, before any fuzzy work.
      3. Otherwise exactly ONE candidate may share a strong name part -- two
         means we cannot tell which team-mate we found (Inaki vs Nico Williams),
         and guessing is worse than leaving the pick pending.
      4. Remaining name parts must be compatible, so "Joao Neves" cannot be
         settled by "Ruben Neves".
      5. A share that is only both names' FIRST part is a forename collision,
         not identity (Marc Cucurella vs team-mate Marc Guiu). Mononyms are
         exempt: their single part is the whole name.

    Returns the candidate string as the other feed spells it.
    """
    cands = [c for c in candidates if c]
    if not cands:
        return None
    key = _player_key(ours)
    for c in cands:
        if _player_key(c) == key:
            return c

    ours_all = _name_tokens(ours)
    strong = {t for t in ours_all if len(t) >= 3 and t not in _PARTICLES}
    if not strong:
        return None
    hits = []
    for c in cands:
        theirs_all = _name_tokens(c)
        shared = strong & {t for t in theirs_all
                           if len(t) >= 3 and t not in _PARTICLES}
        if not shared:
            continue
        if (len(ours_all) >= 2 and len(theirs_all) >= 2
                and shared == {ours_all[0]} == {theirs_all[0]}):
            continue                      # forename collision, not identity
        if not _forenames_compatible(set(ours_all) - shared,
                                     set(theirs_all) - shared):
            continue
        hits.append(c)
    return hits[0] if len(hits) == 1 else None


def match_player_stats(league: str, seasons=None) -> pd.DataFrame:
    """Per-player, per-MATCH actuals -- the feed player picks are graded against.

    fetch_player_logs is one row per player-SEASON: right for rates, useless for
    grading, because a season total cannot say whether a man scored in a given
    fixture. Shot events are the only per-match player data available here, so
    goals/shots/SOT are counted from them directly.

    Goals count PENALTIES: an anytime-scorer pick wins on a penalty, and grading
    on np_goals would mark a penalty-only scorer wrong when the pick actually won.
    Own goals are excluded from every column -- they are credited to the scorer but
    are not a shot for his own team, and they never settle a scorer bet.

    Returns date, game_id, team, player, goals, shots, sot. Returns an EMPTY frame
    if shot events cannot be read (the known upstream Bundesliga crash), so callers
    leave those picks PENDING rather than grading them all wrong.
    """
    lg = config.get(league)
    seasons = list(seasons) if seasons else list(lg.history_seasons)
    try:
        us = sd.Understat(leagues=lg.understat, seasons=seasons)
        ev = us.read_shot_events().reset_index()
    except Exception as exc:
        print(f"WARNING: no per-match player data for {league} "
              f"({type(exc).__name__}: {exc}); player picks stay PENDING")
        return pd.DataFrame(columns=["date", "game_id", "team", "player",
                                     "goals", "shots", "sot"])

    ev = ev[ev["result"] != "Own Goal"].copy()
    ev["team"] = [canonical(t, league) for t in ev["team"]]
    ev["date"] = pd.to_datetime(ev["date"]).dt.tz_localize(None)
    ev["is_goal"] = ev["result"] == "Goal"
    ev["is_sot"] = ev["result"].isin(ON_TARGET)
    out = (ev.groupby(["game_id", "date", "team", "player"], as_index=False)
             .agg(goals=("is_goal", "sum"), shots=("result", "size"),
                  sot=("is_sot", "sum")))
    for c in ("goals", "shots", "sot"):
        out[c] = out[c].astype(int)
    return out


def recent_form(league: str, n: int = 5) -> dict:
    """player -> {"goals": [...], "shots": [...], "sot": [...]}, oldest first.

    The last n matches a player actually played, from BOTH feeds merged the same
    way grading merges them: Understat wherever it has the fixture (it is
    shot-event derived and identifies penalties), the API-Football fallback only
    where Understat is silent. Two sources disagreeing about one match would
    otherwise show the same game twice in a five-game strip.

    This is display data, and it is deliberately built from the same rows the
    record is graded on. A board showing one form window while the model and the
    grader used another is explaining itself with numbers nobody scored it by.

    Returns {} rather than raising: a missing form strip is a smaller problem than
    a board that will not publish.
    """
    frames = []
    try:
        primary = match_player_stats(league)
        if not primary.empty:
            frames.append(primary.assign(_src=0))
    except Exception as exc:
        print(f"WARNING: no primary form data for {league} ({exc})")
    try:
        fallback, _ = api_match_stats(league)
        if not fallback.empty:
            frames.append(fallback.assign(_src=1))
    except Exception as exc:
        print(f"WARNING: no fallback form data for {league} ({exc})")
    if not frames:
        return {}

    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date", "player"])
    # Understat wins a duplicated (player, match): sort it first, then drop the
    # later duplicate.
    merged = (merged.sort_values(["player", "date", "_src"])
                    .drop_duplicates(subset=["player", "date"], keep="first"))

    out = {}
    for player, group in merged.groupby("player", sort=False):
        tail = group.sort_values("date").tail(n)
        entry = {}
        for column in ("goals", "shots", "sot"):
            if column in tail:
                entry[column] = [int(v) for v in tail[column].fillna(0)]
        if entry:
            out[str(player)] = entry
    return out


def load_news(league: str) -> dict:
    """Team news for one league: club -> {"out": [...], "doubt": [...], ...}.

    Injuries, suspensions and confirmed-XI omissions, gathered per matchweek for
    Best Picks fixtures only (see docs/superpowers/specs/2026-07-19-leagues-team-
    news-design.md). Absent file or league -> {}, and the props are then built from
    squad history alone exactly as before.

    Club names are canonicalised here so the file can use ordinary spellings.
    """
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "news.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8")).get(league, {})
    out = {}
    for club, entry in raw.items():
        try:
            out[canonical(club, league)] = entry
        except UnknownTeam:
            print(f"WARNING: news.json has unmapped club {club!r} for {league}")
    return out


def news_unavailable(news: dict, teams) -> tuple[set, set]:
    """(players ruled out, players doubtful) across the given clubs."""
    out, doubt = set(), set()
    for t in teams:
        e = news.get(t) or {}
        out.update(e.get("out") or [])
        doubt.update(e.get("doubt") or [])
    return out, doubt


def lineup_players(news: dict, teams) -> tuple[set, set]:
    """Confirmed starters and bench players across the requested clubs.

    A lineup is only trusted when `lineup_confirmed` is true. Predicted XIs may
    still live in the file for display/research, but they must never turn a
    provisional player pick into a locked one.
    """
    starters, bench = set(), set()
    for team in teams:
        entry = news.get(team) or {}
        if entry.get("lineup_confirmed") is not True:
            continue
        starters.update(entry.get("starters") or [])
        bench.update(entry.get("bench") or [])
    return starters, bench


def lineups_confirmed(news: dict, teams) -> bool:
    """True only when every club has an explicitly confirmed XI."""
    teams = tuple(teams)
    return bool(teams) and all(
        (news.get(team) or {}).get("lineup_confirmed") is True
        and len((news.get(team) or {}).get("starters") or []) == 11
        for team in teams
    )


def news_checked_age_hours(news: dict, teams) -> float | None:
    """Hours since the OLDEST of these clubs was news-checked; None if any is
    unchecked. Used to fail loudly rather than publish a stale Best Pick."""
    import datetime as dt
    stamps = []
    for t in teams:
        c = (news.get(t) or {}).get("checked")
        if not c:
            return None
        try:
            stamps.append(pd.Timestamp(c).tz_convert("UTC") if pd.Timestamp(c).tzinfo
                          else pd.Timestamp(c).tz_localize("UTC"))
        except Exception:
            return None
    if not stamps:
        return None
    now = pd.Timestamp.now("UTC")
    return float((now - min(stamps)).total_seconds() / 3600.0)


def transfers_age_days() -> int | None:
    """Days since squads were last verified against transfer news, or None if the
    window is shut (outside ~10 Jun - 2 Sep) or no date is recorded."""
    import datetime as dt
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "transfers.json"
    if not path.exists():
        return None
    today = dt.date.today()
    if not ((6, 10) <= (today.month, today.day) <= (9, 2)):
        return None                                  # window shut: rosters are stable
    checked = json.loads(path.read_text(encoding="utf-8")).get("_verified_on")
    try:
        return (today - dt.date.fromisoformat(checked)).days
    except (TypeError, ValueError):
        return None


def load_transfers(league: str) -> dict:
    """Manual current-window transfer overrides for one league: player -> new
    canonical club (or None if he left the league).

    Understat only has completed seasons, so summer-window moves are invisible
    until real 2026-27 games exist. This file (data-raw/leagues/transfers.json)
    carries verified moves so the props show players at their CURRENT club. Keyed
    by league; player names must match the Understat spelling; club names are
    canonicalised here. Absent file or league -> no overrides."""
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data-raw" / "leagues" / "transfers.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8")).get(league, {})
    out = {}
    for player, club in raw.items():
        out[player] = canonical(club, league) if club else None
    return out


def shot_events_available(league: str) -> bool:
    """Whether shot-level data could be read (False -> SOT/pens are degraded).
    Used by publish to surface an honest data_warning."""
    lg = config.get(league)
    us = sd.Understat(leagues=lg.understat, seasons=list(lg.history_seasons))
    try:
        us.read_shot_events()
        return True
    except Exception:
        return False


# API-Football competition ids. Duplicated from scripts.sync_rosters rather than
# imported: leagues/ is the engine and must not depend on scripts/, which imports
# it. Four integers that have not changed in years is the cheaper coupling.
_API_LEAGUE_IDS = {"PL": 39, "LALIGA": 140, "BUNDESLIGA": 78, "LIGUE1": 61}


def grading_feed_available(league: str) -> bool:
    """Can a pick in this league be SETTLED afterwards, by EITHER feed?

    Distinct from shot_events_available, which asks a different question: whether
    the RATES behind a prop can be measured. One flag used to answer both, and
    that conflation cost Bundesliga its entire player record.

    Bundesliga's Understat shot events crash upstream, so it had no way to settle
    a pick and its picks published gradeable=false -- excluded from the record and
    from every parlay. That was true when Understat was the only per-match player
    feed. It stopped being true when API-Football became the fallback: it reports
    Bundesliga goals, shots and shots on target per fixture like any other league.

    The rates question is unchanged and still says no there, because rates are
    built from SEASONS of history and the fallback only covers fixtures it is
    asked to fetch. So the shots-on-target MARKET stays withheld in Bundesliga
    while its goal and shots picks become gradeable -- two different answers,
    which is the point of asking two questions.

    Forward-looking by necessity: this claims a pick WILL be settleable, so it
    checks the fallback is actually usable (league known, key present) rather
    than assuming it.
    """
    try:
        if shot_events_available(league):
            return True
    except Exception:
        pass
    return bool(os.environ.get("API_FOOTBALL_KEY")) and league in _API_LEAGUE_IDS


def penalty_takers(logs: pd.DataFrame) -> dict:
    """team -> the player with the most recent-weighted penalty attempts.

    Empirical, not hardcoded: whoever has actually been taking them.
    """
    if logs.empty or logs["pens_att"].sum() == 0:
        return {}
    recent = logs.sort_values("season").copy()
    # weight later seasons far more heavily -- penalty duty changes hands
    rank = recent["season"].rank(method="dense")
    recent["w_pens"] = recent["pens_att"] * (2.0 ** rank)
    tally = (recent.groupby(["team", "player"], as_index=False)["w_pens"].sum()
                   .sort_values("w_pens", ascending=False))
    out = {}
    for _, r in tally.iterrows():
        if r["w_pens"] > 0 and r["team"] not in out:
            out[r["team"]] = r["player"]
    return out


def team_shot_context(league: str, recent_seasons: int = 2) -> dict:
    """How many shots each club takes and CONCEDES per match, vs league average.

    Feeds props.match_props's opp_shot_factor: a player faces more shooting
    opportunity against a club that concedes a lot of shots. Uses only the most
    recent seasons -- shot volume is a tactical property and goes stale fast.

    Returns {"concede_factor": {team: x}, "pens_per_team_match": float}.
    """
    lg = config.get(league)
    seasons = list(lg.history_seasons)[-recent_seasons:]
    us = sd.Understat(leagues=lg.understat, seasons=seasons)
    try:
        ev = us.read_shot_events().reset_index()
    except Exception as exc:
        # Same upstream soccerdata bug that fetch_player_logs guards against (a
        # match roster comes back as a list, not a dict). Degrade to neutral:
        # every opponent concedes at the league average (factor 1.0) and a
        # typical penalty rate. No shot-volume tilt, but the league still builds.
        print(f"WARNING: shot context unavailable for {league} "
              f"({type(exc).__name__}); using neutral opponent factors")
        # pens_per_team_match MUST be 0 here: without shot events fetch_player_logs
        # also degrades and identifies no penalty taker, so any penalty budget we
        # subtract from open play would be assigned to nobody, leaving each team's
        # player goal lambdas summing below the team lambda (deflated scorers).
        return {"concede_factor": {}, "pens_per_team_match": 0.0}
    ev["team"] = [canonical(t, league) for t in ev["team"]]

    # shots conceded = shots taken by the OTHER team in the same game
    per_game = ev.groupby(["game", "team"], as_index=False).size()
    conceded = []
    for game, g in per_game.groupby("game"):
        if len(g) != 2:
            continue                       # a game where one side had no shots at all
        for i, row in g.iterrows():
            other = g[g["team"] != row["team"]]["size"].sum()
            conceded.append({"team": row["team"], "conceded": other})
    c = pd.DataFrame(conceded)
    if c.empty:
        return {"concede_factor": {}, "pens_per_team_match": 0.12}

    rate = c.groupby("team")["conceded"].mean()
    league_avg = float(rate.mean()) or 1.0
    factor = (rate / league_avg).to_dict()

    n_team_matches = len(c)
    pens = int(ev["situation"].isna().sum())
    return {"concede_factor": {k: float(v) for k, v in factor.items()},
            "pens_per_team_match": pens / n_team_matches if n_team_matches else 0.12}


def expected_minutes(logs: pd.DataFrame, matches_per_season: int = 38) -> dict:
    """player -> expected minutes in the next match, from his LATEST season.

    Without this every player who has appeared for the club in five seasons is
    assumed to play 90 minutes, so a squad of ~50 (including players long gone)
    shares out the team's expected goals and the real strikers are crushed down
    to a few percent. Minutes-per-team-match is what actually distributes goals.
    """
    if logs.empty:
        return {}
    latest = logs.sort_values("season").groupby("player").last()
    mins = (latest["minutes"] / matches_per_season).clip(upper=90.0)
    return {p: float(m) for p, m in mins.items()}


def playing_time(logs: pd.DataFrame, matches_per_season: int = 38) -> dict:
    """Player availability and workload as separate quantities.

    Returns player -> {appearance_prob, minutes_if_playing, expected_minutes}.
    Using the latest season keeps tactical role current. A small beta prior stops
    one appearance from becoming 100% availability and keeps probabilities away
    from brittle zero/one extremes until a confirmed lineup overrides them.
    """
    if logs.empty:
        return {}
    latest = logs.sort_values("season").groupby("player").last()
    if "appearances" in latest:
        apps = pd.to_numeric(latest["appearances"], errors="coerce").fillna(0)
    else:
        apps = (latest["minutes"] / 90.0).clip(lower=0)
    apps = apps.clip(lower=0, upper=matches_per_season)
    # Beta(1, 1) smoothing: transparent and deliberately mild.
    p_app = ((apps + 1.0) / (matches_per_season + 2.0)).clip(0.05, 0.98)
    conditional = (latest["minutes"] / apps.replace(0, pd.NA)).fillna(0).clip(0, 90)
    out = {}
    for player in latest.index:
        p = float(p_app.loc[player])
        mins = float(conditional.loc[player])
        out[player] = {
            "appearance_prob": p,
            "minutes_if_playing": mins,
            "expected_minutes": p * mins,
        }
    return out


def current_squad(logs: pd.DataFrame) -> set:
    """Players who appeared in the most recent season -- i.e. are plausibly still
    at the club. Everyone else is a five-seasons-ago ghost who would otherwise
    soak up a share of the team's expected goals."""
    if logs.empty:
        return set()
    newest = logs["season"].max()
    return set(logs[(logs["season"] == newest) & (logs["minutes"] > 0)]["player"])
