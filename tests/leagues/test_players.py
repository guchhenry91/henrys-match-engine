import pandas as pd
import pytest

from leagues.names import UnknownTeam
from leagues.players import (build_player_logs, reconcile_rates_to_roster,
                             roster_snapshot_status,
                             season_end, understat_position)


def _season_stats():
    return pd.DataFrame([
        # a striker who moved clubs between seasons
        {"season": "2425", "team": "Brentford", "player": "Mover", "position": "F S",
         "matches": 30, "minutes": 2400, "np_goals": 12, "np_xg": 10.5, "shots": 70},
        {"season": "2526", "team": "Man City", "player": "Mover", "position": "F S",
         "matches": 28, "minutes": 2300, "np_goals": 15, "np_xg": 14.0, "shots": 85},
        {"season": "2526", "team": "Man City", "player": "Keeper", "position": "GK",
         "matches": 38, "minutes": 3420, "np_goals": 0, "np_xg": 0.0, "shots": 0},
    ])


def _shots():
    return pd.DataFrame([
        {"season": "2526", "team": "Man City", "player": "Mover",
         "situation": "Open Play", "result": "Goal"},
        {"season": "2526", "team": "Man City", "player": "Mover",
         "situation": "Open Play", "result": "Saved Shot"},
        {"season": "2526", "team": "Man City", "player": "Mover",
         "situation": "Open Play", "result": "Missed Shot"},
        # soccerdata maps Understat's "Penalty" situation to NA -- see players.py
        {"season": "2526", "team": "Man City", "player": "Mover",
         "situation": None, "result": "Goal"},
        {"season": "2425", "team": "Brentford", "player": "Mover",
         "situation": "Open Play", "result": "Blocked Shot"},
    ])


def test_one_row_per_player_season_with_canonical_teams():
    df = build_player_logs(_season_stats(), _shots(), "PL")
    assert len(df) == 3
    assert set(df["team"]) == {"Manchester City"}     # canonical, and see below


def test_a_transferred_player_is_attributed_to_his_CURRENT_club():
    """Both of Mover's seasons must carry his 2526 club, or player_rates would
    split him into two half-players at two different clubs."""
    df = build_player_logs(_season_stats(), _shots(), "PL")
    mover = df[df["player"] == "Mover"]
    assert len(mover) == 2
    assert set(mover["team"]) == {"Manchester City"}   # not Brentford


def test_shots_on_target_and_penalties_come_from_shot_events():
    df = build_player_logs(_season_stats(), _shots(), "PL")
    row = df[(df["player"] == "Mover") & (df["season"] == "2526")].iloc[0]
    # Goal + Saved Shot + the scored penalty. Penalties count: Understat's `shots`
    # total includes them, so the on-target ratio must too or sot/shots is skewed.
    # (Missed and Blocked are not on target.)
    assert row["sot"] == 3
    assert row["pens_att"] == 1


def test_position_is_mapped_off_understats_first_real_token():
    assert understat_position("F S") == "FW"
    assert understat_position("D S") == "DF"
    assert understat_position("M S") == "MF"
    assert understat_position("GK") == "GK"
    assert understat_position("S") == "MF"      # sub-only: fall back to MF


def test_season_end_dates_drive_the_decay():
    assert season_end("2526") == pd.Timestamp("2026-05-31")
    assert season_end("2122") == pd.Timestamp("2022-05-31")


def test_unmapped_team_fails_loudly():
    stats = _season_stats()
    stats.loc[0, "team"] = "Wimbledon FC"
    with pytest.raises(UnknownTeam):
        build_player_logs(stats, _shots(), "PL")


def test_penalty_taker_is_derived_from_na_situation_shots():
    """Regression guard: soccerdata maps Understat's "Penalty" situation to NA.
    Matching the string "penalty" finds nothing and every club silently ends up
    with no penalty taker."""
    from leagues.players import penalty_takers
    df = build_player_logs(_season_stats(), _shots(), "PL")
    assert df[df["player"] == "Mover"]["pens_att"].sum() == 1
    assert penalty_takers(df)["Manchester City"] == "Mover"


def test_expected_minutes_are_per_match_not_per_season():
    from leagues.players import expected_minutes
    df = build_player_logs(_season_stats(), _shots(), "PL")
    em = expected_minutes(df)
    # Keeper: 3420 minutes over a 38-match season -> a full 90 every week
    assert em["Keeper"] == 90.0
    # Mover: 2300 minutes in his latest season -> ~60 per match, NOT 90
    assert 55 < em["Mover"] < 65


def test_playing_time_separates_appearance_from_conditional_minutes():
    from leagues.players import playing_time
    df = build_player_logs(_season_stats(), _shots(), "PL")
    pt = playing_time(df)
    mover = pt["Mover"]
    assert 0 < mover["appearance_prob"] < 1
    assert mover["minutes_if_playing"] > mover["expected_minutes"]
    assert abs(mover["appearance_prob"] * mover["minutes_if_playing"]
               - mover["expected_minutes"]) < 1e-9


def test_playing_time_respects_shorter_league_seasons():
    from leagues.players import playing_time
    df = pd.DataFrame([{
        "season": "2526", "player": "Regular", "minutes": 2700,
        "appearances": 34,
    }])
    pt = playing_time(df, matches_per_season=34)["Regular"]
    assert pt["appearance_prob"] > 0.95
    assert 78 < pt["minutes_if_playing"] < 81


def test_only_explicit_complete_lineups_are_confirmed():
    from leagues.players import lineup_players, lineups_confirmed
    news = {
        "A": {"lineup_confirmed": True,
              "starters": [f"A{i}" for i in range(11)], "bench": ["A12"]},
        "B": {"lineup_confirmed": False,
              "starters": [f"B{i}" for i in range(11)]},
    }
    starters, bench = lineup_players(news, ("A", "B"))
    assert "A0" in starters and "A12" in bench
    assert "B0" not in starters
    assert lineups_confirmed(news, ("A", "B")) is False
    news["B"]["lineup_confirmed"] = True
    assert lineups_confirmed(news, ("A", "B")) is True


def test_current_squad_excludes_players_who_did_not_appear_last_season():
    """Five seasons of departed players would otherwise share out the team's
    expected goals and crush the real strikers to a few percent."""
    from leagues.players import current_squad
    stats = _season_stats()
    stats = pd.concat([stats, pd.DataFrame([
        {"season": "2223", "team": "Man City", "player": "LongGone", "position": "F S",
         "matches": 20, "minutes": 1500, "np_goals": 8, "np_xg": 7.0, "shots": 40},
    ])], ignore_index=True)
    df = build_player_logs(stats, _shots(), "PL")
    squad = current_squad(df)
    assert "Mover" in squad and "Keeper" in squad
    assert "LongGone" not in squad


def test_missing_shot_events_degrades_instead_of_crashing():
    """If shot-level data is unavailable (upstream parser bug on some leagues),
    logs still build from season stats: SOT falls back to the league-average
    ratio and penalty attempts are zero, rather than the whole league failing."""
    stats = _season_stats()
    df = build_player_logs(stats, None, "PL")          # shots=None
    assert len(df) == 3
    row = df[(df["player"] == "Mover") & (df["season"] == "2526")].iloc[0]
    # 85 shots at the ~0.35 league on-target prior -> ~30, and no penalties known
    assert row["pens_att"] == 0
    assert 0 < row["sot"] <= row["shots"]


def test_transfer_override_moves_and_removes_players():
    """Summer-window moves aren't in last season's data, so an override layer
    re-attributes a moved player to his new club and drops one who left."""
    stats = _season_stats()   # Mover (Man City), Keeper (Man City)
    # add a player at Brentford who "left" for another league this window
    import pandas as pd
    stats = pd.concat([stats, pd.DataFrame([
        {"season": "2526", "team": "Brentford", "player": "Leaver", "position": "F S",
         "matches": 30, "minutes": 2500, "np_goals": 18, "np_xg": 15.0, "shots": 90},
    ])], ignore_index=True)
    transfers = {"Mover": "Arsenal", "Leaver": None}   # Mover -> Arsenal; Leaver gone
    df = build_player_logs(stats, _shots(), "PL", transfers=transfers)
    assert set(df[df["player"] == "Mover"]["team"]) == {"Arsenal"}   # reattributed
    assert "Leaver" not in set(df["player"])                          # removed
    assert "Keeper" in set(df["player"])                             # untouched


def test_roster_reconciliation_reassigns_known_players_and_fails_closed(
        monkeypatch):
    rates = pd.DataFrame([
        {"team": "Old Club", "player": "Álex One", "rate90": 0.4},
        # attributed to a club whose roster IS complete, but absent from it ->
        # this is the only situation where we have evidence he has gone.
        {"team": "New Club", "player": "Departed", "rate90": 0.3},
        {"team": "Thin Club", "player": "Thin Player", "rate90": 0.2},
        # attributed to a club the snapshot says nothing about at all -> no
        # evidence either way, so keep him (that club is not in the fixture list,
        # so an unverifiable entry cannot reach a card).
        {"team": "Unlisted Club", "player": "Unknowable", "rate90": 0.1},
    ])
    complete = [{"id": str(i), "name": f"Squad {i}"} for i in range(17)]
    complete.append({"id": "99", "name": "Alex One"})
    snapshot = {
        "New Club": {"players": complete},
        "Thin Club": {"players": [{"id": "1", "name": "Thin Player"}]},
    }
    monkeypatch.setattr("leagues.players.load_roster_snapshot",
                        lambda league: snapshot)
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours", lambda *_: 1.0)

    safe, incomplete, unmatched, _ = reconcile_rates_to_roster(rates, "PL")

    # A COMPLETE roster convicts: "Álex One" is reassigned to his real club, and
    # "Departed" -- absent from a league whose rosters are complete -- is dropped.
    recs = {r["player"]: r["team"] for r in safe[["team", "player"]].to_dict("records")}
    assert recs["Álex One"] == "New Club"
    assert "Departed" not in recs
    assert unmatched == ["New Club/Departed"]

    # A THIN roster does NOT convict. "Thin Player" keeps his existing club rather
    # than being deleted, because absence from incomplete evidence is not evidence
    # of absence. Deleting on thin rosters removed Real Madrid, Barcelona, PSG and
    # 14 of 18 Bundesliga clubs -- 70% of two leagues -- from the player model.
    assert recs["Thin Player"] == "Thin Club"
    assert recs["Unknowable"] == "Unlisted Club"
    assert incomplete == ["Thin Club"]      # still reported, so the page can say so


def test_missing_roster_snapshot_keeps_existing_attribution(monkeypatch):
    """No evidence must not mean no player model.

    Withholding every rate on a missing snapshot means one failed feed silently
    empties the whole player product, which punishes the reader for our data
    problem. Keep what we have, and report every club as unverified."""
    rates = pd.DataFrame([
        {"team": "Arsenal", "player": "Player", "rate90": 0.4},
    ])
    monkeypatch.setattr("leagues.players.load_roster_snapshot", lambda league: {})
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours", lambda *_: None)
    safe, incomplete, unmatched, _ = reconcile_rates_to_roster(rates, "PL")
    assert list(safe["player"]) == ["Player"]      # kept, not deleted
    assert incomplete == ["Arsenal"]               # but flagged as unverified
    assert unmatched == []


def test_stale_roster_snapshot_keeps_existing_attribution(monkeypatch):
    """Same rule for evidence that has gone stale: warn, do not delete."""
    rates = pd.DataFrame([
        {"team": "Arsenal", "player": "Player", "rate90": 0.4},
    ])
    snapshot = {
        "Arsenal": {"players": [
            {"id": str(i), "name": "Player" if i == 0 else f"Squad {i}"}
            for i in range(18)
        ]}
    }
    monkeypatch.setattr("leagues.players.load_roster_snapshot",
                        lambda league: snapshot)
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours",
                        lambda *_: 72.01)
    safe, incomplete, _, _ = reconcile_rates_to_roster(rates, "PL")
    assert list(safe["player"]) == ["Player"]
    assert incomplete == ["Arsenal"]


def _snap(monkeypatch, clubs, age=1.0):
    monkeypatch.setattr("leagues.players.load_roster_snapshot", lambda league: clubs)
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours", lambda *_: age)


def _pad(n=18, prefix="Squad"):
    return [{"id": str(i), "name": f"{prefix} {i}"} for i in range(n)]


def test_surname_rescue_refuses_an_ambiguous_surname(monkeypatch):
    """Two team-mates share a surname -> we cannot tell which one we matched, so
    the rescue must decline rather than keep a departed player alive."""
    rates = pd.DataFrame([{"team": "Brentford", "player": "Neves", "rate90": 0.4}])
    _snap(monkeypatch, {"Brentford": {"players": _pad() + [
        {"id": "a", "name": "Joao Neves"}, {"id": "b", "name": "Ruben Neves"}]}})
    safe, _, unmatched, _ = reconcile_rates_to_roster(rates, "PL")
    assert safe.empty
    assert unmatched == ["Brentford/Neves"]


def test_surname_rescue_refuses_a_different_player_with_the_same_surname(monkeypatch):
    """The unique-surname guard alone is not enough: one "Neves" at the club can
    still be a DIFFERENT Neves from ours. Forenames must not contradict."""
    rates = pd.DataFrame([{"team": "Brentford", "player": "Joao Neves", "rate90": 0.4}])
    _snap(monkeypatch, {"Brentford": {"players": _pad() + [
        {"id": "b", "name": "Ruben Neves"}]}})
    safe, _, unmatched, _ = reconcile_rates_to_roster(rates, "PL")
    assert safe.empty, "a different Neves was accepted as ours"
    assert unmatched == ["Brentford/Joao Neves"]


def test_surname_rescue_accepts_a_genuine_variant(monkeypatch):
    """The case it exists for: same man, fuller name in the feed."""
    rates = pd.DataFrame([{"team": "Brentford", "player": "Thiago", "rate90": 0.4}])
    _snap(monkeypatch, {"Brentford": {"players": _pad() + [
        {"id": "t", "name": "Igor Thiago"}]}})
    safe, _, unmatched, _ = reconcile_rates_to_roster(rates, "PL")
    assert list(safe["player"]) == ["Thiago"]
    assert unmatched == []


@pytest.mark.parametrize("club,understat,roster_name,why", [
    # Understat gives a mononym, the feed gives the full name.
    ("Liverpool", "Alisson", "Alisson Becker", "mononym vs full name"),
    # The surname sits in the MIDDLE, so a last-token-only rescue never saw it.
    ("Aston Villa", "Ezri Konsa Ngoyo", "E. Konsa", "surname is not the last token"),
    ("Chelsea", "Benoit Badiashile Mukinayi", "B. Badiashile", "surname mid-name"),
    ("Manchester United", "Amad Diallo Traore", "A. Diallo", "surname mid-name"),
    # NFKD leaves these letters alone, so the keys never matched before the fold.
    ("Bournemouth", "Djordje Petrovic", "Đ. Petrović", "D-stroke folds to d"),
    ("Nice", "Jorgen Odegaard", "J. Ødegaard", "O-slash folds to o"),
    # Korean feeds disagree about name order.
    ("Union Berlin", "Woo-Yeong Jeong", "Jeong Woo-Yeong", "name order reversed"),
    ("Mainz", "Jae-Sung Lee", "Lee Jae-Sung", "name order reversed"),
    # Hyphenated forename abbreviated on its SECOND half.
    ("FC Koln", "Gian-Luca Waldschmidt", "L. Waldschmidt", "initial from 2nd half"),
    # Short forms and spelling variants of the same forename.
    ("Valencia", "Javier Guerra", "Javi Guerra", "nickname prefix"),
    ("Chelsea", "Josh Acheampong", "Joshua Kofi Acheampong", "nickname prefix"),
    ("Monaco", "Anssumane Fati", "Ansu Fati", "short form, doubled letter dropped"),
    ("Crystal Palace", "Yeremi Pino", "Yeremy Pino", "one-character spelling variant"),
    # HTML-escaped feed names are unescaped at ingest; verify the apostrophe form.
    ("Manchester City", "Nico O'Reilly", "N. O'Reilly", "apostrophe survives"),
    # A shared surname is NOT automatically ambiguous. Inaki and Nico Williams
    # are brothers at the same club; the initial distinguishes them, so refusing
    # both (the old behaviour) simply deleted an Athletic Club regular.
    ("Ath Bilbao", "Inaki Williams", "I. Williams", "brother's surname, initial resolves it"),
])
def test_rescue_recovers_real_players_the_feeds_spell_differently(
        monkeypatch, club, understat, roster_name, why):
    """Every one of these is a CURRENT first-team player who was being dropped as
    departed purely because Understat and API-Football spell him differently.
    Measured on live data: 486 players were dropped across the four leagues, ~50
    of them wrongly, including Alisson, Ezri Konsa and Ansu Fati (0.596 goals/90,
    the highest rate of any false drop)."""
    rates = pd.DataFrame([{"team": club, "player": understat, "rate90": 0.4}])
    _snap(monkeypatch, {club: {"players": _pad() + [{"id": "x", "name": roster_name}]}})
    safe, _, unmatched, _ = reconcile_rates_to_roster(rates, "PL")
    assert list(safe["player"]) == [understat], f"should have been rescued: {why}"
    assert unmatched == []


@pytest.mark.parametrize("club,understat,roster_names,why", [
    # The forename-collision guard: sharing only a FIRST name is not identity.
    ("Chelsea", "Marc Cucurella", ["Marc Guiu"], "shared forename only"),
    ("Arsenal", "Martin Odegaard", ["Martín Zubimendi"], "shared forename only"),
    ("Manchester City", "Nico O'Reilly", ["Nico González"], "shared forename only"),
    ("Villarreal", "Daniel Parejo", ["Daniel Budesca"], "shared forename only"),
    # Genuinely different people who happen to share a surname.
    ("Sevilla", "Isaac Romero", ["Rafael Romero"], "same surname, different man"),
    ("Alaves", "Antonio Martinez", ["Toni Martinez"], "unproven nickname, stays out"),
    # Two team-mates share the surname AND both are forename-compatible with the
    # rate row -> genuinely cannot tell which, so withhold. (Contrast the Williams
    # brothers above, where the initial DOES separate them.)
    ("Ath Bilbao", "I. Williams", ["Inaki Williams", "Ivan Williams"],
     "two compatible surname matches at one club"),
])
def test_rescue_still_refuses_weak_or_ambiguous_evidence(
        monkeypatch, club, understat, roster_names, why):
    """The widened rescue must not become a fuzzy matcher. Each of these would
    resurrect a departed player or merge two different people if the guards
    slipped."""
    rates = pd.DataFrame([{"team": club, "player": understat, "rate90": 0.4}])
    _snap(monkeypatch, {club: {"players": _pad() + [
        {"id": str(i), "name": n} for i, n in enumerate(roster_names)]}})
    safe, _, unmatched, _ = reconcile_rates_to_roster(rates, "PL")
    assert list(safe["player"]) == [], f"must NOT be rescued: {why}"
    assert unmatched == [f"{club}/{understat}"]


def test_manual_transfer_override_is_not_undone_by_a_lagging_roster_feed(monkeypatch):
    """The override list exists for the one window the feed cannot cover: a move
    announced hours ago, where the roster still lists the OLD club, Understat
    agrees, and nothing looks inconsistent to any automated check.

    Before this guard the reconciliation saw an exact name match at the old club
    and reassigned the player straight back, silently undoing the override --
    Bruno Guimaraes was set to Arsenal in transfers.json and still published at
    Newcastle. A human who checked two sources beats a feed a day behind."""
    monkeypatch.setattr("leagues.players.load_transfers",
                        lambda league: {"Bruno Guimaraes": "Arsenal"})
    # The feed still has him at his OLD club, with a complete squad, so without
    # the guard it would win.
    _snap(monkeypatch, {
        "Newcastle United": {"players": _pad() + [{"id": "b", "name": "Bruno Guimaraes"}]},
        "Arsenal": {"players": _pad(prefix="Gunner")},
    })
    rates = pd.DataFrame([{"team": "Arsenal", "player": "Bruno Guimaraes", "rate90": 0.2}])
    safe, _, unmatched, _ = reconcile_rates_to_roster(rates, "PL")
    assert list(safe["team"]) == ["Arsenal"], "the lagging feed must not win"
    assert unmatched == []


def test_surname_rescue_accepts_an_abbreviated_forename(monkeypatch):
    """API-Football's squad feed abbreviates some first names ("C. Tolisso" for
    "Corentin Tolisso"), which a strict full-forename comparison rejected
    outright even though the surname is unique at the club -- this silently
    dropped real current players (Real Madrid 35->14, and 15 of 18 Ligue 1
    clubs falling below MIN_SQUAD_FOR_PROPS) the day this format first showed
    up in a fresh sync. An initial is still a forename match, not a stranger."""
    rates = pd.DataFrame([{"team": "Lyon", "player": "Corentin Tolisso", "rate90": 0.4}])
    _snap(monkeypatch, {"Lyon": {"players": _pad() + [
        {"id": "t", "name": "C. Tolisso"}]}})
    safe, _, unmatched, _ = reconcile_rates_to_roster(rates, "LIGUE1")
    assert list(safe["player"]) == ["Corentin Tolisso"]
    assert unmatched == []


def test_surname_rescue_still_refuses_a_wrong_initial(monkeypatch):
    """An initial must actually match -- "R. Neves" should not rescue a rate row
    for "Joao Neves"; that is exactly the wrong-teammate case the forename
    check exists to catch, just spelled as an initial instead of a full name."""
    rates = pd.DataFrame([{"team": "Wolves", "player": "Joao Neves", "rate90": 0.4}])
    _snap(monkeypatch, {"Wolves": {"players": _pad() + [
        {"id": "r", "name": "R. Neves"}]}})
    safe, _, unmatched, _ = reconcile_rates_to_roster(rates, "PL")
    assert list(safe["player"]) == []
    assert unmatched == ["Wolves/Joao Neves"]


# ------------------------------------------------- roster status classification
def test_roster_status_distinguishes_missing_stale_and_ok(monkeypatch):
    """Before this split, a missing snapshot, a stale one, and a specific club
    being under-listed all produced the identical page warning: "the roster
    source lists fewer than 18 players for these clubs". A reader cannot act
    differently on three different problems that read the same."""
    monkeypatch.setattr("leagues.players.load_roster_snapshot", lambda league: {})
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours", lambda *_: None)
    assert roster_snapshot_status("PL") == ("missing", None)

    monkeypatch.setattr("leagues.players.load_roster_snapshot",
                        lambda league: {"Arsenal": {"players": []}})
    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours", lambda *_: 100.0)
    assert roster_snapshot_status("PL") == ("stale", 100.0)

    monkeypatch.setattr("leagues.players.roster_snapshot_age_hours", lambda *_: 1.0)
    assert roster_snapshot_status("PL") == ("ok", 1.0)


# ------------------------------------------------------- ambiguous identities
def test_ambiguous_surname_across_two_clubs_is_reported_not_swallowed(monkeypatch):
    """The same normalized name at two different clubs in the roster source was
    being withheld correctly (never guessed) but the fact of the collision was
    computed and then discarded -- no caller could ever see it happened."""
    rates = pd.DataFrame([
        {"team": "Real Madrid", "player": "Alex Garcia", "rate90": 0.3},
    ])
    snapshot = {
        "Real Madrid": {"players": _pad(prefix="RM") + [{"id": "1", "name": "Alex Garcia"}]},
        "Ath Bilbao": {"players": _pad(prefix="Bilbao") + [{"id": "2", "name": "Alex Garcia"}]},
    }
    _snap(monkeypatch, snapshot)
    safe, incomplete, unmatched, ambiguous = reconcile_rates_to_roster(rates, "PL")
    assert safe.empty                       # withheld, not guessed
    assert len(ambiguous) == 1
    assert "Alex Garcia" in ambiguous[0]
    assert "Real Madrid" in ambiguous[0] and "Ath Bilbao" in ambiguous[0]


def test_no_ambiguity_reported_when_names_are_actually_distinct(monkeypatch):
    rates = pd.DataFrame([{"team": "Brentford", "player": "Thiago", "rate90": 0.4}])
    _snap(monkeypatch, {"Brentford": {"players": _pad() + [
        {"id": "t", "name": "Igor Thiago"}]}})
    _, _, _, ambiguous = reconcile_rates_to_roster(rates, "PL")
    assert ambiguous == []
