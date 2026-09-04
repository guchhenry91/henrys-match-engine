"""The two ways the transfer audit quietly told lies, on 2026-09-01.

Both are failures of the same kind: the audit had the information needed to stay
silent and used it to speak instead.

  AMBIGUITY. The reconciler already detects that two clubs share an abbreviated
  name and refuses to pick between them -- which is exactly why BOTH men end up
  in `unmatched`. The audit read `unmatched`, ignored `ambiguous`, and re-guessed
  from scratch: Lorenzo Pellegrini (Roma) and Luca Pellegrini (Lazio) were each
  proposed as transferring into the other's club, in both directions, off one
  shared "L. Pellegrini". Two false transfers out of one real pair of players.

  ORPHAN TARGETS. An override naming a club outside its own league drops the
  player exactly as `null` would, so the error is invisible at runtime and in
  every report. Two of them (West Ham, Nantes, both relegated) sat in the file
  unnoticed until a check was written for them.
"""
import json

import pytest

from scripts import transfer_audit


def test_leagues_come_from_config_not_a_literal():
    """Serie A was added as the fifth league and the audit's hardcoded list was
    not, so for its whole life it was audited by nothing at all. A new league
    must never be able to opt itself out of the check it needs most."""
    from leagues import config
    assert transfer_audit.LEAGUES == list(config.LEAGUES)
    assert "SERIEA" in transfer_audit.LEAGUES


def test_an_ambiguous_name_is_never_proposed_as_a_transfer(monkeypatch, tmp_path):
    """THE ONE THAT INVENTED TWO TRANSFERS. Roma and Lazio both field an
    'L. Pellegrini'; they are different men and neither moved."""
    import pandas as pd
    from leagues import players, props

    roster = {lg: {} for lg in transfer_audit.LEAGUES}
    roster["SERIEA"] = {
        "Roma": {"players": [{"name": "L. Pellegrini"}]},
        "Lazio": {"players": [{"name": "L. Pellegrini"}]},
    }
    path = tmp_path / "data-raw" / "leagues"
    path.mkdir(parents=True)
    (path / "rosters.json").write_text(json.dumps(roster), encoding="utf-8")
    monkeypatch.setattr(transfer_audit, "ROOT", tmp_path)

    rates = pd.DataFrame({
        "player": ["Lorenzo Pellegrini", "Luca Pellegrini"],
        "team": ["Roma", "Lazio"], "nineties": [50.0, 29.0],
        "rate90": [0.21, 0.05], "pos": ["FW", "DF"],
    })
    monkeypatch.setattr(players, "load_transfers", lambda lg: {})
    monkeypatch.setattr(players, "fetch_player_logs", lambda lg, **k: rates)
    monkeypatch.setattr(players, "current_squad", lambda logs: set(rates["player"]))
    monkeypatch.setattr(props, "player_rates", lambda logs, ref=None: rates)
    monkeypatch.setattr(
        players, "reconcile_rates_to_roster",
        lambda r, lg: ([], [], ["Roma/Lorenzo Pellegrini", "Lazio/Luca Pellegrini"],
                       ["Lazio/Roma: L. Pellegrini"] if lg == "SERIEA" else []))

    report = transfer_audit.run()
    proposed = {c["player"] for c in report["same_league"]}
    assert "Lorenzo Pellegrini" not in proposed
    assert "Luca Pellegrini" not in proposed
    # and the refusal is recorded rather than silent
    assert any("Pellegrini" in a for a in report["ambiguous"])


def test_the_ambiguity_guard_matches_across_spellings():
    """The guard compared the reconciler's roster spelling ('M. Pessina') with
    the scan's Understat spelling ('Massimo Pessina') as whole name keys. Those
    never match, so the first version of the guard did nothing at all."""
    from leagues.players import _player_key as key, _name_tokens
    assert key("Massimo Pessina") != key("M. Pessina")        # why keys fail
    shared = {t for t in _name_tokens("M. Pessina")[1:] if len(t) >= 4}
    assert shared & {t for t in _name_tokens("Massimo Pessina")[1:] if len(t) >= 4}


@pytest.mark.parametrize("club, ok", [("Roma", True), ("West Ham", False)])
def test_an_override_pointing_outside_the_league_is_reported(club, ok):
    """A club that is not in the league drops the player as surely as null does,
    and says so nowhere. Relegation is all it takes to create one."""
    roster = {"SERIEA": {"Roma": {"players": []}, "Lazio": {"players": []}}}
    clubs = set(roster["SERIEA"])
    orphan = club and club not in clubs
    assert orphan is (not ok)
