import pytest
from leagues.names import canonical, UnknownTeam


def test_canonical_passes_through_known_canonical_name():
    assert canonical("Arsenal", "PL") == "Arsenal"


def test_canonical_maps_source_aliases():
    assert canonical("Man United", "PL") == "Manchester United"
    assert canonical("Man Utd", "PL") == "Manchester United"


def test_canonical_strips_accents_and_suffixes():
    assert canonical("Deportivo Alavés", "LALIGA") == "Alaves"
    assert canonical("Olympique de Marseille", "LIGUE1") == "Marseille"


def test_unknown_team_raises_loudly():
    with pytest.raises(UnknownTeam):
        canonical("Nonexistent FC", "PL")
