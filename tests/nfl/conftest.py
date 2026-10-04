import pytest

from nfl import data


@pytest.fixture(autouse=True)
def _no_snap_download(monkeypatch):
    """Tests never download snap counts: none filed, unless a test says otherwise."""
    monkeypatch.setattr(data, "participation", lambda season: (set(), set()))
