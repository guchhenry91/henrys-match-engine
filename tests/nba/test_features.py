

def test_vacated_share_counts_only_real_absences():
    """Phase 3: production left by teammates absent from tonight's game."""
    import pandas as pd
    from nba import features as f
    d = pd.Timestamp("2026-01-01")
    rows = pd.DataFrame([
        (1, "BOS", 10, d, 0.30), (2, "BOS", 10, d, 0.20), (3, "BOS", 10, d, 0.10),
        (1, "BOS", 11, d + pd.Timedelta(days=2), 0.35), (3, "BOS", 11, d + pd.Timedelta(days=2), 0.15)],
        columns=["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "game_date", "_share"])
    v = f.vacated_share(rows).set_index("GAME_ID")["vacated"]
    assert abs(v.loc[11] - 0.20) < 1e-9
