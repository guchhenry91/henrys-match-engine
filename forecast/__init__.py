"""Shared core of the rebuilt models (docs/REBUILD_PLAN.md): forecast the GAME, not the line.

Each sport forecasts the full distribution of a player's stat; the chance of beating
ANY line is read off that distribution, so a bookmaker's line is never a model input.
"""
