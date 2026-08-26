"""NFL engine: player props and game winners, on nflverse data.

Separate from `leagues/` on purpose. The soccer engine is a Dixon-Coles goal
model over 90 minutes; nothing about it transfers to a sport with drives, downs
and 17 games a season. Sharing a package would mean sharing assumptions that do
not hold.

What IS shared is the discipline, because that part is sport-agnostic: pick
freezing before kickoff, append-only records, walk-forward validation on seasons
the model never saw, and a release gate that withholds a market rather than
publishing one it cannot stand behind.
"""
