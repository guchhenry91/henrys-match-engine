# history-data: training history for the model rebuild

Kept OFF `main` on purpose: every Cloudflare job downloads `main` on each run, so
tens of MB of history there would slow every job. The rebuild's training and gate
scripts read this branch (see `docs/REBUILD_PLAN.md` on main). Finished seasons never
change; files here are exports, never edited by hand.

| Folder | Source | What | Made by (on main) |
|---|---|---|---|
| `nfl/` | nflverse (free) | weekly player stats 2012-2025, snap counts 2013-2025, injury reports 2012-2025, `games.csv` (every game since 1999 with spread, total, moneylines, roof, weather, starting QBs), `closing_lines.csv` | downloaded 2026-10-11 |
| `mlb/` | Retrosheet (free) | 2007-2025 batting / pitching with every column, postseason included, pitch counts, handedness, platoon splits per game | `scripts/export_mlb_full_history.py` |
| `leagues/understat/` | Understat (free) | every player's minutes, goals, shots, xG, start/sub in every match, and every shot (penalties marked), 5 leagues, 2017-18 onward | `scripts/export_understat_matches.py` |

NBA full history lives on main (`data-raw/nba/history_full/`, 8 MB) because the live
NBA board reads it.
