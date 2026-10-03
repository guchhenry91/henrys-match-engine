"""The jobs the Cloudflare container runs -- the same commands as the GitHub workflows.

Each job is a list of PHASES. A phase is a list of steps; after a phase finishes,
its changed files are saved (a checkpoint) unless a GATE step in it failed. That
mirrors the workflows: leagues.yml commits its quick freeze-and-grade results
before the long refresh, and the refresh is committed only if publish, the
sanity check and the tests pass.

Step: (name, command, kind) where kind is
  "gate"     -- must succeed, or nothing from this phase is saved and the job fails
  "soft"     -- continue-on-error in the workflow: a failure is reported, not fatal
"""
PY = "python -m "

JOBS = {
    "nba": [[
        ("Box scores and bookmaker lines", PY + "scripts.sync_nba_odds", "gate"),
        ("Publish the NBA board", PY + "nba.publish", "gate"),
        ("Tests", PY + "pytest tests/nba tests/oddsapi -q", "gate"),
    ]],
    "mlb": [[
        ("Box scores and bookmaker lines", PY + "scripts.sync_mlb_odds", "gate"),
        ("Publish the MLB board", PY + "mlb.publish", "gate"),
        ("Tests", PY + "pytest tests/mlb tests/oddsapi -q", "gate"),
    ]],
    "nfl": [[
        ("Refresh book prices", PY + "scripts.sync_nfl_odds", "gate"),
        ("Refresh gap-fill book lines", PY + "scripts.sync_nfl_book_lines", "gate"),
        ("Refresh the injury report", PY + "scripts.sync_nfl_injuries", "gate"),
        ("Re-fit model trust", PY + "scripts.fit_nfl_blend", "gate"),
        ("Publish the NFL board", PY + "nfl.publish", "gate"),
        ("Tests", PY + "pytest tests/nfl -q", "gate"),
    ]],
    # Tuesday: the release gate and the vacancy fit, then the normal NFL run.
    "nfl_backtest": [[
        ("Re-run the release gate", PY + "scripts.nfl_backtest", "gate"),
        ("Re-fit the vacancy correction", PY + "scripts.fit_nfl_vacancy", "gate"),
    ]],
    "lock": [[
        ("Freeze picks due to kick off", PY + "scripts.lock_picks", "gate"),
    ]],
    "leagues": [
        [   # the fast lock job (saved on its own, like its own commit)
            ("Freeze picks due to kick off", PY + "scripts.lock_picks", "gate"),
            ("Grade finished NFL games", PY + "scripts.grade_nfl", "soft"),
            ("Sync results", PY + "scripts.sync_results", "soft"),
            ("Apply results to the boards", PY + "scripts.refresh_results", "gate"),
        ],
        [   # the full refresh
            ("Validate data JSON", PY + "scripts.validate_data_json", "gate"),
            ("Refresh Champions League history", PY + "scripts.sync_ucl_history", "soft"),
            ("Champions League gate", PY + "scripts.ucl_backtest", "soft"),
            ("Champions League board", PY + "ucl.publish", "soft"),
            ("Refresh current rosters", PY + "scripts.sync_rosters", "soft"),
            ("Audit rosters", PY + "scripts.roster_integrity_check", "soft"),
            ("Refresh team news", PY + "scripts.sync_team_news", "soft"),
            ("Fetch final scores", PY + "scripts.sync_results", "soft"),
            ("Fetch per-match player stats", PY + "scripts.sync_player_stats", "soft"),
            ("Fetch bet365 scorer prices", PY + "scripts.sync_soccer_prop_odds", "soft"),
            ("Fetch shots / on-target prices", PY + "scripts.sync_soccer_shot_odds", "soft"),
            ("Refresh confirmed lineups", PY + "scripts.sync_lineups", "soft"),
            ("Publish all four leagues", PY + "leagues.publish", "gate"),
            ("Sanity-check the payloads", PY + "scripts.sanity_check", "gate"),
            ("Run tests", PY + "pytest tests/ -q", "gate"),
            ("Send newly-locked picks to Telegram", PY + "scripts.telegram_picks", "soft"),
        ],
    ],
    "health": [[
        ("Run the health check", PY + "scripts.health_check", "soft"),
    ]],
}

# The Odds API switch comes from the Worker (off for shadow runs); default on.
ENV = {"PUBLISH_WORKERS": "4"}   # standard-4 = 4 vCPU, like a GitHub runner
ENV_DEFAULTS = {"ODDS_API_ENABLED": "true"}
