---
name: trend-radar
description: Rank recent GTA 6 videos by views per hour to see what is trending now. Use when the user asks what is hot, what to clip today, or wants fresh source videos.
---

# Trend radar

Run:

`python "${CLAUDE_PLUGIN_ROOT}/scripts/trends.py" --query "gta 6" --hours 24 --limit 30`

Change `--query` for other angles ("gta 6 map", "gta 6 trailer", a streamer name). Change `--hours` for a wider window.

Then:

1. Show the top 10 as a short table: title, channel, views, views per hour, age, link.
2. Pick 3 that fit the page's niche angle. Prefer official sources and creators who allow sharing.
3. Flag any that look like leaks or "play early": drop them and say so.
4. Offer the next step for the picks: `moment-finder <url> --top 8`.

If the script errors, show the last error line and fix the cause (missing yt-dlp, network). Do not invent a list.
