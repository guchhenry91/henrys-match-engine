---
name: stream-finder
description: Find live YouTube streams about GTA 6 ranked by concurrent viewers. Use when the user asks who is streaming now, wants live moments to clip, or before live-mode.
---

# Stream finder

Run:

`python "${CLAUDE_PLUGIN_ROOT}/scripts/streams.py" --query "gta 6"`

Then:

1. Show the top 10 live streams: channel, title, concurrent viewers, link.
2. Skip streams that are clearly leaks, early-access tricks or pirated copies. Say why in one line.
3. Recommend 1 to 3 streams. Prefer official channels and creators who allow sharing.
4. Offer the next step: `live-mode <url> --minutes 60` to record and clip chat spikes, or `moment-finder` on a finished VOD.

Viewer counts are what the script returns at that moment. Do not guess or round them up.
