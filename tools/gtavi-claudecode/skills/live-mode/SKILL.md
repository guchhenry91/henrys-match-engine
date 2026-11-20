---
name: live-mode
description: Record a live YouTube stream with its chat and clip chat spikes in near real time. Use when the user gives a live URL, says "go live mode", or wants clips from a launch stream as it happens.
---

# Live mode

Run:

`python "${CLAUDE_PLUGIN_ROOT}/scripts/live.py" <youtube-live-url> --minutes 60`

It records the stream in chunks with chat and cuts around chat spikes: 15 seconds before and 20 seconds after each spike.

Before starting:

1. Confirm the stream is public and not a leak, pirated copy or "play early" content. If unsure, do not start.
2. Note the streamer's handle for credit.

While it runs:

1. Check new clips in `jobs/` every few minutes. Review each one: pick the best, skip spoilers and nudity.
2. For each keeper, run `hook-writer` then `clip-cutter` (or use the auto-cut if the script produced one), then `rule-check`.
3. Send the human one message with the best 1 to 3 clips (caption, credit, file path). They say "yes" and post. Speed matters, so keep it to one message.

When done: list what was recorded, what was clipped, and log the posts via `learning-loop`. Ask the streamer for permission when they are reachable.
