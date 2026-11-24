---
name: twitch-reaction
description: Watch the biggest GTA 6 Twitch streamer and make reaction clips: face on top, game below, big title. Use when the user wants streamer reaction clips, "face on top" clips, Twitch clips, or asks who is streaming GTA 6 right now.
---

# Twitch reaction clips

Face on top, game below, a big title on the seam. The format of most viral GTA 6 streamer clips.

## 1. Find the streamer (no Twitch account or key needed)

`python "${CLAUDE_PLUGIN_ROOT}/scripts/streams.py" --query "gta 6"` lists the top live Grand Theft Auto VI streams on Twitch and YouTube, ranked by viewers. Pick a streamer who runs a clipping campaign (check `campaigns.md`), or one whose clips you have permission to post.

## 2. Get the moment

- Live: `python "${CLAUDE_PLUGIN_ROOT}/scripts/live.py" https://twitch.tv/<login> --minutes 60` records the stream, reads Twitch chat anonymously and cuts a clip every time chat explodes.
- After the fact: `python "${CLAUDE_PLUGIN_ROOT}/scripts/moments.py" <twitch VOD or clip URL> --top 8` scores the VOD and writes `candidates.md`.

## 3. Cut the reaction clip

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/cut.py" jobs/<id> --pick 1 --layout react --cam auto \
  --title "Speed REALIZED why GTA 6 took ROCKSTAR 13 YEARS" --title-style box --captions bubble
```

- `--cam auto` finds the streamer's webcam itself (face detection plus the webcam frame around it). It prints where it found the cam. If it finds no stable face it falls back to the blur layout and says so. Check the first frame once; if the crop is wrong, pass `--cam x:y:w:h` fractions.
- `--title-style box`: white band with bold black text on the seam. `--title-style outline`: white text with a thick black outline across the top of the game, for short titles ("Fighting Cops in GTA 6").
- `--captions bubble`: white speech bubble ("Oh my god!"). `word`: word-by-word captions. `none`: title only.
- Write the title with `hook-writer` (reaction titles section).

## Rules

- Clip a streamer only inside their campaign or with permission. Credit them in the post caption, not on the video.
- Titles describe what happened. Never invent a quote or an opinion.
- Run `rule-check` before posting. No spoilers in the title during launch week.
