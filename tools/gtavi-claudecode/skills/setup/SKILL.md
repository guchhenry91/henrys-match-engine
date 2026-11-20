---
name: setup
description: First-run setup for the GTA 6 clip desk. Use when the user installs the plugin, says "set up", "start", or when ffmpeg, yt-dlp or faster-whisper are missing or the page folder does not exist yet.
---

# Setup

Do the work yourself. Ask the human only what is listed here.

## 1. Install tools (no questions)

Check each with `ffmpeg -version`, `yt-dlp --version`, `python -c "import faster_whisper, numpy"`. Install only what is missing:

- ffmpeg. Windows: `winget install --id Gyan.FFmpeg -e --source winget`. macOS: `brew install ffmpeg`. Linux: `sudo apt install -y ffmpeg`
- Python packages, all systems: `python -m pip install -U -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"` (faster-whisper, numpy, requests and a fresh yt-dlp; the scripts prefer the pip yt-dlp, because an old yt-dlp gets HTTP 403 from YouTube)
- Re-run the pip line once a week: YouTube changes often and yt-dlp updates follow.

If a winget/brew/apt install needs a new terminal for PATH, say so in one line and re-check. Never spend money or add a card.

## 2. Ask at most 4 questions, in ONE numbered message

1. Page name and handle (@).
2. Platforms (for example X, Instagram Reels, TikTok, YouTube Shorts).
3. Niche angle (one lane: tips and hidden spots, fast news, funny moments, cars and customization).
4. Time zone.

If the human already gave an answer, do not ask it again. Skip the message if all four are known.

## 3. Show the plan, ask ONE "yes"

List the folder you will create and the files in it, then ask for a single "yes". After the yes, create (in the current directory, in a folder named after the page):

- `playbook.md` (copy from `${CLAUDE_PLUGIN_ROOT}/skills/playbook/references/playbook.md` and fill the [BRACKETED] fields with the answers)
- `facts.md` (verified facts only, each with an official source link)
- `idea-bank.md`, `watchlist.md`, `log.md`
- `scoreboard.csv` with this header line only:
  `post_id,date,time,platform,pillar,format,hook,length_sec,credit,views_12h,views_24h,views_72h,likes,reposts_or_shares,replies_or_comments,saves_or_bookmarks,followers_at_read,test_name,notes`
- `learning.md` with headings Wins, Losses, Current Rules
- `takedowns.md`
- `campaigns.md` (title line and an empty table: name, platform, rate per 1K views, budget left, rules, link, checked on)
- `payouts.csv` with header `date,campaign,platform,post_url,views,rate_per_1k,est_payout,status` (managed by `scripts/payouts.py`)
- folders `media/official`, `media/raw`, `media/edits`

## 4. Finish

Tell the human, in 3 lines: what was installed, where the folder is, and the next step (`campaign-finder`, then `trend-radar`). Remind them that they approve every post and that nothing is uploaded automatically.
