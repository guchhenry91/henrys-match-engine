---
name: secrets-map
description: Build and post a verified map of GTA 6 hidden spots, rare cars, collectibles, money spots and easter eggs found by creators. Use when the user wants secrets, hidden spots, easter eggs or evergreen "hidden in Leonida" tip posts, or says scan a video for secrets.
---

# Secrets map

Entries live in `secrets.csv` (id, category, name, where, how, source_channel, source_url_with_timestamp, found_on, status, verified_by, notes). Categories: place, money, car, collectible, easter_egg, other. Status: unverified, verified, wrong.

## Commands

Run from the project folder with `.venv\Scripts\python scripts\secrets_map.py ...` (`--csv path` before the subcommand to use another file).

- `scan jobs/<id>`: reads `transcript.json` and `info.json`, finds secret-type mentions (hidden, secret, easter egg, money, cash, stash, collectible, rare car, location, spawn, unlock) and adds them as `unverified` with a quote and a `?t=` link to the source. Duplicates are skipped. Exploit-type mentions are skipped and printed.
- `add --name N --where W --how H --category place --channel C --url U`: manual entry.
- `verify <id> --by "<who/how>"`: only after the spot was checked in game (or confirmed by official material). Needs where, how and source channel filled in.
- `wrong <id>`: the tip did not hold up.
- `posts --out secrets-posts.md`: HOW TO tip and Spotlight drafts per verified entry, plus a "5 hidden spots in Leonida" carousel outline when 3 or more are verified.
- `map --out secrets.md`: table grouped by category.

## Flow

1. `scan` a transcribed video. Open each `?t=` link, read the quote, and rewrite the auto name/where/how in `secrets.csv` into a real tip.
2. Check it in the game. `verify` or `wrong`.
3. `posts`, then run `rule-check` on the drafts before they go out.

## Rules

1. Post verified entries only. Unverified is a lead, not a fact.
2. Credit the finder on every post (`🎥 channel` plus the timestamp link). Never present a creator's find as ours.
3. No glitches, exploits, money glitches, duplication, leaks or "play early" content. The script refuses them; do not work around it.
4. Scan output is a transcript guess. Rewrite the name, where and how by hand; do not post the raw quote.
5. No spoilers for story moments in launch week. Never imply the page is Rockstar. Use "GTA 6". X posts have no hashtags.
