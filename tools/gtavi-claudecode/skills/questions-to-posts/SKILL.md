---
name: questions-to-posts
description: Turn repeated fan questions (Reddit threads and comments on our own posts) into HOW TO post drafts. Use when the user asks what people are asking, wants HOW TO ideas, or says to turn comment questions into posts.
---

# Questions to posts

Finds the questions fans keep asking, checks whether `facts.md` answers them, and drafts HOW TO posts from verified facts only.

## Command

Run from the page folder (where `facts.md` lives):

```
python scripts/questions.py --subs GTA6,GTA,GrandTheftAutoV --days 3 [--comments comments.txt] --out questions.md
```

- `--comments comments.txt`: comments pasted from our own posts, one per line.
- `--facts facts.md` (default), `--ideas idea-bank.md` (default), `--top 25`, `--min-count 1`.
- Uses public Reddit JSON, 1 request per second, waits and retries on 429. If Reddit blocks it, paste comments into `comments.txt` and run again.

## What you get

- `questions.md`: clusters ranked by count then upvotes. Each has the canonical question, count, example links, and either `answer found in facts.md: <line>` or `UNANSWERED: needs an official source`.
- Below the clusters: a HOW TO draft per answered cluster, in the playbook template (`HOW TO [DO THE THING] IN GTA 6`, numbered steps).
- `idea-bank.md`: every unanswered cluster is appended as a research task.

## After the run

1. Read each draft. The script copies steps from `facts.md` lines; it cannot judge order or whether a line is really a step. Fix order, trim to 3 to 5 steps, keep the wording true to the fact.
2. For each unanswered cluster, find an official source and add it to `facts.md` with the link, then rerun.
3. Run `rule-check` on the final text, then post. Same-day turnaround for repeated questions beats polish.

## Rules

1. Facts only from official sources (Rockstar, Take-Two, PlayStation, Xbox), logged in `facts.md` with the link. Recheck anything older than 48 hours.
2. Never invent steps, numbers, dates or menu names. No source, no step. An unanswered question stays a research task.
3. No leaks, rumors, piracy, exploits or "play early" tricks, even if fans ask for them. Skip those clusters.
4. Credit: if a post builds on a fan's clip or find, credit with `🎥 @creator`. Do not quote or name Reddit users; the questions are only a signal of what to cover.
5. Never imply the page is Rockstar. Use "GTA 6".
6. Keep all-caps to the hook line only.
