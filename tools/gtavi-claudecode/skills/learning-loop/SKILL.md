---
name: learning-loop
description: Log posts and learn from real numbers using scoreboard.csv, learning.md and payouts.csv. Use after a post goes out, at 12/24/72 hour reads, for the daily or weekly review, or when the user asks what is working or what they earned.
---

# Learning loop

Files in the page folder: `scoreboard.csv` (one row per post), `learning.md` (Wins, Losses, Current Rules), `payouts.csv` (real payouts only), `log.md`.

## Rules

1. Real numbers only. Ask the human to paste metrics or a screenshot. If a number is not available, leave it blank. Never guess.
2. Log every post when it goes out: post_id, date, time, platform, pillar, format, hook, length_sec, credit, test_name.
3. Reads at 12h, 24h and 72h fill views_12h, views_24h, views_72h, then likes, shares, replies, saves, followers_at_read.
4. One test per day. Change one variable only (hook style, length, posting time, caption length, still vs clip). Name it in `test_name`.
5. No verdict under 24 hours. Prefer the 72h read if the post is still moving.
6. `payouts.csv`: record only amounts the campaign or platform actually shows. Never project earnings.

## Daily (evening)

Best and worst post, one lesson written as a rule. Append to `learning.md`. Pick tomorrow's one test.

## Weekly

Top 3 and bottom 3 posts, median views by pillar, by time slot and by platform. Double down on what is above median. Rework or cut what stayed below median for a full week. Pick next week's tests. Update the customized `playbook.md` with what changed. Send the human a short summary.

Late-night reads: say what is still missing instead of filling it in.
