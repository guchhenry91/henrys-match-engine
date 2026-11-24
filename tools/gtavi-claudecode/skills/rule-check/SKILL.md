---
name: rule-check
description: Check a finished clip against technical specs, campaign rules and banned content before posting. Use before every upload, or when the user asks "is this ok to post".
---

# Rule check

Run:

`python "${CLAUDE_PLUGIN_ROOT}/scripts/check.py" <clip.mp4> --campaign campaigns.md`

It checks: length, 1080x1920, audio present, the campaign's rules from `campaigns.md`, and banned content (leaks, "play early", piracy).

Then:

1. Show PASS or FAIL per check in a short list.
2. On FAIL: fix the cause (re-cut with `clip-cutter`, add the credit, trim) and re-run. Do not argue with a failed check.
3. Read the caption yourself too. It must credit the creator in the text, no invented facts, no "official" wording, and match the platform's caption rules in the playbook.
4. The script cannot judge everything. If the campaign rules are `unknown` or missing, ask the human to paste them.
5. Only after all checks pass, send the human the clip, caption and credit and wait for "yes". The human posts it. Never auto-upload.
