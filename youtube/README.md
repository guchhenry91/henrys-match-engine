# YouTube

The YouTube agent skill pack, kept here so it loads in every Claude Code session in this repo.
It has nothing to do with the match engine and touches none of its files.

- **Source:** https://github.com/Jakeschincariol/youtube-agent-skill (MIT, see `LICENSE`),
  commit `a2feb21` (2026-09-16). Reviewed before it was added: the six Python tools are
  standard library only, make no network calls and read only the files you pass them; no skill
  publishes anything.
- **The skills** live in `.claude/skills/yt-*`, because that is the only place Claude Code looks
  for project skills. The one local change: they read the voice profile from `youtube/voice.md`
  here instead of `~/.claude/youtube/voice.md`, which does not survive a cloud session.
- **`voice.md`**: fill it in once (or send Claude three of your videos and ask it to write it).
  Every skill reads it.

| command | what it does |
| --- | --- |
| `/yt-script` | an idea into a script; five hooks scored against 21 formulas |
| `/yt-package` | title and thumbnail linted as one pairing |
| `/yt-edit` | a transcript into an edit decision list (dead air, filler, retakes) |
| `/yt-retention` | a YouTube Studio retention export: where people left, and why |
| `/yt-shorts` | the Shorts hiding inside a long video |
| `/yt-chapters` | chapters that pass YouTube's own rules |
| `/yt-seo` | description, tags, the three queries it should win |
| `/yt-comment` | comments triaged, replies in your voice |
| `/yt-plan` | a week of uploads sized to the hours you have |
| `/yt-viral` | what's working in a niche, ranked by multiple over each channel's median |
| `/yt-audit` | the whole channel, ending in ONE fix |

To update: re-clone the source, copy `skills/yt-*` over `.claude/skills/`, and re-apply the
voice.md path change.
