---
name: multilang
description: Turn one finished clip into localized versions (Spanish, Portuguese, German, Russian and more) with translated word-by-word captions, translated hook, optional stock voiceover of the hook, and per-platform captions. Use when the user says "translate the clip", "make it in Spanish", "other languages", "localize", "5 languages".
---

# Multilang

Non-English GTA pages are underserved, so one clip can feed several audiences.

Run after `clip-cutter` (the clip must already exist as `jobs/<id>/edits/clip_<n>.mp4` + `.json`):

`python "${CLAUDE_PLUGIN_ROOT}/scripts/translate.py" jobs/<id> --pick <n> --langs es,pt,de,ru [--voice]`

Output in `jobs/<id>/edits/`: `clip_<n>_<lang>.mp4` per language and one `clip_<n>_captions.md` with X, Instagram and TikTok/Shorts captions per language.

- Layout, hook box and loudness match the original cut. Override with `--layout`, `--hook` if needed.
- `--voice` adds a stock text-to-speech voice (edge-tts: es-ES-AlvaroNeural, pt-BR-AntonioNeural, de-DE-ConradNeural, ru-RU-DmitryNeural) that reads the translated hook only, in the first 2.5 seconds, with the original audio ducked under it.
- Never clone or imitate a real person's or streamer's voice. Stock voices only, and do not present the voiceover as the creator speaking.

Rules:

2. Translate facts, do not invent. Do not add claims, dates, prices or names that are not in the original clip or hook. Machine translation can slip: read each language's captions and hook before the human posts, and fix wrong or odd lines (ask the human if a language is unfamiliar).
3. Run `rule-check` on each language version, same as the original.
4. Post localized versions on separate localized pages or accounts, or as multi-language uploads where the platform supports them (for example YouTube multiple audio tracks and translated titles). Do not spam the same page with five versions of one clip.
5. Captions: X no hashtags; Instagram 3 to 5 hashtags; TikTok and Shorts short plus a few hashtags.
6. Do not upload anything. Show the files and captions to the human and wait for "yes".
