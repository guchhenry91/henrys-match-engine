"""One clip, many languages.

Translates the spoken lines (word-by-word captions) and the hook of an already cut clip, then renders
jobs/<id>/edits/clip_<n>_<lang>.mp4 with the same layout as cut.py, plus clip_<n>_captions.md with
per-platform captions in each language.

usage: python translate.py jobs/<id> --pick <n> --langs es,pt,de,ru [--voice] [--layout ...] [--hook "..."]

--voice adds a stock text-to-speech voice (edge-tts, Microsoft neural stock voices) that speaks the
translated hook only, over the first 2.5 s, with the original audio ducked under it. It never clones
or imitates a real person's or streamer's voice.
"""
import argparse, asyncio, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ffprobe, load_json, run, tool
from cut import CAP_Y, build_ass, video_filter

VOICES = {"es": "es-ES-AlvaroNeural", "pt": "pt-BR-AntonioNeural", "de": "de-DE-ConradNeural",
          "ru": "ru-RU-DmitryNeural", "fr": "fr-FR-HenriNeural", "it": "it-IT-DiegoNeural",
          "pl": "pl-PL-MarekNeural", "tr": "tr-TR-AhmetNeural", "ja": "ja-JP-KeitaNeural",
          "ko": "ko-KR-InJoonNeural", "ar": "ar-SA-HamedNeural", "hi": "hi-IN-MadhurNeural"}
HOOK_SECONDS = 2.5
HASHTAGS = ["#GTA6", "#GTAVI", "#gaming", "#rockstargames", "#gta"]


def translator(lang):
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        sys.exit("deep-translator missing. Run: pip install -r requirements.txt")
    return GoogleTranslator(source="auto", target=lang)


def tr(t, text):
    text = (text or "").strip()
    if not text:
        return ""
    try:
        return (t.translate(text) or text).strip()
    except Exception as e:  # network / rate limit: keep original rather than crash the whole batch
        print(f"warning: translate failed ({e}); keeping original text", file=sys.stderr)
        return text


def sentences(words):
    """Split clip words into sentences (end punctuation or a pause > 0.8 s)."""
    out, cur = [], []
    for w in words:
        if cur and w["s"] - cur[-1]["e"] > 0.8:
            out.append(cur); cur = []
        cur.append(w)
        if w["w"][-1:] in ".?!":
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def translate_words(t, words):
    """Translate each sentence, then spread the translated words evenly over the sentence's timing."""
    res = []
    for sent in sentences(words):
        new = tr(t, " ".join(w["w"] for w in sent)).split()
        if not new:
            continue
        a, b = sent[0]["s"], sent[-1]["e"]
        step = (b - a) / len(new)
        res += [{"w": x, "s": a + i * step, "e": a + (i + 1) * step} for i, x in enumerate(new)]
    return res


def make_voice(lang, text, dest):
    try:
        import edge_tts
    except ImportError:
        sys.exit("edge-tts missing. Run: pip install -r requirements.txt")
    voice = VOICES.get(lang)
    if not voice:
        print(f"no stock voice configured for '{lang}', skipping --voice", file=sys.stderr)
        return None
    asyncio.run(edge_tts.Communicate(text, voice).save(str(dest)))
    return dest


def render(src, s0, e0, layout, cam, ass, mp4, voice_mp3=None):
    v = video_filter(layout, cam, ass)
    cmd = [tool("ffmpeg"), "-y", "-v", "error", "-ss", f"{s0:.2f}", "-to", f"{e0:.2f}", "-i", src]
    if voice_mp3:
        dur = float(ffprobe(voice_mp3)["format"]["duration"])
        tempo = f"atempo={min(2.0, dur / HOOK_SECONDS):.3f}," if dur > HOOK_SECONDS else ""
        cmd += ["-i", str(voice_mp3)]
        a = (f"[0:a]volume=0.2:enable='lt(t,{HOOK_SECONDS})'[bed];"
             f"[1:a]{tempo}atrim=0:{HOOK_SECONDS},afade=t=out:st={HOOK_SECONDS - 0.15}:d=0.15,apad[tts];"
             f"[bed][tts]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[ao]")
        cmd += ["-filter_complex", v + ";" + a, "-map", "[vo]", "-map", "[ao]"]
    else:
        cmd += ["-filter_complex", v, "-map", "[vo]", "-map", "0:a?", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-ar", "48000", "-movflags", "+faststart", str(mp4)]
    run(cmd)


def platform_block(lang, hook, credit, url):
    line = hook or ""
    cred = ""
    src = f"\n{url}" if url else ""
    x = "\n".join(p for p in [line, cred] if p) + src
    ig = "\n".join(p for p in [line, cred, " ".join(HASHTAGS[:4])] if p)
    tk = "\n".join(p for p in [line, cred, " ".join(HASHTAGS[:3])] if p)
    return (f"## {lang}\n\n**X** (no hashtags)\n\n{x}\n\n**Instagram** (4 hashtags)\n\n{ig}\n\n"
            f"**TikTok / YouTube Shorts**\n\n{tk}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--pick", type=int, default=1)
    ap.add_argument("--langs", default="es,pt,de,ru")
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--layout", choices=["fill", "blur", "facecam"])
    ap.add_argument("--cam"); ap.add_argument("--hook")
    a = ap.parse_args()
    d = Path(a.job); out = d / "edits"
    side = out / f"clip_{a.pick}.json"
    if not side.exists():
        sys.exit(f"{side} not found. Cut the clip first with cut.py --pick {a.pick}.")
    meta = load_json(side)
    cands = load_json(d / "candidates.json")
    s0, e0 = meta["start"], meta["end"]
    layout = a.layout or meta.get("layout", "blur")
    cam = a.cam or "0.70:0.68:0.30:0.32"
    hook = a.hook if a.hook is not None else meta.get("hook", "")
    credit = ""
    words = [w for w in load_json(d / "transcript.json") if s0 <= w["s"] < e0 and w["w"]]
    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    caps = ["# Captions, one clip in several languages\n",
            f"Source: {meta.get('source_title') or ''} ({meta.get('source_channel') or ''})\n",
            "Translations are machine translated. Read each one before posting.\n"]
    for lang in langs:
        t = translator(lang)
        h = tr(t, hook) if hook else ""
        tw = translate_words(t, words)
        ass = out / f"clip_{a.pick}_{lang}.ass"
        ass.write_text(build_ass(tw, s0, e0, h, CAP_Y[layout]), encoding="utf-8")
        mp4 = out / f"clip_{a.pick}_{lang}.mp4"
        mp3 = None
        if a.voice and h:
            mp3 = make_voice(lang, h, Path(tempfile.gettempdir()) / f"clip_{a.pick}_{lang}_hook.mp3")
        render(cands["source"], s0, e0, layout, cam, ass, mp4, mp3)
        caps.append(platform_block(lang, h, credit, meta.get("source_url")))
        print(mp4)
    (out / f"clip_{a.pick}_captions.md").write_text("\n".join(caps), encoding="utf-8")
    print(out / f"clip_{a.pick}_captions.md")


if __name__ == "__main__":
    main()
