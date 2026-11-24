"""Questions into posts: collect fan questions, cluster them, check facts.md, draft HOW TO posts.

python questions.py --subs GTA6,GTA,GrandTheftAutoV --days 3 [--comments comments.txt] --out questions.md
"""
import argparse, difflib, json, re, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = "gta6-clip-desk/1.0 (questions-to-posts; read-only public JSON; contact via github Bodila51/gtavi-claudecode)"
STARTERS = ("how", "where", "what", "can", "is", "does", "why", "when")
STOP = set("""a an the to of in on at for and or is are was were be been do does did can could would should will
i you we they he she it my your our this that these those there here with about as by from if any
gta gta6 gta5 vi 6 5 game games get got have has had im ive its dont not no yes so just really""".split())
_last = [0.0]


def fetch(url):
    """GET JSON, 1 request/sec, retry on 429 with backoff."""
    for attempt in range(4):
        wait = 1.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                delay = int(e.headers.get("Retry-After", "") or 0) or 10 * (attempt + 1)
                print(f"429 on {url}, waiting {delay}s", file=sys.stderr)
                time.sleep(delay)
                continue
            print(f"HTTP {e.code} on {url}, skipped", file=sys.stderr)
            return None
        except Exception as e:
            print(f"error on {url}: {e}, skipped", file=sys.stderr)
            return None
    print(f"gave up on {url} after repeated 429", file=sys.stderr)
    return None


def is_question(text):
    t = text.strip().lower()
    if len(t) < 10 or len(t) > 300:
        return False
    return t.endswith("?") or t.startswith(STARTERS)


def sentences(text):
    """Split a body into candidate lines; keep those that are questions."""
    out = []
    for line in re.split(r"[\n\r]+|(?<=[.!?])\s+", text):
        line = re.sub(r"\s+", " ", line).strip()
        if is_question(line):
            out.append(line)
    return out


def tokens(s):
    s = re.sub(r"https?://\S+", " ", s.lower())
    return {w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP and len(w) > 1}


def similar(a, b):
    ta, tb = a["tok"], b["tok"]
    if ta and tb and len(ta & tb) / len(ta | tb) >= 0.5:
        return True
    return difflib.SequenceMatcher(None, a["norm"], b["norm"]).ratio() >= 0.75


def collect_reddit(subs, days):
    cutoff = time.time() - days * 86400
    items, posts = [], []
    for sub in subs:
        data = fetch(f"https://www.reddit.com/r/{sub}/new.json?limit=100")
        if not data:
            continue
        for ch in data.get("data", {}).get("children", []):
            d = ch["data"]
            if d.get("created_utc", 0) < cutoff:
                continue
            link = "https://www.reddit.com" + d.get("permalink", "")
            posts.append({"id": d["id"], "score": d.get("score", 0), "link": link, "n": d.get("num_comments", 0)})
            for q in ([d.get("title", "")] if is_question(d.get("title", "")) else []) + sentences(d.get("selftext", "")):
                items.append({"text": q, "score": d.get("score", 0), "link": link, "src": f"r/{sub}"})
    top = sorted((p for p in posts if p["n"] > 0), key=lambda p: p["score"], reverse=True)[:8]
    for p in top:
        data = fetch(f"https://www.reddit.com/comments/{p['id']}.json?limit=200&depth=1")
        if not data or len(data) < 2:
            continue
        for ch in data[1].get("data", {}).get("children", []):
            if ch.get("kind") != "t1":
                continue
            d = ch["data"]
            for q in sentences(d.get("body", "")):
                items.append({"text": q, "score": d.get("score", 0),
                              "link": "https://www.reddit.com" + d.get("permalink", p["link"]), "src": "reddit comment"})
    return items


def collect_file(path):
    items = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        for q in sentences(line) or ([line] if is_question(line) else []):
            items.append({"text": q, "score": 0, "link": "own post comment", "src": "comments.txt"})
    return items


def cluster(items):
    for it in items:
        it["norm"] = " ".join(sorted(tokens(it["text"]))) or it["text"].lower()
        it["tok"] = tokens(it["text"])
    clusters = []
    for it in items:
        for c in clusters:
            if similar(it, c["items"][0]):
                c["items"].append(it)
                break
        else:
            clusters.append({"items": [it]})
    for c in clusters:
        c["count"] = len(c["items"])
        c["votes"] = sum(max(i["score"], 0) for i in c["items"])
        # canonical = shortest question that still ends with ? else the highest scored
        qs = sorted(c["items"], key=lambda i: (not i["text"].endswith("?"), -i["score"], len(i["text"])))
        c["question"] = qs[0]["text"]
        c["links"] = list(dict.fromkeys(i["link"] for i in c["items"] if i["link"].startswith("http")))[:3]
    return sorted(clusters, key=lambda c: (c["count"], c["votes"]), reverse=True)


def load_facts(path):
    p = Path(path)
    if not p.exists():
        return []
    lines = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip().lstrip("-*# ").strip()
        if len(s) > 15 and not s.startswith(("|---", "```")):
            lines.append(s)
    return lines


def match_facts(c, facts):
    qt = tokens(c["question"])
    for i in c["items"][1:]:
        qt |= tokens(i["text"])
    qt = tokens(c["question"]) or qt
    scored = []
    for f in facts:
        ov = len(qt & tokens(f))
        if ov >= 2 or (ov == 1 and len(qt) == 1):
            scored.append((ov, f))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:4]]


def howto_title(q):
    t = q.strip().rstrip("?").strip()
    t = re.sub(r"^(how (do|can|could|would|should) (i|we|you|one)|how to|how)\s+", "", t, flags=re.I)
    t = re.sub(r"\b(in|on|for)? ?gta ?(6|vi)\b", "", t, flags=re.I).strip(" ,.")
    return f"HOW TO {t.upper()} IN GTA 6" if t else "HOW TO [DO THE THING] IN GTA 6"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--subs", default="GTA6,GTA,GrandTheftAutoV")
    ap.add_argument("--days", type=float, default=3)
    ap.add_argument("--comments", help="text file of pasted comments from our own posts, one per line")
    ap.add_argument("--facts", default="facts.md")
    ap.add_argument("--ideas", default="idea-bank.md")
    ap.add_argument("--out", default="questions.md")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--top", type=int, default=25, help="max clusters to report")
    a = ap.parse_args()

    items = collect_reddit([s.strip() for s in a.subs.split(",") if s.strip()], a.days)
    if a.comments:
        if Path(a.comments).exists():
            items += collect_file(a.comments)
        else:
            print(f"comments file not found: {a.comments}", file=sys.stderr)
    if not items:
        sys.exit("No questions found (Reddit empty, blocked or rate limited). Try again later or pass --comments.")

    clusters = [c for c in cluster(items) if c["count"] >= a.min_count][: a.top]
    facts = load_facts(a.facts)
    if not facts:
        print(f"warning: {a.facts} missing or empty, every cluster will be UNANSWERED", file=sys.stderr)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = [f"# Questions report {today}", "",
           f"{len(items)} questions from {a.subs} (last {a.days:g} days){' + ' + a.comments if a.comments else ''}, "
           f"{len(clusters)} clusters. Ranked by count, then upvotes.", ""]
    answered, unanswered = [], []
    for n, c in enumerate(clusters, 1):
        hits = match_facts(c, facts)
        out += [f"## {n}. {c['question']}", f"- count: {c['count']}, upvotes: {c['votes']}"]
        out += [f"- example: {l}" for l in c["links"]] or ["- example: (own comments)"]
        if hits:
            out.append(f"- answer found in facts.md: {hits[0]}")
            answered.append((c, hits))
        else:
            out.append("- UNANSWERED: needs an official source")
            unanswered.append(c)
        out.append("")

    out += ["# HOW TO drafts (answered clusters)", "",
            "Steps are copied from facts.md lines only. Edit wording, keep every fact and its source. "
            "Run rule-check before posting. Drop a step if it is not a real step.", ""]
    if not answered:
        out += ["None yet: no question matched facts.md.", ""]
    for c, hits in answered:
        out += [f"### {c['question']}", "```", howto_title(c["question"]), ""]
        for i, h in enumerate(hits[:5], 1):
            out.append(f"{i}. {h}")
        out += ["", "Source: see the link on each facts.md line", "```", ""]

    Path(a.out).write_text("\n".join(out), encoding="utf-8")

    if unanswered:
        ib = Path(a.ideas)
        old = ib.read_text(encoding="utf-8") if ib.exists() else "# Idea bank\n"
        add = [f"\n## Research tasks from questions ({today})"]
        for c in unanswered:
            add.append(f"- [ ] Find an official source (Rockstar, Take-Two, PlayStation, Xbox) for: \"{c['question']}\" "
                       f"({c['count']} asks). Add to facts.md with link, then make a HOW TO post.")
        ib.write_text(old.rstrip("\n") + "\n" + "\n".join(add) + "\n", encoding="utf-8")

    print(f"{len(clusters)} clusters: {len(answered)} answered, {len(unanswered)} unanswered -> {a.out}"
          + (f", research tasks -> {a.ideas}" if unanswered else ""))


if __name__ == "__main__":
    main()
