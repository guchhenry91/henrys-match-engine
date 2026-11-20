"""Secrets map: collect hidden spots, cars, collectibles and easter eggs from creator videos, verify them, then post.

Entries live in secrets.csv. Nothing is posted unless its status is "verified".
Exploits, money glitches, duplication tricks, leaks and "play early" content are never collected.

usage:
  python secrets_map.py scan jobs/<id>                 propose unverified entries from the transcript
  python secrets_map.py add --name N --where W --how H --category place --channel C --url U
  python secrets_map.py verify <id> --by "played it myself on 2026-11-20"
  python secrets_map.py wrong <id>
  python secrets_map.py posts --out secrets-posts.md   tip posts + carousel from VERIFIED entries only
  python secrets_map.py map --out secrets.md           table grouped by category
"""
import argparse, csv, datetime as dt, difflib, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, load_json

COLS = ["id", "category", "name", "where", "how", "source_channel", "source_url_with_timestamp",
        "found_on", "status", "verified_by", "notes"]
CATS = ["place", "money", "car", "collectible", "easter_egg", "other"]
STATUSES = ["unverified", "verified", "wrong"]

# secret-type mentions, per category (first match order decides the category)
CAT_WORDS = [
    ("easter_egg", r"easter eggs?|hidden detail|reference to|secret message"),
    ("car", r"rare cars?|hidden cars?|secret cars?|unique vehicle|spawns?\b|car location"),
    ("collectible", r"collectibles?|collect(?:ing)? all|hidden packages?|stunt jumps?|treasure|souvenirs?"),
    ("money", r"cash|money|stash|safe code|payout|robbery loot"),
    ("place", r"hidden (?:place|spot|location|area|room|cave|island)|secret (?:place|spot|location|area|room|cave|island)|location|unlocks?"),
]
KEYWORDS = re.compile(r"\b(hidden|secrets?|easter eggs?|money|cash|stash|collectibles?|rare cars?|locations?|spawns?|unlocks?|glitch[- ]free)\b", re.I)
# never collected: exploits, glitches, dupes, leaks, early play
BANNED = re.compile(r"glitch(?![- ]free)|\bexploit|\bdupe|duplicat|infinite (?:money|cash)|unlimited (?:money|cash)|money (?:glitch|hack|exploit)|"
                    r"\bleak(?:s|ed)?\b|play(?:ing)? early|early access|before (?:the )?release|pre-?release|\bcheats?\b|\bmods?\b|\bhack", re.I)
DEDUPE_RATIO = 0.78


def csv_path(a):
    return Path(a.csv) if a.csv else ROOT / "secrets.csv"


def load_rows(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return [{c: (r.get(c) or "") for c in COLS} for r in csv.DictReader(f)]


def save_rows(p, rows):
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def next_id(rows):
    nums = [int(r["id"][1:]) for r in rows if re.fullmatch(r"S\d+", r["id"])]
    return f"S{(max(nums) if nums else 0) + 1:03d}"


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def similar(a, b):
    ka, kb = norm(f"{a['name']} {a['where']}"), norm(f"{b['name']} {b['where']}")
    return bool(ka and kb) and difflib.SequenceMatcher(None, ka, kb).ratio() >= DEDUPE_RATIO


def link_at(url, sec):
    if not url.startswith("http"):
        return f"{url}#t={int(sec)}"
    return f"{url}{'&' if '?' in url else '?'}t={int(sec)}s"


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def categorize(text):
    for cat, pat in CAT_WORDS:
        if re.search(pat, text, re.I):
            return cat
    return "other"


def find_row(rows, rid):
    for r in rows:
        if r["id"].lower() == rid.lower():
            return r
    sys.exit(f"no entry {rid}")


# ---- scan -------------------------------------------------------------------------------------

def cmd_scan(a):
    jd = Path(a.job)
    words = load_json(jd / "transcript.json")
    info = load_json(jd / "info.json") if (jd / "info.json").exists() else {}
    title = info.get("title", "")
    channel = info.get("channel") or info.get("uploader") or ""
    url = info.get("webpage_url", "")
    hits = [i for i, w in enumerate(words) if KEYWORDS.search(w["w"])]
    # group hits that sit within 12 s of each other
    groups, cur = [], []
    for i in hits:
        if cur and words[i]["s"] - words[cur[-1]]["s"] > 12:
            groups.append(cur); cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    p = csv_path(a)
    rows = load_rows(p)
    today = dt.date.today().isoformat()
    added = skipped = dupes = 0
    for g in groups:
        t0, t1 = words[g[0]]["s"], words[g[-1]]["e"]
        lo = next((k for k, w in enumerate(words) if w["s"] >= t0 - 6), g[0])
        hi = max(k for k, w in enumerate(words) if w["s"] <= t1 + 10)
        quote = " ".join(w["w"] for w in words[lo:hi + 1]).strip()
        if BANNED.search(quote):
            skipped += 1
            print(f"skip {mmss(t0)} (exploit/glitch/leak/early-play wording): {quote[:90]}")
            continue
        cat = categorize(quote)
        kw = sorted({KEYWORDS.search(words[i]["w"]).group(1).lower() for i in g})
        name = f"{cat.replace('_', ' ')} mention ({', '.join(kw[:3])}) at {mmss(t0)}"
        row = {"id": "", "category": cat, "name": name, "where": quote[:300], "how": "",
               "source_channel": channel, "source_url_with_timestamp": link_at(url, max(0, t0 - 3)) if url else "",
               "found_on": today, "status": "unverified", "verified_by": "",
               "notes": f"auto-scan of '{title}' at {mmss(t0)}; fill in real name/where/how before verifying"}
        if any(similar(row, r) for r in rows):
            dupes += 1
            continue
        row["id"] = next_id(rows)
        rows.append(row)
        added += 1
    save_rows(p, rows)
    print(f"{added} new unverified, {dupes} duplicates, {skipped} skipped (exploit-type) -> {p}")


# ---- add / verify / wrong ---------------------------------------------------------------------

def cmd_add(a):
    text = " ".join([a.name, a.where, a.how, a.notes])
    if BANNED.search(text):
        sys.exit("refused: exploits, glitches, dupes, leaks and play-early content are not collected")
    p = csv_path(a)
    rows = load_rows(p)
    row = {"id": "", "category": a.category, "name": a.name, "where": a.where, "how": a.how,
           "source_channel": a.channel, "source_url_with_timestamp": a.url,
           "found_on": dt.date.today().isoformat(), "status": "unverified", "verified_by": "", "notes": a.notes}
    dup = next((r for r in rows if similar(row, r)), None)
    if dup:
        sys.exit(f"looks like a duplicate of {dup['id']} ({dup['name']}); not added")
    row["id"] = next_id(rows)
    rows.append(row)
    save_rows(p, rows)
    print(f"added {row['id']} (unverified)")


def cmd_verify(a):
    p = csv_path(a)
    rows = load_rows(p)
    r = find_row(rows, a.id)
    if r["status"] == "wrong":
        sys.exit(f"{r['id']} is marked wrong; edit the csv by hand if that is a mistake")
    if not (r["where"] and r["how"] and r["source_channel"]):
        sys.exit(f"{r['id']} needs where, how and source_channel filled in before it can be verified")
    r["status"], r["verified_by"] = "verified", a.by
    save_rows(p, rows)
    print(f"{r['id']} verified by {a.by}")


def cmd_wrong(a):
    p = csv_path(a)
    rows = load_rows(p)
    r = find_row(rows, a.id)
    r["status"] = "wrong"
    save_rows(p, rows)
    print(f"{r['id']} marked wrong")


# ---- posts / map ------------------------------------------------------------------------------

def credit(r):
    return f"🎥 {r['source_channel']}" + (f"\n{r['source_url_with_timestamp']}" if r["source_url_with_timestamp"] else "")


def cmd_posts(a):
    rows = [r for r in load_rows(csv_path(a)) if r["status"] == "verified"]
    ready = [r for r in rows if r["where"] and r["how"] and r["source_channel"] and not BANNED.search(f"{r['name']} {r['where']} {r['how']}")]
    left = len(rows) - len(ready)
    out = ["# Secrets posts (verified entries only)", "",
           "Check each tip once more in game before posting. Credit line stays on every post. No hashtags on X.", ""]
    if not ready:
        out.append("No verified entries yet. Run `secrets_map.py verify <id> --by ...` after you checked one in game.")
    for r in ready:
        steps = [s.strip() for s in re.split(r"(?<=[.;])\s+|\n", r["how"]) if s.strip()][:4]
        body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) or r["how"]
        out += [f"## {r['id']} {r['name']} ({r['category']})", "",
                "HOW TO tip:", "", "```",
                f"HOW TO FIND {r['name'].upper()} IN GTA 6", "", f"Where: {r['where']}", "", body, "", credit(r), "```", "",
                "Spotlight:", "", "```",
                f"{r['name']}: {r['where']}", "", credit(r), "```", ""]
    if len(ready) >= 3:
        pick = ready[:5]
        out += [f"## Carousel outline: {len(pick)} hidden spots in Leonida", "",
                f"- Slide 1 (cover): {len(pick)} hidden spots in Leonida, all checked in game",
                *[f"- Slide {i + 1}: {r['name']}. {r['where']} Credit: {r['source_channel']}" for i, r in enumerate(pick)],
                f"- Slide {len(pick) + 2} (end): Which one did you find first? Credit to the creators on every slide.", "",
                "Caption (X, no hashtags):", "", "```",
                f"{len(pick)} hidden spots in Leonida", "",
                *[f"- {r['name']} ({r['source_channel']})" for r in pick], "```", ""]
    elif ready:
        out += ["_Carousel needs at least 3 verified entries._", ""]
    if left:
        out += [f"_{left} verified entries skipped: missing where/how/channel or exploit wording._"]
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"{len(ready)} verified entries -> {a.out}")


def cmd_map(a):
    rows = load_rows(csv_path(a))
    out = ["# GTA 6 secrets map", "", "Only verified entries are safe to post. Unverified and wrong are listed for tracking.", ""]
    esc = lambda s: s.replace("|", "/").replace("\n", " ")
    for cat in CATS:
        grp = [r for r in rows if r["category"] == cat]
        if not grp:
            continue
        out += [f"## {cat.replace('_', ' ').title()} ({len(grp)})", "",
                "| id | name | where | how | status | verified by | source |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in sorted(grp, key=lambda r: (STATUSES.index(r["status"]) if r["status"] in STATUSES else 9, r["id"])):
            src = f"[{esc(r['source_channel']) or 'link'}]({r['source_url_with_timestamp']})" if r["source_url_with_timestamp"] else esc(r["source_channel"])
            out.append(f"| {r['id']} | {esc(r['name'])} | {esc(r['where'])} | {esc(r['how'])} | {r['status']} | {esc(r['verified_by'])} | {src} |")
        out.append("")
    if len(out) == 4:
        out.append("No entries yet.")
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"{len(rows)} entries -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="path to secrets.csv (default: ./secrets.csv)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("job"); s.set_defaults(f=cmd_scan)
    s = sub.add_parser("add")
    s.add_argument("--name", required=True); s.add_argument("--where", required=True); s.add_argument("--how", default="")
    s.add_argument("--category", choices=CATS, default="other"); s.add_argument("--channel", default="")
    s.add_argument("--url", default=""); s.add_argument("--notes", default=""); s.set_defaults(f=cmd_add)
    s = sub.add_parser("verify"); s.add_argument("id"); s.add_argument("--by", required=True); s.set_defaults(f=cmd_verify)
    s = sub.add_parser("wrong"); s.add_argument("id"); s.set_defaults(f=cmd_wrong)
    s = sub.add_parser("posts"); s.add_argument("--out", default="secrets-posts.md"); s.set_defaults(f=cmd_posts)
    s = sub.add_parser("map"); s.add_argument("--out", default="secrets.md"); s.set_defaults(f=cmd_map)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
