"""Tamper-evident pick logs: each frozen pick is sealed into a hash chain.

THE IDEA (from edge-picks, 2026-10-23). The picks logs are append-only by rule, but
nothing proved it: a frozen line or probability could be rewritten and the record
would never show it. Now every frozen pick gets a seal -- a SHA-256 over the fields
that define the pick, chained to the seal before it -- in a file beside its log
(`<log>.seal.json`). `verify` recomputes the chain; the health check fails on any
pick whose sealed fields changed, or that vanished, after it was sealed.

WHAT IS SEALED is the pick itself (who, what, which side, the line, the stated chance,
the price it froze on, when it froze) -- never the fields added later by grading,
closing prices or voiding, and never the start time, which a reschedule may correct.
A field absent when the pick was sealed is not covered, so a price stamped later
(soccer closing odds "where none was frozen") does not break the chain.

A RE-LOCK IS A NEW VERSION, NOT AN EDIT. leagues.picks.release_moved_lock frees a pick
whose fixture time moved (before the game, ungraded) and it is frozen again later with
a new lock time; that version gets its own seal and the old one is "superseded". A
sealed pick that is simply missing is reported apart from an edited one: it may be a
released lock awaiting its new kickoff.

Picks frozen before 2026-10-23 are sealed on the first run after it: the chain
proves nothing changed SINCE then, not before.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("kind", "market", "player", "player_id", "team", "side", "line", "milestone",
          "play", "pick", "selection", "label", "probability", "p_pick", "odds",
          "book_price", "locked_at")


def seal_path(log_path) -> Path:
    p = Path(log_path)
    return p.with_name(p.stem + ".seal.json")


def entries(log) -> dict:
    """{key: entry} for every frozen pick, flat logs and NFL-style {section: {...}} alike."""
    out = {}
    for k, e in (log or {}).items():
        if str(k).startswith("_") or not isinstance(e, dict):
            continue
        if "locked_at" in e:
            out[str(k)] = e
        else:
            for k2, x in e.items():
                if not str(k2).startswith("_") and isinstance(x, dict) and "locked_at" in x:
                    out[f"{k}/{k2}"] = x
    return out


def _digest(prev: str, entry: dict, fields) -> str:
    body = json.dumps({f: entry.get(f) for f in fields}, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256((prev + body).encode("utf-8")).hexdigest()


def load(log_path) -> dict:
    try:
        return json.loads(seal_path(log_path).read_text(encoding="utf-8"))
    except Exception:
        return {"head": "", "seals": []}


def seal(log_path, now=None) -> int:
    """Seal every frozen pick in the log not yet sealed (oldest lock first). Returns
    how many were added. Never rewrites an existing seal."""
    log_path = Path(log_path)
    if not log_path.exists():
        return 0
    log = json.loads(log_path.read_text(encoding="utf-8"))
    chain = load(log_path)
    done = {(s["key"], s.get("locked_at")) for s in chain["seals"]}
    new = sorted(((k, e) for k, e in entries(log).items() if (k, str(e.get("locked_at"))) not in done),
                 key=lambda kv: (str(kv[1].get("locked_at")), kv[0]))
    if not new:
        return 0
    head = chain.get("head") or ""
    at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    for k, e in new:
        fields = [f for f in FIELDS if f in e]
        head = _digest(head, e, fields)
        chain["seals"].append({"key": k, "locked_at": str(e.get("locked_at")), "fields": fields,
                               "digest": head, "sealed_at": at})
    chain["head"] = head
    chain["n"] = len(chain["seals"])
    out = seal_path(log_path)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(chain, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    return len(new)


def verify(log_path) -> dict:
    """{"sealed": n, "broken": [keys whose sealed fields changed], "missing": [keys gone],
    "superseded": n re-locked as a new version}."""
    log_path = Path(log_path)
    chain = load(log_path)
    try:
        log = entries(json.loads(log_path.read_text(encoding="utf-8")))
    except Exception:
        log = {}
    broken, missing, head, superseded = [], [], "", 0
    for s in chain["seals"]:
        e = log.get(s["key"])
        if e is None:
            missing.append(s["key"])
            head = s["digest"]                   # keep walking: report every problem
            continue
        if str(e.get("locked_at")) != s.get("locked_at"):
            superseded += 1                      # released and frozen again: a new version
            head = s["digest"]
            continue
        if _digest(head, e, s["fields"]) != s["digest"]:
            broken.append(s["key"])
        head = s["digest"]
    if chain["seals"] and head != chain.get("head"):
        broken.append("(chain head)")
    return {"sealed": len(chain["seals"]), "broken": broken, "missing": missing,
            "superseded": superseded}
