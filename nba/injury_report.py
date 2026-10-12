"""The NBA's official pre-game injury report: read, and kept for the live board.

Teammates-out (nba/forecast._vacated) is trained on who this report listed OUT before
tip-off (scripts/export_nba_injury_reports.py, 5pm ET report, 2021-22 on). The live
board must read the same thing, so each NBA run fetches TODAY's latest report and
  - returns its OUT names (the board's absences for tonight's games), and
  - writes them into this season's file, so the season's played games are later
    rebuilt from the report too, exactly as training was.

Source: ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<slot>.pdf.
Older seasons publish on the hour (05PM); from 2025-26 every 15 minutes (05_30PM).
Never fatal: no report (before the first one of the day, an off day, a network fault)
means no absences from it, which is what training saw on a day with no report.
"""
from __future__ import annotations

import io
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data-raw" / "nba" / "injury_reports"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
ET = ZoneInfo("America/New_York")
EARLIEST = (11, 0)          # no report is looked for before 11am ET
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
ROW = re.compile(r"([A-Z][A-Za-z'\.\- ]*?(?: Jr\.| Sr\.| II| III| IV| V)?,\s*[A-Z][A-Za-z'\.\- ]*?)\s{2,}"
                 r"(" + "|".join(STATUSES) + r")\b")
TEAMS = ("Hawks|Celtics|Nets|Hornets|Bulls|Cavaliers|Mavericks|Nuggets|Pistons|Warriors|Rockets|"
         "Pacers|Clippers|Lakers|Grizzlies|Heat|Bucks|Timberwolves|Pelicans|Knicks|City ?Thunder|Thunder|"
         "Magic|76ers|Suns|Trail ?Blazers|Kings|Spurs|Raptors|Jazz|Wizards")
# The layout sometimes glues a suffix to the surname ("LivelyII,Dereck").
GLUED_SUFFIX = re.compile(r"([a-z])(Jr\.|Sr\.|II|III|IV)(?=,)")
# Text the layout puts in front of a player on the same line: a matchup code, a team.
PREFIX = re.compile(r"^.*?\b(?:" + TEAMS + r")\s+|^(?:[A-Z]{2,3}(?:@[A-Z]{2,3})?\s+)+")


def clean(name: str) -> str:
    """The player's "Last, First" with any matchup / team text in front removed --
    also run on names already stored, so a parser fix reaches old seasons too."""
    name = GLUED_SUFFIX.sub(r"\1 \2", str(name or ""))
    prev = None
    while prev != name:
        prev, name = name, PREFIX.sub("", name).strip()
    return name


def parse(pdf: bytes) -> list:
    """[(player "Last, First", status)] from one report."""
    import pypdf
    rows = []
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout") or ""
        for line in text.splitlines():
            for m in ROW.finditer(line):
                rows.append((clean(re.sub(r"\s+", " ", m.group(1)).strip()), m.group(2)))
    return sorted(set(rows))


def _slot(h: int, m: int, quarter: bool) -> str:
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12:02d}_{m:02d}{ampm}" if quarter else f"{h12:02d}{ampm}"


def slots_back_from(t: datetime) -> list:
    """Report names to try, newest first, from `t` (ET) back to EARLIEST."""
    t = t.replace(minute=t.minute - t.minute % 15, second=0, microsecond=0)
    out = []
    while (t.hour, t.minute) >= EARLIEST:
        out.append(_slot(t.hour, t.minute, True))
        if t.minute == 0:
            out.append(_slot(t.hour, 0, False))
        t -= timedelta(minutes=15)
    return out


def fetch(day: str, slot: str, opener=urllib.request.urlopen) -> bytes | None:
    try:
        with opener(URL.format(d=day, t=slot), timeout=30) as r:
            return r.read()
    except Exception:
        return None


def latest(now=None, opener=urllib.request.urlopen):
    """(day, slot, [(player, status)]) for today's newest report, or None."""
    now = (now or datetime.now(timezone.utc)).astimezone(ET)
    day = now.date().isoformat()
    for slot in slots_back_from(now):
        pdf = fetch(day, slot, opener)
        if pdf:
            try:
                return day, slot, parse(pdf)
            except Exception as exc:                   # noqa: BLE001
                print(f"  injury report {day} {slot} unreadable ({type(exc).__name__})")
                return None
    return None


def refresh(season: int, now=None, opener=urllib.request.urlopen, base: Path | None = None) -> list:
    """Today's OUT names ("Last, First"); [] when there is no report. See fetch_today."""
    return fetch_today(season, now, opener, base)["out"]


def fetch_today(season: int, now=None, opener=urllib.request.urlopen, base: Path | None = None) -> dict:
    """{"ok", "day", "slot", "out", "listed", "why"}: today's report, with its rows
    written into the season's file (replacing an earlier fetch of the same day).
    ok=False when there is no report -- the board must then NOT read "nobody is out"
    into teammates-out (third review, 2026-10-12)."""
    try:
        got = latest(now, opener)
    except Exception as exc:                           # noqa: BLE001
        print(f"  injury report unavailable ({type(exc).__name__})")
        return {"ok": False, "out": [], "why": f"fetch failed ({type(exc).__name__})"}
    if not got:
        print("  injury report: none published yet today")
        return {"ok": False, "out": [], "why": "none published yet today"}
    day, slot, rows = got
    base = Path(base or DIR)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{season}.csv.gz"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["date", "player", "status"])
    have = have[have["date"] != day]
    new = pd.DataFrame([{"date": day, "player": p, "status": s} for p, s in rows],
                       columns=["date", "player", "status"])
    frame = pd.concat([have, new], ignore_index=True) if len(have) else new
    frame.to_csv(path, index=False, compression="gzip")
    out = [p for p, s in rows if s == "Out"]
    print(f"  injury report {day} {slot}: {len(out)} OUT of {len(rows)} listed")
    return {"ok": True, "day": day, "slot": slot, "out": out, "listed": len(rows), "why": None}
