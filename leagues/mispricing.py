"""bet365 prices that beat the Betfair Exchange -- an edge that needs no model.

WHY. The audit (tracking/trust.py) found the model's own opinions have not beaten
the bookmakers. This does not need them to. The Betfair Exchange is where bettors
trade with each other at almost no margin, so its prices are the sharpest
independent read of the true chance. When bet365 offers MORE than that fair price,
the bet has positive expected value whatever any model thinks -- the standard way
sharp bettors win at a soft book.

football-data.co.uk's fixtures.csv carries both on the same row: bet365 (B365*)
and the exchange (BFE*) for match result, over/under 2.5 goals and the Asian
handicap. For each, the exchange's prices are de-vigged into fair chances and every
bet365 price is scored: EV = fair chance x bet365 price - 1. A price clearing
MIN_EV is flagged. Rows where the exchange looks illiquid or stale (its prices
adding up far from 100%) are skipped rather than trusted.

Every flag is logged at the FIRST price seen -- the one a bettor acting on the
alert would get -- and graded from the final score, so this edge earns or loses
its own record in Betting performance like everything else.
"""
from __future__ import annotations

import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from leagues.names import UnknownTeam, canonical
from leagues.odds import DIV, FEED

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data-raw" / "leagues" / "mispriced_log.json"
MIN_EV = 0.03
OVERROUND_OK = (0.98, 1.025)     # a THIN exchange book (over 2.5%) de-vigs with ~2-3% error, the size of MIN_EV
LONDON = ZoneInfo("Europe/London")
LEAGUE_OF = {v: k for k, v in DIV.items()}


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x > 1.0 else None


def _fair(prices):
    inv = [1.0 / p for p in prices]
    total = sum(inv)
    if not OVERROUND_OK[0] <= total <= OVERROUND_OK[1]:
        return None
    return [x / total for x in inv]


def _kickoff(date, time):
    try:
        d = datetime.strptime(f"{date} {time or '15:00'}", "%d/%m/%Y %H:%M")
    except ValueError:
        return None
    return d.replace(tzinfo=LONDON).astimezone(timezone.utc)


def scan(frame: pd.DataFrame) -> list:
    """Every bet365 price in our leagues that beats the exchange's fair price."""
    out = []
    for _, r in frame.iterrows():
        league = LEAGUE_OF.get(r.get("Div"))
        if not league:
            continue
        try:
            home, away = canonical(r["HomeTeam"], league), canonical(r["AwayTeam"], league)
        except UnknownTeam:
            continue
        ko = _kickoff(r.get("Date"), r.get("Time"))
        if ko is None:
            continue
        base = {"league_key": league, "home": home, "away": away, "date": ko.isoformat()}
        markets = [
            ("match_result", None, [("home", "B365H", "BFEH"), ("draw", "B365D", "BFED"),
                                    ("away", "B365A", "BFEA")]),
            ("over_under_2.5", None, [("over", "B365>2.5", "BFE>2.5"), ("under", "B365<2.5", "BFE<2.5")]),
            ("asian_handicap", r.get("AHh"), [("home", "B365AHH", "BFEAHH"), ("away", "B365AHA", "BFEAHA")]),
        ]
        for market, line, sels in markets:
            b365 = [_f(r.get(b)) for _, b, _ in sels]
            ex = [_f(r.get(e)) for _, _, e in sels]
            if None in b365 or None in ex:
                continue
            if market == "asian_handicap":
                try:
                    line = float(line)
                except (TypeError, ValueError):
                    continue
            fair = _fair(ex)
            if fair is None:
                continue
            for (sel, _, _), price, exch, p in zip(sels, b365, ex, fair):
                ev = p * price - 1.0
                if ev >= MIN_EV:
                    out.append({**base, "market": market, "selection": sel, "line": line,
                                "b365": price, "exchange": exch,
                                "fair_p": round(p, 4), "ev": round(ev, 4)})
    return out


def fetch() -> pd.DataFrame:
    req = urllib.request.Request(FEED, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return pd.read_csv(io.StringIO(resp.read().decode("utf-8-sig", "replace")))


def key(e) -> str:
    return f"{e['league_key']}|{e['date'][:10]}|{e['home']}|{e['away']}|{e['market']}|{e['selection']}|{e.get('line')}"


def settle(e, hg: int, ag: int) -> float | None:
    """Profit for one unit at the logged bet365 price; AH quarter lines split."""
    price = float(e["b365"])
    if e["market"] == "match_result":
        won = {"home": hg > ag, "draw": hg == ag, "away": ag > hg}[e["selection"]]
        return price - 1.0 if won else -1.0
    if e["market"] == "over_under_2.5":
        won = (hg + ag > 2.5) if e["selection"] == "over" else (hg + ag < 2.5)
        return price - 1.0 if won else -1.0
    line = float(e["line"])                 # the HOME handicap
    margin = (hg - ag) if e["selection"] == "home" else (ag - hg)
    hcap = line if e["selection"] == "home" else -line
    halves = [hcap - 0.25, hcap + 0.25] if abs((hcap * 4) % 2) == 1 else [hcap]
    total = 0.0
    for h in halves:
        m = margin + h
        total += (price - 1.0) if m > 0 else 0.0 if m == 0 else -1.0
    return total / len(halves)


# CLOSING prices, the fast verdict on an edge. football-data's season file carries
# bet365's and the exchange's prices at kickoff for every played match. An edge
# whose price still beats the exchange's CLOSING fair chance was real value when
# taken -- the standard test of a betting edge, visible after a few dozen bets
# where profit needs hundreds.
CLOSE_COLS = {
    "match_result": ({"home": "B365CH", "draw": "B365CD", "away": "B365CA"},
                     {"home": "BFECH", "draw": "BFECD", "away": "BFECA"}, None),
    "over_under_2.5": ({"over": "B365C>2.5", "under": "B365C<2.5"},
                       {"over": "BFEC>2.5", "under": "BFEC<2.5"}, None),
    "asian_handicap": ({"home": "B365CAHH", "away": "B365CAHA"},
                       {"home": "BFECAHH", "away": "BFECAHA"}, "AHCh"),
}


def closing_rows(texts: dict) -> dict:
    """{(league, date, home, away): season-file row} from {league: CSV text}."""
    out = {}
    for league, text in (texts or {}).items():
        try:
            frame = pd.read_csv(io.StringIO(text), encoding="latin-1")
        except Exception:
            continue
        for _, r in frame.iterrows():
            try:
                home, away = canonical(r["HomeTeam"], league), canonical(r["AwayTeam"], league)
                day = datetime.strptime(str(r["Date"]), "%d/%m/%Y").date().isoformat()
            except (UnknownTeam, KeyError, TypeError, ValueError):
                continue
            out[(league, day, home, away)] = r
    return out


def stamp_close(e: dict, row) -> bool:
    """Closing fair chance and bet365 price on one logged edge. True if stamped."""
    book, exch, line_col = CLOSE_COLS[e["market"]]
    if line_col is not None:
        try:
            if float(row.get(line_col)) != float(e["line"]):
                return False                 # the handicap moved: no like-for-like close
        except (TypeError, ValueError):
            return False
    order = list(exch)
    fair = _fair([_f(row.get(exch[s])) for s in order]) if all(
        _f(row.get(exch[s])) for s in order) else None
    if fair is None:
        return False
    p = fair[order.index(e["selection"])]
    e["open"] = {"p": e["fair_p"], "price": e["b365"], "book": "bet365"}
    e["close_p"] = round(p, 4)
    e["close_price"] = _f(row.get(book[e["selection"]]))
    # The edge measured against the SHARP CLOSE: positive means the price taken
    # still beat the exchange's final verdict on this outcome.
    e["close_ev"] = round(e["close_p"] * float(e["b365"]) - 1.0, 4)
    return True


def update(found: list, results: dict, now=None, log_path=None, closes=None) -> dict:
    """Log new flags at their first price, grade finished ones; returns the payload."""
    now = now or datetime.now(timezone.utc)
    log_file = Path(log_path) if log_path else LOG
    try:
        log = json.loads(log_file.read_text(encoding="utf-8"))
    except Exception:
        log = {}
    stamp = now.isoformat(timespec="seconds")
    for e in found:
        if datetime.fromisoformat(e["date"]) <= now:
            continue                         # already started: not a bet anyone can make
        k = key(e)
        if k not in log:
            log[k] = {**e, "first_seen": stamp, "book_price": e["b365"], "book": "bet365"}
        log[k]["last_seen"], log[k]["ev_latest"], log[k]["b365_latest"] = stamp, e["ev"], e["b365"]
    for k, e in log.items():
        if e.get("graded"):
            continue
        match = (e["league_key"], e["date"][:10], e["home"], e["away"])
        if "close_p" not in e and (closes or {}).get(match) is not None:
            stamp_close(e, closes[match])
        res = results.get(match)
        if res is None:
            continue
        units = settle(e, *res)
        e["actual"] = f"{res[0]}-{res[1]}"
        e["units"] = round(units, 4)
        e["graded"] = "correct" if units > 0 else "wrong" if units < 0 else "void"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    upcoming = sorted((e for e in log.values() if not e.get("graded")
                       and datetime.fromisoformat(e["date"]) > now),
                      key=lambda e: (-e.get("ev_latest", e["ev"]), e["date"]))
    settled = sorted((e for e in log.values() if e.get("graded")),
                     key=lambda e: e["date"], reverse=True)
    return {"updated": stamp, "min_ev": MIN_EV, "source": "Betfair Exchange (football-data.co.uk)",
            "upcoming": upcoming, "settled": settled}


def results_from_boards(out_dir: Path) -> dict:
    """(league, date, home, away) -> (home goals, away goals) from the league boards."""
    res = {}
    for league, slug in (("PL", "pl"), ("LALIGA", "laliga"), ("BUNDESLIGA", "bundesliga"),
                         ("LIGUE1", "ligue1"), ("SERIEA", "seriea")):
        try:
            d = json.loads((out_dir / f"{slug}.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        for m in (d.get("season") or []):
            r = m.get("result") if isinstance(m, dict) else None
            if isinstance(r, dict) and r.get("home_goals") is not None:
                res[(league, str(m["date"])[:10], m["home"], m["away"])] = (int(r["home_goals"]), int(r["away_goals"]))
    return res
