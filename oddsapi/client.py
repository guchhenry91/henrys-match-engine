"""Budget-enforcing client for The Odds API (50,000 credits a month).

THE BUDGET IS THE POINT OF THIS MODULE, agreed with the owner on 2026-09-26
before a single credit was spent. Every limit below is checked BEFORE a request
leaves, so a limit is never learned by going over it:

  * OFF SWITCH. Nothing is sent unless ODDS_API_ENABLED is "true". It ships off.
  * PER RUN   200 credits.
  * PER DAY   1,000 credits, all sports together (UTC day, from the ledger).
  * PER MONTH hard stop at 40,000 USED, read from the API's own x-requests-used
    header -- the account's number, not our tally, so a run we did not log
    still counts. 10,000 of the 50,000 are never touched.
  * HISTORICAL endpoints cost 10x and are refused unless a caller passes an
    explicit approved allowance for that one job.

HOW CREDITS ARE CHARGED (the-odds-api.com v4 docs): /sports and /events are free;
/odds and /events/{id}/odds cost markets x regions (every 10 bookmakers count as
one region, and an EMPTY event-odds reply costs nothing); /scores is 1, or 2 with
daysFrom. `cost()` computes the worst case up front so the pre-flight check can
refuse a call that would cross a limit.

THE KEY TRAVELS IN THE URL (?apiKey=...), unlike the API-Sports key which is a
header. So no URL is ever printed, logged or put in an exception message, and
anything that might carry the key goes through `_scrub` first. A leaked key in a
public Actions log would let anyone spend this budget.

Every call is written to data-raw/odds_api/ledger.json -- cost, sport, purpose
and what the account says is left -- so where the credits went is never a matter
of reading thirty workflow logs, which is what the API-Football burn of
September cost to diagnose.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://api.the-odds-api.com/v4"
ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data-raw" / "odds_api" / "ledger.json"

RUN_CAP = 200
DAY_CAP = 1_000
MONTH_HARD_STOP = 40_000
MONTH_WARN_AT = (10_000, 30_000)
DAY_WARN_AT = 600
LEDGER_KEEP_CALLS = 500
RATE_LIMIT_PAUSE_SECONDS = 5.0

FREE = ("sports", "events")          # path heads that never cost a credit


class BudgetExceeded(RuntimeError):
    """A limit would be crossed. Not an error to retry -- a reason to stop."""


class Disabled(RuntimeError):
    """ODDS_API_ENABLED is not "true", or there is no key."""


def enabled() -> bool:
    return (os.environ.get("ODDS_API_ENABLED", "").strip().lower() == "true"
            and bool(os.environ.get("ODDS_API_KEY")))


def cost(markets: int, regions: int = 1, bookmakers: int | None = None) -> int:
    """Worst-case credits for one /odds or /events/{id}/odds call.

    `bookmakers`, when given, replaces regions: every 10 books is one region."""
    if bookmakers:
        regions = math.ceil(bookmakers / 10)
    return max(0, int(markets)) * max(1, int(regions))


def _is_free(path: str) -> bool:
    parts = [p for p in path.strip("/").split("/") if p]
    # /sports, /sports/{sport}/events -- but NOT /sports/{sport}/events/{id}/odds
    return bool(parts) and parts[-1] in FREE and "historical" not in parts


def _is_historical(path: str) -> bool:
    return "historical" in path.strip("/").split("/")


class Client:
    def __init__(self, key=None, dry_run=False, opener=urllib.request.urlopen,
                 ledger_path=None, clock=None, sleeper=time.sleep,
                 historical_allowance: int = 0):
        self.key = key or os.environ.get("ODDS_API_KEY") or ""
        self.dry_run = dry_run
        self.opener = opener
        self.ledger_path = Path(ledger_path or LEDGER)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper
        self.historical_allowance = int(historical_allowance)
        # ONE BUDGET PER JOB, not per script: a Cloudflare job (JOB_ID) can run more
        # than one spending script -- the soccer refresh runs two -- and each used to
        # start from zero, so a "200 a run" cap allowed 400 (review, 2026-10-04).
        self.run_id = os.environ.get("JOB_ID") or None
        self.historical_used = 0
        self.planned = []                 # dry-run: what WOULD have been spent
        self.warnings = []
        self._usage_fresh = False
        self._relay = None
        self.ledger = self._load()
        self.run_used = (sum(int(c.get("credits") or 0) for c in self.ledger.get("calls") or []
                             if c.get("run") == self.run_id) if self.run_id else 0)

    # --- ledger -----------------------------------------------------------------
    def _load(self) -> dict:
        try:
            raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        raw.setdefault("_note", (
            "Every The Odds API call: cost, sport, purpose, and what the account "
            "reports left. Written by oddsapi/client.py -- never by hand. Limits: "
            f"{RUN_CAP}/run, {DAY_CAP}/day, hard stop at {MONTH_HARD_STOP} used "
            "per month."))
        raw.setdefault("days", {})
        raw.setdefault("calls", [])
        raw.setdefault("alerts", [])
        return raw

    def _save(self) -> None:
        self.ledger["calls"] = self.ledger["calls"][-LEDGER_KEEP_CALLS:]
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.ledger, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(self.ledger_path)

    def _today(self) -> str:
        return self.clock().date().isoformat()

    def day_used(self) -> int:
        """Today's credits: this engine's own, PLUS what the dashboard's relay spent
        (cloudflare/worker.js) -- the 1,000 a day is one limit for the account."""
        own = int((self.ledger["days"].get(self._today()) or {}).get("credits", 0))
        return own + self.relay_spent()

    def relay_spent(self) -> int:
        """The dashboard relay's spend today, read once per run from the Worker.
        0 when not on Cloudflare or the Worker cannot be reached (the monthly stop,
        read from the API's own counter, still covers everything)."""
        if self._relay is None:
            self._relay = 0
            url, token = os.environ.get("STATE_URL"), os.environ.get("STATE_TOKEN")
            if url and token and not self.dry_run:
                try:
                    req = urllib.request.Request(
                        f"{url.rstrip('/')}/internal/relay-spend?day={self._today()}",
                        headers={"authorization": f"Bearer {token}",
                                 "user-agent": "henrys-match-engine-runner/1.0"})
                    with urllib.request.urlopen(req, timeout=20) as r:
                        self._relay = int(json.loads(r.read().decode()).get("credits") or 0)
                except Exception as exc:
                    print(f"  (dashboard relay spend unavailable: {type(exc).__name__})")
        return self._relay

    def month_used(self) -> int | None:
        """The ACCOUNT's own figure from the last response, or None if never seen."""
        v = self.ledger.get("account_used")
        return int(v) if v is not None else None

    # --- the guard ----------------------------------------------------------------
    def preflight(self, path: str, est: int) -> None:
        """Raise BEFORE sending if any limit would be crossed. Free calls pass."""
        if not self.dry_run and not enabled():
            raise Disabled("The Odds API is switched off (ODDS_API_ENABLED is not "
                           "'true', or ODDS_API_KEY is missing); nothing sent")
        if _is_historical(path):
            if self.historical_used + est > self.historical_allowance:
                raise BudgetExceeded(
                    f"historical odds cost 10x and need an approved allowance; "
                    f"this job has {self.historical_allowance - self.historical_used} "
                    f"left, the call needs {est}")
        if est <= 0:
            return
        if self.run_used + est > RUN_CAP:
            raise BudgetExceeded(f"run cap: {self.run_used} used + {est} > {RUN_CAP}")
        if self.day_used() + est > DAY_CAP:
            raise BudgetExceeded(f"day cap: {self.day_used()} used today + {est} "
                                 f"> {DAY_CAP}")
        used = self.month_used()
        if used is not None and used + est > MONTH_HARD_STOP:
            raise BudgetExceeded(f"monthly hard stop: account has used {used}; "
                                 f"+{est} would pass {MONTH_HARD_STOP}")

    def refresh_usage(self) -> int | None:
        """Read the account's usage with a FREE call (/sports).

        Needed because the month resets on the billing date: a ledger that last
        saw 40,000 used would otherwise block forever, never making the call
        that would reveal the reset. Costs nothing."""
        self.get("sports", sport="-", purpose="usage check")
        return self.month_used()

    # --- the request ----------------------------------------------------------------
    def get(self, path: str, *, sport: str, purpose: str, est: int | None = None,
            **params):
        """One request. `est` is the worst-case credit cost (see cost()); free
        endpoints need none. Returns the decoded JSON body."""
        est = 0 if _is_free(path) else est
        if est is None:
            raise ValueError(f"/{path} is a paid endpoint: pass est=cost(...)")
        if est > 0 and not self.dry_run and not self._usage_fresh:
            # First PAID call this run: read the account's usage for free first,
            # so the monthly stop is checked against today's number, not a
            # ledger that could be days old or from before a billing reset.
            self.refresh_usage()
        self.preflight(path, est)
        if self.dry_run:
            self.planned.append({"path": path, "sport": sport, "purpose": purpose,
                                 "est": est})
            print(f"  [dry run] would call /{path} for {sport} ({purpose}), "
                  f"up to {est} credit(s)")
            return None

        query = urllib.parse.urlencode(
            {"apiKey": self.key, **{k: v for k, v in params.items() if v is not None}})
        request = urllib.request.Request(
            f"{BASE}/{path.strip('/')}?{query}",
            headers={"User-Agent": "henrys-match-engine/1.0"})
        payload, headers = self._send(request, path)
        spent = self._read_usage(headers, fallback=est)
        self._usage_fresh = self._usage_fresh or self.month_used() is not None
        self._record(path, sport, purpose, spent)
        if _is_historical(path):
            self.historical_used += spent
        return payload

    def _send(self, request, path):
        for attempt in (1, 2):
            try:
                with self.opener(request, timeout=25) as response:
                    body = response.read().decode("utf-8")
                    return json.loads(body) if body else None, \
                        getattr(response, "headers", None) or {}
            except urllib.error.HTTPError as exc:
                # ONE wait on a rate limit; every other API "no" is final --
                # asking again only spends the budget twice on the same answer.
                if exc.code == 429 and attempt == 1:
                    self.sleeper(RATE_LIMIT_PAUSE_SECONDS)
                    continue
                if exc.code in (401, 403):
                    raise RuntimeError(f"The Odds API refused the key (HTTP {exc.code}) "
                                       f"for /{_scrub(path, self.key)}") from None
                raise RuntimeError(f"The Odds API HTTP {exc.code} for "
                                   f"/{_scrub(path, self.key)}") from None
            except Exception as exc:
                # A transport error's text can echo the URL -- and the key in it.
                raise RuntimeError(f"The Odds API request failed for "
                                   f"/{_scrub(path, self.key)}: "
                                   f"{_scrub(type(exc).__name__ + ': ' + str(exc), self.key)}"
                                   ) from None
        raise RuntimeError(f"The Odds API rate-limited /{_scrub(path, self.key)} twice")

    def _read_usage(self, headers, fallback: int) -> int:
        def num(name):
            try:
                return int(float(headers.get(name)))
            except (TypeError, ValueError, AttributeError):
                return None
        last, used, left = (num("x-requests-last"), num("x-requests-used"),
                            num("x-requests-remaining"))
        if used is not None:
            self.ledger["account_used"] = used
        if left is not None:
            self.ledger["account_remaining"] = left
        # Trust the API's charge; without the header assume the worst case.
        return last if last is not None else fallback

    def _record(self, path, sport, purpose, spent) -> None:
        now = self.clock()
        self.run_used += spent
        day = self.ledger["days"].setdefault(self._today(), {"credits": 0, "by_sport": {}})
        day["credits"] += spent
        day["by_sport"][sport] = day["by_sport"].get(sport, 0) + spent
        self.ledger["updated"] = now.isoformat()
        self.ledger["calls"].append({
            "at": now.isoformat(timespec="seconds"), "path": _scrub(path, self.key),
            "sport": sport, "purpose": purpose, "credits": spent, "run": self.run_id,
            "account_used": self.ledger.get("account_used"),
            "account_remaining": self.ledger.get("account_remaining")})
        self._check_alerts(day["credits"])
        self._save()

    def _check_alerts(self, day_credits: int) -> None:
        """Warnings at 10,000 and 30,000 used this month, and on any day over 600.
        Each fires ONCE (remembered in the ledger); the caller relays them."""
        sent = set(self.ledger["alerts"])
        used = self.month_used()
        found = []
        for mark in MONTH_WARN_AT:
            if used is not None and used >= mark:
                # Once per calendar month per mark, so a new month can warn again.
                found.append((f"month:{mark}:{self._month_tag()}", f"The Odds API: {used:,} of 50,000 credits used "
                                   f"this month (warning at {mark:,}; hard stop "
                                   f"{MONTH_HARD_STOP:,})"))
        if day_credits > DAY_WARN_AT:
            found.append((f"day:{self._today()}",
                          f"The Odds API: {day_credits} credits spent today "
                          f"(warning above {DAY_WARN_AT}; cap {DAY_CAP})"))
        for key, text in found:
            if key not in sent:
                self.ledger["alerts"].append(key)
                self.warnings.append(text)
                print(f"::warning::{text}")
        self.ledger["alerts"] = self.ledger["alerts"][-100:]

    def _month_tag(self) -> str:
        return self.clock().strftime("%Y-%m")

    def report(self) -> str:
        """One line every caller prints, so a run's spend is never invisible."""
        if self.dry_run:
            return (f"The Odds API (dry run): {len(self.planned)} call(s) planned, "
                    f"up to {sum(p['est'] for p in self.planned)} credit(s); none spent")
        out = (f"The Odds API: {self.run_used} credit(s) this run, "
               f"{self.day_used()} today")
        if self.month_used() is not None:
            out += (f"; account has used {self.month_used():,} this month "
                    f"(hard stop {MONTH_HARD_STOP:,})")
        return out


def _scrub(text: str, key: str) -> str:
    """Remove the key from anything that might be printed."""
    text = str(text)
    if key:
        text = text.replace(key, "***")
        text = text.replace(urllib.parse.quote(key), "***")
    if "apiKey=" in text:
        text = re.sub(r"apiKey=[^&\s]+", "apiKey=***", text)
    return text
