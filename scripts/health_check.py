"""Daily self-check: is every board current, grading, and running?

WHY THIS EXISTS. In one week three failures ran unnoticed for hours to weeks:
the NFL board priced all of 2026 on 2025 form (four weeks), the soccer refresh
failed its sanity check every run for ten hours (nothing published), and MLB
publish crashed on a date format. Each was visible in the data the whole time --
nobody was looking. This looks, every few hours, and says so on the site and by
Telegram when something is wrong.

It reads only what the pipeline already published (no paid API calls) plus the
latest run of each GitHub workflow. Writes data/health.json:
  {"status": "ok"|"warn"|"fail", "checked_at", "since", "checks": [...]}
A check is FAIL when a board can no longer be trusted, WARN when it needs a look.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "health.json"
WORKFLOWS = {"leagues.yml": "Soccer refresh", "nfl.yml": "NFL refresh",
             "nba.yml": "NBA refresh", "mlb.yml": "MLB refresh", "lock.yml": "Pick freeze"}
RANK = {"ok": 0, "warn": 1, "fail": 2}


def _utc(x):
    if x is None:
        return None
    try:
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _read(rel):
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except Exception:
        return None


def _hours(now, when):
    return None if when is None else (now - when).total_seconds() / 3600.0


class Report:
    def __init__(self):
        self.checks = []

    def add(self, sport, name, status, message):
        self.checks.append({"sport": sport, "check": name, "status": status, "message": message})


# Hours since each sport's workflow last SUCCEEDED (filled by workflow_age). A board
# that has not changed is deliberately not republished, so its "updated" time ages
# while the pipeline is fine -- on 2026-10-03 that raised a false "NBA board 31h
# old" in the off-season. A board is stale only when BOTH it and its workflow's
# last success are old.
LAST_OK = {}
SPORT_WORKFLOW = {"Soccer": "leagues.yml", "NFL": "nfl.yml", "NBA": "nba.yml", "MLB": "mlb.yml"}


def board_age(r, now, sport, rel, warn_h, fail_h, active=True):
    d = _read(rel)
    if d is None:
        r.add(sport, "board", "fail", f"{rel} is missing or unreadable")
        return None
    h = _hours(now, _utc(d.get("updated")))
    ok_h = LAST_OK.get(SPORT_WORKFLOW.get(sport))
    if h is not None and ok_h is not None and ok_h < h:
        if ok_h <= warn_h:
            r.add(sport, "board", "ok", f"unchanged since {h:.0f}h ago; refresh ran {ok_h:.1f}h ago")
            return d
        h = ok_h
    if h is None:
        r.add(sport, "board", "warn", "board has no update time")
    elif active and h > fail_h:
        r.add(sport, "board", "fail", f"board last updated {h:.0f}h ago (limit {fail_h}h)")
    elif active and h > warn_h:
        r.add(sport, "board", "warn", f"board last updated {h:.0f}h ago")
    else:
        r.add(sport, "board", "ok", f"updated {h:.1f}h ago")
    return d


def nfl(r, now):
    d = board_age(r, now, "NFL", "data/nfl/board.json", 14, 30)
    if d:
        thru, season, week = d.get("player_data_through") or {}, d.get("season"), d.get("week")
        if season and week and thru:
            stale = (thru.get("season") != season) or (int(thru.get("week") or 0) < int(week) - 2)
            if stale and int(week) > 1:
                r.add("NFL", "player data", "fail",
                      f"props priced on {thru.get('season')} week {thru.get('week')} data for "
                      f"{season} week {week} -- form and workload are out of date")
            else:
                r.add("NFL", "player data", "ok", f"through {thru.get('season')} week {thru.get('week')}")
    log = _read("data-raw/nfl/picks_log.json") or {}
    stuck = [k for sec in ("games", "props") for k, e in (log.get(sec) or {}).items()
             if not k.startswith("_") and isinstance(e, dict) and not e.get("graded")
             and (_hours(now, _utc(e.get("kickoff"))) or 0) > 60]
    r.add("NFL", "grading", "warn" if stuck else "ok",
          f"{len(stuck)} pick(s) ungraded 60h+ after kickoff" if stuck else "no stuck picks")


def line_sport(r, now, sport, key, start_key):
    # NBA and MLB run twice/three times a day with a long overnight gap (MLB 22:00 ->
    # 14:00 is 16h, NBA 21:30 -> 15:00 is 17.5h), so a 14h warning fired every
    # morning on boards that were perfectly current. 20h clears the gap; a missed
    # run still shows by the next slot.
    d = board_age(r, now, sport, f"data/{key}/board.json", 20, 30,
                  active=(_read(f"data/{key}/board.json") or {}).get("status") == "live")
    log = _read(f"data-raw/{key}/picks_log.json") or {}
    stuck = [k for k, e in log.items() if isinstance(e, dict) and not e.get("graded")
             and (_hours(now, _utc(e.get(start_key))) or 0) > 36]
    r.add(sport, "grading", ("fail" if len(stuck) > 20 else "warn") if stuck else "ok",
          f"{len(stuck)} pick(s) ungraded 36h+ after the start" if stuck else "no stuck picks")
    return d


def mlb_boxes(r, now):
    path = ROOT / "data-raw" / "mlb" / "current"
    files = sorted(path.glob("team_*.csv.gz"))
    if not files:
        return
    import pandas as pd
    t = pd.read_csv(files[-1])
    last = pd.to_datetime(t["game_date"], format="mixed").max()
    log = _read("data-raw/mlb/picks_log.json") or {}
    pending_old = [e for e in log.values() if isinstance(e, dict) and not e.get("graded")
                   and (_hours(now, _utc(e.get("tipoff"))) or 0) > 30]
    if pending_old and (now.date() - last.date()).days > 2:
        r.add("MLB", "box scores", "fail", f"newest box score {last.date()}, picks waiting on later games")
    else:
        r.add("MLB", "box scores", "ok", f"newest box score {last.date()}")


def soccer(r, now):
    best = board_age(r, now, "Soccer", "data/leagues/best.json", 8, 26)
    pp = _read("data/leagues/player_picks.json") or {}
    waiting = pp.get("awaiting_data") or []
    old = [w for w in waiting if (_hours(now, _utc(w.get("date"))) or 0) > 72]
    r.add("Soccer", "player grading", "warn" if old else "ok",
          f"{len(old)} player pick(s) still waiting on stats 72h+ after kickoff" if old else "no stuck picks")
    if pp.get("ungradeable_leagues"):
        r.add("Soccer", "player grading", "warn", f"cannot grade: {', '.join(pp['ungradeable_leagues'])}")


def odds_budget(r):
    led = _read("data-raw/odds_api/ledger.json") or {}
    used = led.get("account_used")
    if used is None:
        return
    status = "fail" if used >= 38000 else "warn" if used >= 30000 else "ok"
    r.add("Odds API", "monthly credits", status, f"{used:,} of 40,000 used this month")


def _cloudflare_runs(wf):
    """The same shape as _runs, from the Cloudflare job queue (cloudflare/worker.js).
    Since 2026-10-04 the jobs run there, not in GitHub Actions; the container gets
    STATE_URL/STATE_TOKEN, so their presence says which pipeline to read."""
    job = wf.removesuffix(".yml")
    url = (os.environ["STATE_URL"].rstrip("/") + "/internal/jobs?ns=live&job="
           + urllib.parse.quote(job))
    req = urllib.request.Request(url, headers={
        "authorization": f"Bearer {os.environ['STATE_TOKEN']}",
        "user-agent": "henrys-match-engine-runner/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rows = json.loads(resp.read().decode())
    return [{"status": "completed",
             "conclusion": "success" if x["status"] == "done" else "failure",
             "createdAt": datetime.fromtimestamp(x["enqueued_at"], timezone.utc).isoformat()}
            for x in rows if x.get("status") in ("done", "failed")][:10]


def _runs(wf):
    if os.environ.get("STATE_URL") and os.environ.get("STATE_TOKEN"):
        return _cloudflare_runs(wf)
    out = subprocess.run(["gh", "run", "list", "--workflow", wf, "-L", "10", "--json",
                          "conclusion,status,createdAt"], capture_output=True, text=True,
                         timeout=60, check=True).stdout
    return [x for x in json.loads(out) if x.get("status") == "completed"]


def workflow_age(now):
    """Fill LAST_OK: hours since each workflow's newest successful run."""
    for wf in WORKFLOWS:
        try:
            ok = [x for x in _runs(wf) if x.get("conclusion") == "success"]
        except Exception:
            continue
        if ok:
            LAST_OK[wf] = _hours(now, _utc(ok[0]["createdAt"]))


def workflows(r, now):
    """The newest COMPLETED run of each workflow. Needs gh + GITHUB_TOKEN (in Actions)."""
    for wf, name in WORKFLOWS.items():
        try:
            runs = _runs(wf)
        except Exception as exc:
            r.add("Pipeline", name, "warn", f"could not read workflow runs ({type(exc).__name__})")
            continue
        if not runs:
            continue
        failed = 0
        for x in runs:
            if x.get("conclusion") == "failure":
                failed += 1
            else:
                break
        if failed >= 2:
            r.add("Pipeline", name, "fail", f"last {failed} runs failed")
        elif failed == 1:
            r.add("Pipeline", name, "warn", "last run failed")
        else:
            r.add("Pipeline", name, "ok", "last run succeeded")


def telegram(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("no Telegram secrets; alert not sent")
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML"}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=20) as r:
            print(f"Telegram alert sent (HTTP {r.status})")
    except Exception as exc:
        print(f"WARNING: Telegram alert failed ({type(exc).__name__})")


def run(now=None, check_workflows=True) -> dict:
    now = now or datetime.now(timezone.utc)
    r = Report()
    LAST_OK.clear()
    if check_workflows:
        workflow_age(now)
    soccer(r, now)
    nfl(r, now)
    line_sport(r, now, "NBA", "nba", "tipoff")
    line_sport(r, now, "MLB", "mlb", "tipoff")
    mlb_boxes(r, now)
    odds_budget(r)
    if check_workflows:
        workflows(r, now)
    status = max((c["status"] for c in r.checks), key=RANK.get, default="ok")
    return {"status": status, "checked_at": now.isoformat(timespec="seconds"), "checks": r.checks}


def main():
    prev = _read("data/health.json") or {}
    rep = run()
    problems = sorted((c["sport"], c["check"], c["status"]) for c in rep["checks"] if c["status"] != "ok")
    before = sorted((c["sport"], c["check"], c["status"]) for c in prev.get("checks", []) if c["status"] != "ok")
    rep["since"] = prev.get("since") if problems == before and prev.get("since") else rep["checked_at"]
    for c in rep["checks"]:
        print(f"[{c['status'].upper():4s}] {c['sport']:9s} {c['check']:16s} {c['message']}")
    print(f"overall: {rep['status']}")
    if problems != before or not OUT.exists():
        OUT.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(f"status changed -> wrote {OUT.relative_to(ROOT)}")
        bad = [c for c in rep["checks"] if c["status"] != "ok"]
        if bad:
            lines = [f"{'🔴' if c['status'] == 'fail' else '🟠'} <b>{c['sport']}</b> {c['check']}: {c['message']}" for c in bad]
            telegram("⚠️ <b>Henry's Engine health check</b>\n" + "\n".join(lines))
        elif before:
            telegram("✅ <b>Henry's Engine health check</b>: everything is back to normal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
