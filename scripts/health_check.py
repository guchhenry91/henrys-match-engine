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


# Sports whose jobs are paused for the off-season (worker.js PAUSED): their board is
# expected to age, so its freshness is not judged. Keep in step with worker.js.
PAUSED = {"MLB"}


def line_sport(r, now, sport, key, start_key):
    # NBA and MLB run twice/three times a day with a long overnight gap (MLB 22:00 ->
    # 14:00 is 16h, NBA 21:30 -> 15:00 is 17.5h), so a 14h warning fired every
    # morning on boards that were perfectly current. 20h clears the gap; a missed
    # run still shows by the next slot.
    d = board_age(r, now, sport, f"data/{key}/board.json", 20, 30,
                  active=sport not in PAUSED
                  and (_read(f"data/{key}/board.json") or {}).get("status") == "live")
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


def engines(r):
    """Every props board must be the REBUILT model's (third review, 2026-10-12): on
    12 Oct the NFL publish crashed and the retired model's board stayed up, looking
    current, for hours."""
    for sport, rel in (("NFL", "data/nfl/board.json"), ("NBA", "data/nba/board.json"),
                       ("MLB", "data/mlb/board.json"), ("Soccer", "data/leagues/player_picks.json")):
        d = _read(rel)
        if d is None:
            continue
        eng = d.get("engine")
        if eng:
            name = eng.get("name", "?") if isinstance(eng, dict) else eng
            r.add(sport, "engine", "ok", f"rebuilt model ({name})")
        else:
            r.add(sport, "engine", "fail", "board was not built by the rebuilt model -- its last publish failed or predates it")


def _upcoming(rows, now, key, within_h):
    """How many of `rows` start in the next `within_h` hours."""
    n = 0
    for x in rows or []:
        h = _hours(_utc(x.get(key)), now) if isinstance(x, dict) else None
        n += h is not None and 0 < h <= within_h
    return n


def props_output(r, now):
    """A props board that went EMPTY while games are coming (2026-10-18: NFL game day,
    every market priced, nothing published -- the owner found it, not the pipeline).
    Nothing published is sometimes honest (no pick clears the bar), so an empty board
    is a WARN; a crashed forecast or a broken join is a FAIL."""
    d = _read("data/nfl/board.json") or {}
    soon = _upcoming(d.get("games"), now, "kickoff", 36)
    st = d.get("props_status") or {}
    mk = st.get("markets") or {}
    errs = [k.removesuffix("_error") for k, v in mk.items() if k.endswith("_error") and v]
    broken = [k for k, v in mk.items() if isinstance(v, dict) and v.get("slate") and not v.get("in_forecast")]
    picks = sum(len((v or {}).get("picks") or []) for v in (d.get("props") or {}).values()
                if isinstance(v, dict))
    priced = sum(v.get("priced") or 0 for v in mk.values() if isinstance(v, dict))
    # Bookmaker LINES, not slate rows: the anytime-TD slate lists every player on the
    # game (155 for one Thursday game) whether or not a book prices him, so counting
    # the slate read "170 lines" when books had posted 15 (2026-10-21).
    lines = sum(v.get("with_book_price") or 0 for v in mk.values() if isinstance(v, dict))
    if st.get("forecast") or errs or broken:
        why = st.get("forecast") and "the forecast failed" or (
            f"pricing failed: {', '.join(errs)}" if errs else
            f"no player joined to the forecast: {', '.join(broken)}")
        r.add("NFL", "props output", "fail", why)
    elif soon and not picks:
        r.add("NFL", "props output", "warn",
              f"{soon} game(s) in 36h, {lines} bookmaker lines, {priced} priced, 0 picks published"
              if lines else f"{soon} game(s) in 36h and no bookmaker prop lines yet")
    elif soon:
        r.add("NFL", "props output", "ok", f"{picks} pick(s) for {soon} game(s) in 36h")

    u = _read("data/ucl/board.json") or {}
    soon = _upcoming(u.get("matches"), now, "kickoff", 48)
    if soon:
        pr = u.get("props")
        if pr is None:
            r.add("UCL", "props output", "fail", f"{soon} match(es) in 48h but the props section was not built")
        elif not pr.get("upcoming"):
            r.add("UCL", "props output", "warn", f"{soon} match(es) in 48h, 0 player picks")
        else:
            r.add("UCL", "props output", "ok", f"{len(pr['upcoming'])} player pick(s), {soon} match(es) in 48h")

    soon = sum(_upcoming((_read(f"data/leagues/{lg}.json") or {}).get("matches"), now, "date", 36)
               for lg in ("pl", "laliga", "bundesliga", "seriea", "ligue1"))
    if soon:
        n = len((_read("data/leagues/player_picks.json") or {}).get("upcoming") or [])
        r.add("Soccer", "props output", "ok" if n else "warn",
              f"{n} player pick(s), {soon} match(es) in 36h" if n else f"{soon} match(es) in 36h, 0 player picks")


# NBA and MLB were missing (found 2026-10-23): a pick frozen after tip-off there was
# never counted. Their entries carry "tipoff" where the others carry "kickoff".
PICK_LOGS = (["data-raw/nfl/picks_log.json", "data-raw/ucl/picks_log.json", "data-raw/ucl/props_log.json",
              "data-raw/nba/picks_log.json", "data-raw/mlb/picks_log.json"]
             + [f"data-raw/leagues/{lg}/{f}" for lg in ("pl", "laliga", "bundesliga", "seriea", "ligue1")
                for f in ("picks_log.json", "player_picks_log.json")])


def _entries(log):
    for k, e in (log or {}).items():
        if isinstance(e, dict) and "locked_at" in e:
            yield e
        elif isinstance(e, dict):                    # NFL: {"games": {...}, "props": {...}}
            yield from (x for x in e.values() if isinstance(x, dict) and "locked_at" in x)


def late_locks(r, now, days=1):
    """A pick frozen AFTER its kick-off is not a prediction (2026-10 soccer: props
    locked mid-match after failed refreshes). Counts the last `days` days' locks."""
    late, where = 0, set()
    for rel in PICK_LOGS:
        for e in _entries(_read(rel)):
            lk, ko = _utc(e.get("locked_at")), _utc(e.get("kickoff") or e.get("tipoff"))
            # Tainted picks count too: they are voided, so the record stays honest,
            # but each one is a pick the pipeline made too late (2026-10-18: ten NFL
            # props frozen after the 17:00 kick-offs, all void).
            if lk and ko and lk > ko and (_hours(now, lk) or 99 * days) <= 24 * days:
                late += 1
                where.add(rel.split("/")[1] + ("" if "leagues" not in rel else "/" + rel.split("/")[2]))
    # One day, not three: the check runs every few hours, and a three-day window kept
    # the alarm red for days over late locks already fixed (19 Oct).
    r.add("All", "late locks", "fail" if late else "ok",
          f"{late} pick(s) frozen after kick-off in the last {24 * days}h ({', '.join(sorted(where))})"
          if late else f"no pick frozen after kick-off in the last {24 * days}h")


# Every log the seal step covers (scripts/seal_picks.py).
SEAL_LOGS = list(PICK_LOGS) + ["data-raw/leagues/parlays_log.json"]


def seals(r):
    """TAMPER CHECK (tracking/seal.py): every frozen pick is sealed into a hash chain;
    a sealed pick whose line, side, chance or price changed -- or that vanished --
    means the record was edited after the fact."""
    from tracking import seal
    sealed, bad, gone = 0, [], []
    for rel in SEAL_LOGS:
        v = seal.verify(ROOT / rel)
        sealed += v["sealed"]
        bad += [f"{rel.split('/', 1)[1]}: {k}" for k in v["broken"]]
        gone += [f"{rel.split('/', 1)[1]}: {k}" for k in v["missing"]]
    if bad:
        r.add("All", "pick log seal", "fail",
              f"{len(bad)} sealed pick(s) edited after freezing: " + "; ".join(bad[:5]))
    elif gone:
        # A released lock (fixture time moved) is gone until it is frozen again.
        r.add("All", "pick log seal", "warn",
              f"{len(gone)} sealed pick(s) no longer in the log (released lock?): " + "; ".join(gone[:5]))
    else:
        r.add("All", "pick log seal", "ok", f"{sealed} frozen picks sealed; none changed since")


def nba_boxes(r, now):
    """Freshness of the NBA box scores the picks grade from (like mlb_boxes)."""
    try:
        from nba import current
        rows = current.player_games()
    except Exception as exc:                          # noqa: BLE001
        r.add("NBA", "box scores", "warn", f"could not read box scores ({type(exc).__name__})")
        return
    if rows.empty:
        return
    import pandas as pd
    last = pd.to_datetime(rows["GAME_DATE"], format="mixed").max()
    log = _read("data-raw/nba/picks_log.json") or {}
    waiting = [e for e in log.values() if isinstance(e, dict) and not e.get("graded")
               and (_hours(now, _utc(e.get("tipoff"))) or 0) > 18]
    if waiting and (now.date() - last.date()).days > 1:
        r.add("NBA", "box scores", "fail", f"newest box score {last.date()}, picks waiting on later games")
    else:
        r.add("NBA", "box scores", "ok", f"newest box score {last.date()}")


# PRE-START RUNS (cloudflare/worker.js preRun): an NFL run ~2h50 before each kickoff
# (the third odds check) and an NBA run ~75 min before each tip (the late freeze).
# 22 Oct's Thursday game got none and nothing noticed; picks still froze, from an
# older board. Window: (job, earliest, latest) minutes before the start.
PRE_RUN = {"NFL": ("nfl", "data-raw/nfl/picks_log.json", "kickoff", 200, 120),
           "NBA": ("nba", "data-raw/nba/picks_log.json", "tipoff", 95, 30)}
PRE_RUN_SINCE = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)   # when both were live


def pre_runs(r, now, jobs_for=None):
    """Warn when a game started in the last 24h without its pre-start run."""
    if jobs_for is None:
        if not (os.environ.get("STATE_URL") and os.environ.get("STATE_TOKEN")):
            return
        jobs_for = _cloudflare_jobs
    for sport, (job, rel, key, early, late) in PRE_RUN.items():
        starts = {_utc(e.get(key)) for e in _entries(_read(rel))}
        starts = sorted(t for t in starts if t and t >= PRE_RUN_SINCE and 0 < _hours(now, t) <= 24)
        if not starts:
            continue
        try:
            queued = [datetime.fromtimestamp(x["enqueued_at"], timezone.utc) for x in jobs_for(job)]
        except Exception as exc:                      # noqa: BLE001
            r.add(sport, "pre-start run", "warn", f"could not read the job history ({type(exc).__name__})")
            continue
        missed = [t for t in starts if not any(
            timedelta(minutes=late) <= t - q <= timedelta(minutes=early) for q in queued)]
        r.add(sport, "pre-start run", "warn" if missed else "ok",
              f"no pre-start run before {', '.join(t.strftime('%d %b %H:%M') for t in missed)} UTC"
              if missed else f"ran before all {len(starts)} start(s) in the last 24h")


def _cloudflare_jobs(job):
    url = (os.environ["STATE_URL"].rstrip("/") + "/internal/jobs?ns=live&job=" + urllib.parse.quote(job))
    req = urllib.request.Request(url, headers={
        "authorization": f"Bearer {os.environ['STATE_TOKEN']}",
        "user-agent": "henrys-match-engine-runner/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def unlocked_games(r, now, days=2):
    """A finished NBA or MLB game with no frozen pick at all (24 Oct 2026: World Series
    Game 1 was played and left no entry -- MLB had inherited the NBA's late freeze --
    while every check read ok). Each game gets a team-winner pick, so a played game
    missing from the log means its picks were never frozen."""
    import pandas as pd
    since = (now - timedelta(days=days)).date().isoformat()
    for sport, rel in (("NBA", "data-raw/nba/picks_log.json"), ("MLB", "data-raw/mlb/picks_log.json")):
        try:
            if sport == "NBA":
                from nba import current
                rows = current.player_games()
                played = {} if rows.empty else dict(zip(rows["GAME_ID"].astype(str), rows["GAME_DATE"].astype(str)))
            else:
                path = ROOT / "data-raw" / "mlb" / "current" / f"team_{now.year}.csv.gz"
                t = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["GAME_ID", "game_date"])
                played = dict(zip(t["GAME_ID"].astype(str).str.removeprefix("MLB"), t["game_date"].astype(str)))
        except Exception:                             # noqa: BLE001
            continue
        recent = {g for g, d in played.items() if d[:10] >= since}
        if not recent:
            continue
        log = _read(rel) or {}
        frozen = {str(e.get("game_id")) for e in log.values() if isinstance(e, dict) and e.get("locked_at")}
        missing = sorted(recent - frozen)
        r.add(sport, "picks frozen", "fail" if missing else "ok",
              f"{len(missing)} finished game(s) since {since} have no frozen pick: {', '.join(missing[:5])}"
              if missing else f"every game since {since} has frozen picks")


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
    nba_boxes(r, now)
    props_output(r, now)
    late_locks(r, now)
    unlocked_games(r, now)
    seals(r)
    if check_workflows:
        pre_runs(r, now)
    engines(r)
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
    # Written EVERY run (third review): written only on change, its checked_at read
    # 36 hours old while every check was in fact current. Telegram still only on change.
    OUT.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    if problems != before or not prev:
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
