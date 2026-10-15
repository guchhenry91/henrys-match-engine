"""Champions League player props: anytime goalscorer, 2+ shots, 1+ shot on target.

ONLY WHAT PASSED ITS TEST (scripts/ucl_props_backtest.py, 2026-10-13, 307 past
Champions League matches between big-five clubs, 2021-22 to 2025-26): priced from
league form alone, the published picks landed where they said --
  goal  166 picks, said 47.5%, landed 45.2% (league gate: 47.2 / 45.4)
  sot   458 picks, said 70.3%, landed 72.9% (league gate: 68.9 / 68.5)
2+ SHOTS IS TESTED AND GRADED ON ESPN (ucl/espn.py). bet365 settles shots on Opta,
which counts blocked shots; API-Football's player shots leave them out (57% landed for
74% said -- a definition gap, not a model one). On ESPN's counts, blocked included:
  shots 423 picks before lineups, said 72.9%, landed 70.5% -- PASS, and that is how
  these picks lock (no Champions League lineup feed). With the XI known it reads
  74.1 / 68.9, so the confirmed-starter reading runs hot; shots picks have no bet365
  price anyway and are proving picks only.

Scope: both clubs in the big five (ucl/props_names.py). The model (leagues/forecast.py)
needs Understat's per-match player data, which exists for nobody else.

Same rules as the league props (leagues/publish.py): likely starter (start >= 75%),
the market's bar, top 3 a match; a scorer pick checked against bet365's anytime price
(value at +3%, or a proving pick the model backs); on target has no bet365 price in
any feed, so its picks are PROVING picks -- frozen and graded, never offered as bets.
A player who does not take the field is VOID.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leagues import forecast as fc
from leagues import picks
from ucl.props_names import to_understat

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data-raw" / "ucl" / "props_log.json"
CACHE = ROOT / "data-raw" / "ucl" / "props_forecast.json"
STATS = ROOT / "data-raw" / "ucl" / "player_stats.json"
ENGINE = "forecast-v1-ucl"
MARKETS = {"goal": ("anytime_pct", 0.40), "shots": ("p_shots_2plus", 0.62),
           "sot": ("p_sot_1plus", 0.62)}
MIN_START = 75.0
PER_MATCH = 3
HORIZON_HOURS = 7 * 24.0     # shown up to a week ahead, like the soccer props (10 days)
FAR_HOURS = 48.0             # beyond this, nothing moves fast:
FAR_REBUILD_HOURS = 12.0     # ...the forecast is rebuilt twice a day (container cost)
REBUILD_HOURS = 3.0          # inside two days, every 3 hours...
NEAR_REBUILD_HOURS = 0.75    # ...or this long once a match is inside 3 hours (news moves)
DOUBT_FACTOR = 0.5
GRADE_AFTER_HOURS = 2.5


def _utc(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _norm(s) -> str:
    from nfl.odds import norm_name
    return norm_name(str(s or ""))


def _load(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _save(path: Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n",
                          encoding="utf-8")


# ---------------------------------------------------------------- features

def batches(fixtures: list) -> list:
    """Fixtures in date order, cut so no club appears twice in one batch (a club's
    other European game must not read as a game every player missed)."""
    out, cur, seen = [], [], set()
    for f in sorted(fixtures, key=lambda f: str(f["date"])):
        clubs = {f["home_us"], f["away_us"]}
        if clubs & seen:
            out.append(cur)
            cur, seen = [], set()
        cur.append(f)
        seen |= clubs
    if cur:
        out.append(cur)
    return out


def upcoming_rows(batch: list, season: int) -> pd.DataFrame:
    rows = []
    for f in batch:
        for team, opp, home in ((f["home_us"], f["away_us"], 1), (f["away_us"], f["home_us"], 0)):
            rows.append({"league": "UCL", "season": season, "match_id": f["mid"],
                         "date": pd.Timestamp(str(f["date"])[:10]), "team": team,
                         "opponent": opp, "is_home": home})
    return pd.DataFrame(rows)


def features(league: pd.DataFrame, batch: list, season: int) -> pd.DataFrame:
    """Entering features for one batch, from league matches before it only."""
    first = pd.Timestamp(min(str(f["date"])[:10] for f in batch))
    teams = {c for f in batch for c in (f["home_us"], f["away_us"])}
    d = league[(league["date"] < first) & (league["season"] >= season - 1)]
    pids = set(d.loc[d["team"].isin(teams), "player_id"])
    d = d[d["team"].isin(teams) | d["player_id"].isin(pids)]
    built = fc.build(d, upcoming_rows(batch, season))
    return built[built["is_next"]].copy()


# ---------------------------------------------------------------- forecast

def eligible(matches: list, now) -> list:
    """Board matches between two big-five clubs kicking off within the horizon."""
    out = []
    for m in matches:
        if not m.get("kickoff") or m.get("id") is None:
            continue
        h, a = to_understat(m["home"]), to_understat(m["away"])
        hours = (_utc(m["kickoff"]) - _utc(now)).total_seconds() / 3600.0
        if h and a and 0 < hours <= HORIZON_HOURS:
            out.append({"fid": str(m["id"]), "mid": -int(m["id"]), "date": m["kickoff"],
                        "kickoff": m["kickoff"], "home": m["home"], "away": m["away"],
                        "home_us": h, "away_us": a, "hours": hours})
    return out


def due(cache: dict, fixtures: list, now) -> bool:
    if not fixtures:
        return False
    have = set((cache.get("fixtures") or {}))
    if any(f["fid"] not in have for f in fixtures):
        return True
    try:
        age = (_utc(now) - _utc(cache["built_at"])).total_seconds() / 3600.0
    except Exception:                                      # noqa: BLE001
        return True
    first = min(f["hours"] for f in fixtures)
    every = (NEAR_REBUILD_HOURS if first <= 3.0 else REBUILD_HOURS if first <= FAR_HOURS
             else FAR_REBUILD_HOURS)
    return age >= every


def forecast(fixtures: list, news: dict, now, season: int) -> dict:
    """{fixture id: [player rows]} -- the heavy step (~2-3 minutes): fit the league
    props model, then price each fixture from league form before it."""
    year = season
    league = fc.load(range(year - fc.TRAIN_SEASONS, year + 1))
    if league.empty:
        return {}
    frame = fc.build(league)
    train = frame[~frame["is_next"] & (frame["season"] >= year - fc.TRAIN_SEASONS)]
    model = fc.PropForecast().fit(train)
    # Each player's last five league appearances, for the board's form tiles.
    recent = (league.sort_values("date").groupby("player_id").tail(5)
              .groupby("player_id")[["goals", "shots", "sot"]].agg(list).to_dict("index"))
    out = {}
    for b in batches(fixtures):
        rows = features(league, b, year)
        if rows.empty:
            continue
        roles = model.roles(rows)
        by_mid = {f["mid"]: f for f in b}
        for i, r in enumerate(rows.itertuples()):
            f = by_mid[r.match_id]
            club = f["home"] if r.team == f["home_us"] else f["away"]
            n = news.get(club) or {}
            name = _norm(r.player)
            if name in {_norm(x) for x in n.get("out") or []}:
                roles[i] = [1.0, 0.0, 0.0]
            elif name in {_norm(x) for x in n.get("doubt") or []}:
                roles[i, 1:] *= DOUBT_FACTOR
                roles[i, 0] = 1 - roles[i, 1:].sum()
        pred = model.predict(rows, roles)
        for i, r in enumerate(rows.itertuples()):
            f = by_mid[r.match_id]
            out.setdefault(f["fid"], []).append({
                "team": f["home"] if r.team == f["home_us"] else f["away"],
                "player": r.player, "player_id": int(r.player_id),
                "anytime_pct": round(100 * float(pred["goal"][i]), 1),
                "p_sot_1plus": round(100 * float(pred["sot"][i]), 1),
                "p_shots_2plus": round(100 * float(pred["shots"][i]), 1),
                "start_pct": round(100 * float(roles[i, 2]), 1),
                "appearance_pct": round(100 * float(roles[i, 1] + roles[i, 2]), 1),
                "penalty_taker": bool((r.pen_share or 0) >= 0.5),
                "last_five": {k: [int(x) for x in v]
                              for k, v in (recent.get(int(r.player_id)) or {}).items()}})
    return out


# ---------------------------------------------------------------- picks

def _scorer_price(store: dict, fid: str, player: str):
    from leagues import prop_odds
    prices = ((store.get("fixtures") or {}).get(str(fid)) or {}).get("anytime") or {}
    hits = prop_odds._find(prices, player)
    return hits[0] if len(hits) == 1 and hits[0] <= prop_odds.MAX_ODDS else None


def candidates(fixture: dict, rows: list, odds_store: dict) -> list:
    """The picks one fixture would publish, with the price check applied."""
    from leagues import forecast_board
    from tracking import pricecheck
    out = []
    for market, (field, bar) in MARKETS.items():
        ranked = sorted((r for r in rows if (r.get("start_pct") or 0) >= MIN_START),
                        key=lambda r: -(r.get(field) or 0))[:PER_MATCH]
        for r in ranked:
            prob = (r.get(field) or 0) / 100.0
            if prob < bar:
                continue
            pc = {"probability": prob, "p_model": prob, "market": market}
            if market == "goal":
                price = _scorer_price(odds_store, fixture["fid"], r["player"])
                if price:
                    pc.update(book_price=float(price), book="bet365",
                              book_p=forecast_board.scorer_fair(price),
                              probability=forecast_board.checked_p("goal", prob, price))
            ok = pricecheck.qualifies(pc, is_bet365=pc.get("book") == "bet365")
            no_price = pc["price_check"].get("reason") == "no price"
            if no_price:
                pc["pick_kind"] = "proving"            # on the model alone, as the league does
                pc["awaiting_price"] = True            # the card shows "worth it at X+"
            elif not ok or pc["probability"] < bar:
                continue
            out.append({"fixture_id": fixture["fid"], "kickoff": fixture["kickoff"],
                        "home": fixture["home"], "away": fixture["away"],
                        "market": market, "label": picks.PROP_MARKETS[market][2],
                        "player": r["player"], "team": r["team"],
                        "p_pick": round(float(pc["probability"]), 4), "p_model": round(prob, 4),
                        "start_pct": r.get("start_pct"), "appearance_pct": r.get("appearance_pct"),
                        "last_five": (r.get("last_five") or {}).get(picks.PROP_MARKETS[market][0], []),
                        "last_five_line": picks.PROP_MARKETS[market][1],
                        "penalty_taker": r.get("penalty_taker"),
                        "book_price": pc.get("book_price"), "price_check": pc["price_check"],
                        "pick_kind": pc.get("pick_kind"), "awaiting_price": pc.get("awaiting_price", False),
                        "bar": bar, "engine": ENGINE})
    return out


def _confidence(p: float) -> int:
    return 5 if p >= 0.70 else 4 if p >= 0.60 else 3 if p >= 0.50 else 2 if p >= 0.40 else 1


def key_of(pick: dict) -> str:
    return f"{pick['fixture_id']}:{pick['market']}:{pick['player']}"


def lock_one(log: dict, pick: dict, now) -> dict:
    """Freeze one published prop -- shared by publish and the fast locker."""
    key = key_of(pick)
    if key in log:
        return log[key]
    e = picks.lock_prop(log, key, market=pick["market"], player=pick["player"],
                        team=pick["team"], p_pick=pick["p_pick"],
                        confidence=_confidence(pick["p_pick"]), kickoff=pick["kickoff"],
                        now=now, bar=pick.get("bar"), appearance_pct=pick.get("appearance_pct"),
                        p_model=pick.get("p_model"))
    for k in ("fixture_id", "home", "away", "engine", "pick_kind", "price_check", "start_pct",
              "last_five", "last_five_line"):
        if pick.get(k) is not None:
            e[k] = pick[k]
    if pick.get("book_price"):
        e["odds"] = pick["book_price"]
    return e


# ---------------------------------------------------------------- grading

def fetch_stats(fids: list, client=None) -> dict:
    """API-Football per-player stats for finished fixtures we hold picks on."""
    store = _load(STATS, {"fixtures": {}})
    have = store.setdefault("fixtures", {})
    todo = [f for f in fids if f not in have]
    if not todo:
        return store
    if client is None:
        import os
        from leagues.api_football import Client
        if not os.environ.get("API_FOOTBALL_KEY"):
            return store
        client = Client(limit=len(todo) + 1)
    from scripts.fetch_ucl_player_history import parse
    for fid in todo:
        try:
            players = parse(client.get("fixtures/players", fixture=fid))
        except Exception as exc:                          # noqa: BLE001
            print(f"  UCL props: stats for {fid} unavailable ({exc})")
            break
        if players:
            have[fid] = {"players": players}
    _save(STATS, store)
    return store


def espn_stats(log: dict, keys: list, opener=None) -> dict:
    """ESPN player lines (shots INCLUDING blocked) for the fixtures of `keys`, cached in
    STATS under "espn:<fixture id>", players' clubs in the board's spelling. Never
    fatal: a match ESPN cannot be found for stays pending."""
    from ucl import espn
    store = _load(STATS, {"fixtures": {}})
    have = store.setdefault("fixtures", {})
    days = {}
    kw = {"opener": opener} if opener else {}
    for k in keys:
        e = log[k]
        fid = f"espn:{e['fixture_id']}"
        if fid in have:
            continue
        day = str(e["kickoff"])[:10]
        try:
            if day not in days:
                days[day] = espn.events_on(day, **kw)
            home, away = e.get("home") or "", e.get("away") or ""
            eid = espn.find_event(days[day], (home, to_understat(home)), (away, to_understat(away)))
            if not eid:
                continue
            players = []
            for p in espn.players(eid, **kw):
                club = home if espn.same_club(p["team"], home) else (
                    away if espn.same_club(p["team"], away) else p["team"])
                players.append({**p, "team": club, "minutes": 90 if p.get("played") else 0})
            have[fid] = {"event": eid, "players": players}
        except Exception as exc:                          # noqa: BLE001
            print(f"  UCL props: ESPN stats for {e.get('home')} v {e.get('away')} unavailable ({exc})")
    _save(STATS, store)
    return have


def find_player(name: str, squad: list):
    """This player's line in one club's matchday squad, or None if he is not in it.

    The two feeds spell players differently (Understat's "Mathis Cherki" is ESPN's
    "Rayan Cherki"), so a full-name join that fails falls back to the SURNAME, accepted
    only when exactly one squad member has it. Absent from the squad by both = not in
    the matchday squad = did not play."""
    from leagues.players import resolve_squad_name
    hit = resolve_squad_name(name, [p["player"] for p in squad])
    if hit:
        return next((p for p in squad if p["player"] == hit), None)
    last = _norm(name).split()[-1:] if _norm(name) else []
    same = [p for p in squad if last and _norm(p["player"]).split()[-1:] == last]
    return same[0] if len(same) == 1 else None


def grade(log: dict, now, client=None, espn_opener=None) -> int:
    """Grade every frozen prop whose match has finished. A player not in his club's
    matchday squad, or listed with no minutes, is VOID; a fixture whose stats are not
    filed stays pending. A 'did not play' void from the last 7 days is checked again,
    so a name the join missed is corrected rather than left void."""
    def due(e):
        hours = (_utc(now) - _utc(e["kickoff"])).total_seconds() / 3600
        if not e.get("graded"):
            return hours >= GRADE_AFTER_HOURS
        return bool(e.get("did_not_play")) and hours <= 7 * 24
    pending = [k for k, e in log.items() if not k.startswith("_") and due(e)]
    if not pending:
        return 0
    api = [k for k in pending if log[k]["market"] != "shots"]
    stats = dict(fetch_stats(sorted({log[k]["fixture_id"] for k in api}), client)["fixtures"]
                 if api else {})
    shots = [k for k in pending if log[k]["market"] == "shots"]
    if shots:
        stats.update(espn_stats(log, shots, espn_opener))
    n = 0
    for k in pending:
        e = log[k]
        fx = stats.get(("espn:" if e["market"] == "shots" else "") + e["fixture_id"])
        if not fx:
            continue
        squad = [p for p in fx["players"] if p["team"] == e["team"]]
        if not squad:
            continue                                      # club not in the filed stats yet
        p = find_player(e["player"], squad)
        base = {k2: v for k2, v in e.items()
                if k2 not in ("graded", "void", "did_not_play", "void_reason", "actual")}
        if not p or not p.get("minutes"):
            if e.get("graded"):
                continue                                  # still did not play: unchanged
            log[k] = picks.void_did_not_play(base)
        else:
            log[k] = picks.grade_prop(base, {"goals": p.get("goals", 0), "sot": p.get("sot", 0),
                                             "shots": p.get("shots", 0), "minutes": p["minutes"]})
            log[k]["matched_as"] = p["player"]
            if e.get("graded"):
                log[k]["regraded"] = "player was missed by the name join"
        n += 1
    return n


# ---------------------------------------------------------------- the board section

def build(matches: list, news: dict, now=None, season: int | None = None,
          lock_window: float = 2.0, odds_store: dict | None = None, client=None,
          espn_opener=None) -> dict:
    """The "props" section of data/ucl/board.json. Never fatal to the caller."""
    from ucl import config, market
    now = _utc(now if now is not None else datetime.now(timezone.utc))
    season = season or config.CURRENT_SEASON
    log = picks.load_log(LOG)
    fixtures = eligible(matches, now)
    cache = _load(CACHE, {})
    if due(cache, fixtures, now):
        try:
            got = forecast(fixtures, news, now, season)
            cache = {"built_at": now.isoformat(), "fixtures": got}
            _save(CACHE, cache)
            print(f"  UCL props: forecast {len(got)} fixture(s)")
        except Exception as exc:                          # noqa: BLE001
            print(f"  UCL props: forecast failed ({type(exc).__name__}: {exc}); keeping the last")
    store = odds_store if odds_store is not None else market.load()
    upcoming = []
    for f in fixtures:
        cands = candidates(f, (cache.get("fixtures") or {}).get(f["fid"]) or [], store)
        # ONE PICK PER PLAYER PER GAME (tracking/pricecheck.one_per_player): the most
        # likely of his markets -- unless another is already frozen.
        best = {}
        for c in cands:
            frozen = key_of(c) in log
            cur = best.get(c["player"])
            if cur is None or (frozen and key_of(cur) not in log) or \
                    (key_of(cur) not in log and c["p_pick"] > cur["p_pick"]):
                best[c["player"]] = c
        for pick in best.values():
            k = key_of(pick)
            if k in log:
                e = log[k]
                pick.update(p_pick=e["p_pick"], locked=True, locked_at=e.get("locked_at"),
                            pick_kind=e.get("pick_kind", pick.get("pick_kind")))
            elif f["hours"] <= lock_window:
                lock_one(log, pick, now)
                pick.update(locked=True)
            upcoming.append(pick)
    # A frozen pick stays on the board until it is graded, even if today's numbers
    # would no longer pick it.
    shown = {key_of(p) for p in upcoming}
    for k, e in log.items():
        if k.startswith("_") or e.get("graded") or k in shown or not e.get("fixture_id"):
            continue
        upcoming.append({"fixture_id": e["fixture_id"], "kickoff": e["kickoff"],
                         "home": e.get("home"), "away": e.get("away"), "market": e["market"],
                         "label": picks.PROP_MARKETS[e["market"]][2], "player": e["player"],
                         "team": e["team"], "p_pick": e["p_pick"], "p_model": e.get("p_model"),
                         "pick_kind": e.get("pick_kind"), "price_check": e.get("price_check"),
                         "book_price": e.get("odds"), "start_pct": e.get("start_pct"),
                         "last_five": e.get("last_five"), "last_five_line": e.get("last_five_line"),
                         "locked": True, "locked_at": e.get("locked_at"), "engine": ENGINE})
    try:
        graded = grade(log, now, client, espn_opener)
    except Exception as exc:                              # noqa: BLE001
        print(f"  UCL props: grading failed ({exc})")
        graded = 0
    upcoming = [p for p in upcoming if not (log.get(key_of(p)) or {}).get("graded")]
    picks.save_log(log, LOG)
    entries = [e for k, e in log.items() if not k.startswith("_")]
    settled = sorted((dict(e, key=k) for k, e in log.items()
                      if not k.startswith("_") and e.get("graded")),
                     key=lambda e: str(e.get("kickoff")), reverse=True)
    rec = {m: picks.record([e for e in entries if e.get("market") == m]) for m in MARKETS}
    upcoming.sort(key=lambda p: (str(p["kickoff"]), -p["p_pick"]))
    if fixtures or upcoming or graded:
        print(f"  UCL props: {len(upcoming)} pick(s) on {len(fixtures)} fixture(s); {graded} graded")
    return {"engine": ENGINE, "markets": {m: {"label": picks.PROP_MARKETS[m][2], "bar": b}
                                          for m, (_f, b) in MARKETS.items()},
            "scope": "both clubs in the big five leagues", "upcoming": upcoming,
            "record": rec, "settled": settled[:60],
            "test": _load(ROOT / "data-raw" / "ucl" / "props_backtest_report.json", {}).get("markets"),
            "forecast_at": cache.get("built_at")}
