"""Send newly-LOCKED picks to Telegram, once each, the moment they freeze.

Reuses the same TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets already wired up
for failure alerts in leagues.yml. This is a separate channel of messages, not
a replacement: failure alerts say something broke, this says a pick is ready.

WHY "only when locked, only once": the pipeline republishes every 30 minutes
through the match window. A pick is PROVISIONAL for days before it locks (see
leagues.publish's LOCK_WINDOW_HOURS) and can change every run -- sending it
early would spam a moving target and could message a pick that later flips.
Sending it every run after it locks would spam the same, now-frozen pick
repeatedly. So: send exactly once, exactly when `provisional` first turns
false, and remember what's been sent so a later run never repeats it.
"""
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "leagues"
SENT_LOG = ROOT / "data-raw" / "leagues" / "telegram_sent.json"

LG_EMOJI = {"PL": "🏴", "LALIGA": "🇪🇸", "BUNDESLIGA": "🇩🇪", "LIGUE1": "🇫🇷"}
MARKET_NAME = {"goal": "to score", "shots": "2+ shots", "sot": "on target"}


def _load_sent() -> set:
    if not SENT_LOG.exists():
        return set()
    return set(json.loads(SENT_LOG.read_text(encoding="utf-8")).get("sent", []))


def _save_sent(sent: set) -> None:
    SENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    SENT_LOG.write_text(json.dumps({"sent": sorted(sent)}, indent=2), encoding="utf-8")


def _read(name: str) -> dict:
    path = OUT / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def newly_locked_best(sent: set) -> list[tuple[str, str]]:
    """(id, message) for each Best Pick that just locked and hasn't been sent."""
    out = []
    for u in _read("best").get("upcoming", []):
        if u.get("provisional"):
            continue
        key = f"best:{u['league_key']}:{u['id']}"
        if key in sent:
            continue
        emoji = LG_EMOJI.get(u["league_key"], "⚽")
        pct = round((u.get("p_pick") or 0) * 100)
        out.append((key, f"{emoji} <b>{u['home']} v {u['away']}</b>\n"
                          f"Pick: <b>{u['pick']}</b> ({pct}%)"))
    return out


def newly_locked_players(sent: set) -> list[tuple[str, str]]:
    out = []
    for u in _read("player_picks").get("upcoming", []):
        if u.get("provisional"):
            continue
        key = f"player:{u['league_key']}:{u['id']}:{u['market']}:{u['player']}"
        if key in sent:
            continue
        emoji = LG_EMOJI.get(u["league_key"], "⚽")
        pct = round((u.get("p_pick") or 0) * 100)
        market = MARKET_NAME.get(u["market"], u["market"])
        out.append((key, f"{emoji} <b>{u['player']}</b> ({u['team']}) {market} ({pct}%)"))
    return out


def send(token: str, chat: str, text: str) -> int:
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=urllib.parse.urlencode(
            {"chat_id": chat, "text": text, "parse_mode": "HTML"}).encode(),
        method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.status


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("No Telegram secrets set; skipping picks notification.")
        return 0

    sent = _load_sent()
    best = newly_locked_best(sent)
    players = newly_locked_players(sent)
    if not best and not players:
        print("No newly-locked picks to send.")
        return 0

    lines = []
    if best:
        lines.append("🎯 <b>Best Picks locked</b>")
        lines += [m for _, m in best]
    if players:
        if lines:
            lines.append("")
        lines.append("👤 <b>Player Picks locked</b>")
        lines += [m for _, m in players]
    text = "\n\n".join(lines)

    try:
        code = send(token, chat, text)
        print(f"Telegram picks message sent (HTTP {code}): "
              f"{len(best)} best + {len(players)} player picks")
    except Exception as exc:
        print(f"WARNING: Telegram picks send failed ({exc}) -- not marking as sent, "
              f"will retry next run.")
        return 0  # non-fatal: never block/fail the pipeline over a notification

    sent |= {k for k, _ in best} | {k for k, _ in players}
    _save_sent(sent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
