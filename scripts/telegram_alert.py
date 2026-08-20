"""Send one plain alert to Telegram. For a scheduled task that STOOD DOWN.

WHY THIS EXISTS. The matchday-news task aborts when the working tree is dirty --
correct, since publishing from a dirty tree would ship someone's half-finished
model to the live site. But an abort writes "sweep skipped" into a report nobody
reads, so the task does not fail, it goes QUIET. On 2026-08-19 a stray publish
left the tree dirty at 17:05Z and every run after that stood down unnoticed until
the owner happened to ask why nothing was happening.

A silent stand-down is indistinguishable from a quiet matchday with no news to
report, which is the normal outcome. That ambiguity is the bug. One message costs
nothing and removes it.

Reuses the TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets already wired up for
telegram_picks.py, and like that script it exits 0 when they are absent -- a
missing notifier must never fail the run it is reporting on.

    python -m scripts.telegram_alert "matchday-news stood down: working tree dirty"
"""
import os
import sys
import urllib.parse
import urllib.request


def send(token: str, chat: str, text: str) -> int:
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=urllib.parse.urlencode(
            {"chat_id": chat, "text": text, "parse_mode": "HTML"}).encode(),
        method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.status


def main() -> int:
    message = " ".join(sys.argv[1:]).strip()
    if not message:
        print("usage: python -m scripts.telegram_alert \"<message>\"")
        return 2
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(f"No Telegram secrets set; alert not sent: {message}")
        return 0
    try:
        send(token, chat, f"⚠️ <b>Henry's Match Engine</b>\n{message}")
        print(f"alert sent: {message}")
    except Exception as exc:
        # Never let the notifier sink the thing it is reporting on.
        print(f"alert FAILED to send ({exc}): {message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
