"""Does the API-Sports account cover basketball and baseball ODDS with bet365?

Prints only. Uses /status (does not count against the daily quota) and, where the
plan is active, ONE /bookmakers request per sport to confirm bet365 is listed --
at most 2 requests per API, against a 100/day free allowance. The key is sent as
a header and never printed.
"""
import json
import os
import sys
import urllib.request

APIS = {"basketball": "https://v1.basketball.api-sports.io",
        "baseball": "https://v1.baseball.api-sports.io"}


def _get(base, path, key):
    req = urllib.request.Request(f"{base}/{path}", headers={"x-apisports-key": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    keys = {n: os.environ.get(n) for n in ("API_FOOTBALL_KEY", "API_NFL_KEY")}
    for kname, key in keys.items():
        if not key:
            print(f"{kname}: not set")
            continue
        for sport, base in APIS.items():
            try:
                st = _get(base, "status", key).get("response") or {}
                sub = st.get("subscription") or {}
                req = st.get("requests") or {}
                print(f"{kname} -> {sport}: plan={sub.get('plan')} active={sub.get('active')} "
                      f"ends={sub.get('end')} requests today={req.get('current')}/{req.get('limit_day')}")
                if sub.get("active"):
                    books = _get(base, "bookmakers", key).get("response") or []
                    names = [b.get("name") for b in books]
                    print(f"   bookmakers listed: {len(names)}; bet365 present: "
                          f"{any('bet365' in str(n).lower() for n in names)}")
            except Exception as exc:
                print(f"{kname} -> {sport}: {type(exc).__name__}: {str(exc)[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
