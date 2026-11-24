"""Payout log. Only numbers the human typed; rates are never invented.

usage: python payouts.py add --platform tiktok --post-url URL --views 12000 --rate-per-1k 0.5 --campaign NAME [--status pending]
       python payouts.py report
File: payouts.csv in the current folder (override with --file).
"""
import argparse, csv, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

FIELDS = ["date", "campaign", "platform", "post_url", "views", "rate_per_1k", "est_payout", "status"]


def read(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="payouts.csv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("add")
    a_.add_argument("--platform", required=True); a_.add_argument("--post-url", required=True)
    a_.add_argument("--views", type=int, required=True); a_.add_argument("--rate-per-1k", type=float, required=True)
    a_.add_argument("--campaign", required=True); a_.add_argument("--status", default="pending")
    sub.add_parser("report")
    a = ap.parse_args()

    if a.cmd == "add":
        est = round(a.views / 1000 * a.rate_per_1k, 2)
        new = not Path(a.file).exists()
        with open(a.file, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            if new: w.writeheader()
            w.writerow({"date": date.today().isoformat(), "campaign": a.campaign, "platform": a.platform,
                        "post_url": a.post_url, "views": a.views, "rate_per_1k": a.rate_per_1k,
                        "est_payout": f"{est:.2f}", "status": a.status})
        print(f"added: {a.campaign} / {a.platform} {a.views} views x {a.rate_per_1k}/1k = {est:.2f} ({a.status})")
        return

    rows = read(a.file)
    if not rows:
        print("no payouts logged yet"); return
    tot = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        t = tot[r["campaign"]]; t[0] += 1; t[1] += int(r["views"]); t[2] += float(r["est_payout"])
    print(f"{'campaign':24} {'posts':>5} {'views':>9} {'est_payout':>11}")
    for c, (n, v, p) in sorted(tot.items()):
        print(f"{c[:24]:24} {n:>5} {v:>9} {p:>11.2f}")
    print(f"{'TOTAL':24} {sum(t[0] for t in tot.values()):>5} {sum(t[1] for t in tot.values()):>9} {sum(t[2] for t in tot.values()):>11.2f}")
    print("est_payout = views/1000 x the rate you typed. Estimates, not confirmed earnings.")


if __name__ == "__main__":
    main()
