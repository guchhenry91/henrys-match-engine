"""Atomic update+publish for the World Cup predictor.

The scheduled tasks gather data (edit data-raw/results.json, news.json, etc.),
then call THIS script to finish the job deterministically. It cannot leave the
live site out of sync: it always re-runs the model, commits any changes,
pushes, and triggers the Render deploy — and self-heals if a previous run
recorded results without publishing them.

Usage:  python deploy.py "optional commit message"
"""
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
HOOK_FILE = os.path.join(os.path.expanduser("~"), ".claude", "worldcup-deploy-hook.txt")
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "John", "GIT_AUTHOR_EMAIL": "guchhenry91@gmail.com",
           "GIT_COMMITTER_NAME": "John", "GIT_COMMITTER_EMAIL": "guchhenry91@gmail.com"}


def git(*args, check=True):
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, env=GIT_ENV)
    if check and r.returncode != 0:
        print(f"git {' '.join(args)} -> {r.returncode}\n{r.stderr.strip()}")
    return r


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "auto update: results + news"

    # 1. sync (autostash so local edits to results/news survive a rebase)
    git("pull", "--rebase", "--autostash", check=False)

    # 2. always regenerate predictions from current data (grades locked picks,
    #    re-rates Elo, re-sims knockout). This is what self-heals a stale publish.
    pred = subprocess.run([sys.executable, os.path.join(ROOT, "predict.py")],
                          capture_output=True, text=True)
    print(pred.stdout.strip() or pred.stderr.strip())
    if pred.returncode != 0:
        print("ABORT: predict.py failed — not deploying.")
        sys.exit(1)

    # 3. commit iff something changed
    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        git("add", "-A")
        git("commit", "-m", msg)
        push = git("push", "origin", "main", check=False)
        if push.returncode != 0:
            print("ABORT: git push failed — fix auth/remote before deploy.")
            sys.exit(1)
        print("Pushed:", msg)
    else:
        print("No data changes since last run.")

    # 4. ALWAYS trigger the deploy so the live site can never lag the repo.
    try:
        with open(HOOK_FILE, encoding="utf-8") as f:
            hook = f.read().strip()
        with urllib.request.urlopen(urllib.request.Request(hook, method="POST"), timeout=20) as resp:
            print(f"Deploy triggered: HTTP {resp.status}")
    except Exception as e:
        print(f"WARNING: deploy hook failed ({e}) — repo is current but live site may lag.")
        sys.exit(1)

    rec = json.load(open(os.path.join(ROOT, "data", "predictions.json"), encoding="utf-8"))["record"]
    print(f"RECORD: {rec['correct']}-{rec['wrong']} of {rec['total']} "
          f"({round(rec['correct'] / rec['total'] * 100) if rec['total'] else 0}%), "
          f"{rec['pending']} to play")


if __name__ == "__main__":
    main()
