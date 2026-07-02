"""Atomic update+publish for the World Cup predictor.

The scheduled tasks gather data (edit data-raw/*.json), then call THIS script
to finish deterministically: re-run the model, commit, push, trigger the
Render deploy. It self-heals a stale publish and refuses to half-complete.

Usage:  python deploy.py "commit message"        # stages data/ + data-raw/ only
        python deploy.py "commit message" --all  # stage everything (manual dev use)
"""
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
HOOK_FILE = os.path.join(os.path.expanduser("~"), ".claude", "worldcup-deploy-hook.txt")
LOCK_FILE = os.path.join(ROOT, ".deploy.lock")
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "John", "GIT_AUTHOR_EMAIL": "guchhenry91@gmail.com",
           "GIT_COMMITTER_NAME": "John", "GIT_COMMITTER_EMAIL": "guchhenry91@gmail.com"}


def git(*args, fatal=True):
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, env=GIT_ENV)
    if r.returncode != 0:
        print(f"git {' '.join(args)} -> {r.returncode}\n{r.stderr.strip()}")
        if fatal:
            print("ABORT: git step failed — not publishing a half-done state.")
            sys.exit(1)
    return r


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "auto update: results + news"
    stage_all = "--all" in sys.argv[2:]

    # concurrency guard: two scheduled tasks must not race the repo
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print("Another deploy.py run is in progress — skipping (it will publish).")
        sys.exit(0)

    try:
        # 1. sync; on a rebase conflict, bail out cleanly rather than commit markers
        pull = git("pull", "--rebase", "--autostash", fatal=False)
        if pull.returncode != 0:
            git("rebase", "--abort", fatal=False)
            print("ABORT: pull/rebase conflict — resolve manually before publishing.")
            sys.exit(1)

        # 2. regenerate predictions (grades locked picks, re-rates Elo, re-sims KO)
        pred = subprocess.run([sys.executable, os.path.join(ROOT, "predict.py")],
                              capture_output=True, text=True)
        print(pred.stdout.strip() or pred.stderr.strip())
        if pred.returncode != 0:
            print("ABORT: predict.py failed — not deploying.")
            sys.exit(1)

        # 3. commit iff something changed (scoped: the pipeline owns data only)
        paths = ["-A"] if stage_all else ["data", "data-raw"]
        git("add", *paths)
        if git("diff", "--cached", "--quiet", fatal=False).returncode != 0:
            git("commit", "-m", msg)
            git("push", "origin", "main")
            print("Pushed:", msg)
        else:
            print("No data changes since last run.")

        # 4. ALWAYS trigger the deploy so the live site can never lag the repo.
        with open(HOOK_FILE, encoding="utf-8-sig") as f:
            hook = f.read().strip()
        try:
            with urllib.request.urlopen(urllib.request.Request(hook, method="POST"),
                                        timeout=20) as resp:
                print(f"Deploy triggered: HTTP {resp.status}")
        except Exception as e:
            print(f"WARNING: deploy hook failed ({e}) — repo is current but live site may lag.")
            sys.exit(1)

        rec = json.load(open(os.path.join(ROOT, "data", "predictions.json"),
                             encoding="utf-8"))["record"]
        print(f"RECORD: {rec['correct']}-{rec['wrong']} of {rec['total']} "
              f"({round(rec['correct'] / rec['total'] * 100) if rec['total'] else 0}%), "
              f"{rec.get('void', 0)} void, {rec['pending']} pending")
    finally:
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
