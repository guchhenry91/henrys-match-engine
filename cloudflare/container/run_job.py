"""Run one Match Engine job inside the Cloudflare container, then save what changed.

Started by cloudflare/worker.js with:
  JOB          which job (cloudflare/container/jobs.py)
  JOB_ID       the queue row, reported back when finished
  STATE_URL    the Worker's base URL; STATE_TOKEN its internal bearer token
  STATE_NS     "live" (the real data) or "shadow" (a trial copy -- nothing live)
  plus the provider keys (API_FOOTBALL_KEY, API_NFL_KEY, ODDS_API_KEY, TELEGRAM_*).

1. CODE: download the newest main from the public GitHub repo (no login).
2. STATE: overlay the saved data files from D1 (via the Worker). A file saved by an
   earlier job wins -- unless the repo's own copy changed since (a human commit),
   in which case the repo's copy wins and the change is reported.
3. CACHES: restore the soccerdata and nflverse caches, so runs do not refetch.
4. RUN the job's phases; after each phase save every changed file under data/ and
   data-raw/ -- unless a gate step in that phase failed (nothing half-done saved).
5. Report the result to the Worker; on failure, alert Telegram.
"""
import base64
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobs import CACHE_FOR, ENV, ENV_DEFAULTS, JOBS, KEEP_ON_FAILURE  # noqa: E402

REPO = "guchhenry91/henrys-match-engine"
WORK = Path("/work")
CODE = WORK / "repo"
TRACKED = ("data", "data-raw")
SKIP_PARTS = {"_cache", "__pycache__", "_fdcache"}
CACHES = {"cache/soccerdata.tar.gz": Path.home() / "soccerdata",
          "cache/nflverse.tar.gz": CODE / "data-raw" / "nfl" / "_cache"}
# THE LIVE RECORD IS NEVER REPLACED BY AN OLDER COPY. Append-only logs, ledgers and
# the published boards live in D1; a copy committed to the repo (a local run, a
# manual deploy.py) must not overwrite them. A deliberate correction is applied only
# when cloudflare/state_overrides.json names the file AND the exact repo sha.
PROTECTED = re.compile(r"(^data/)|((picks_log|player_picks_log|six_scores_log|parlays_log|mispriced_log"
                       r"|telegram_sent|ledger|record_history|odds_history|out_since)\.json$)")
OVERRIDES = Path("cloudflare/state_overrides.json")
LOG = []
_last_post = [0.0]
RESTORED = {}                   # cache key -> fingerprint of what was restored


STEP_TIMEOUT = 20 * 60          # no single step may run longer than this
NL = chr(10)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)
    # Live progress: container output is not visible from the Worker, so every
    # line is reported as it happens (best effort -- never fails the job).
    # At most one update every 10 seconds: each is an HTTP call and a D1 write.
    if time.time() - _last_post[0] < 10:
        return
    _last_post[0] = time.time()
    try:
        if os.environ.get("JOB_ID"):
            api("POST", "/internal/job/progress", {"id": os.environ["JOB_ID"],
                                                   "log": NL.join(LOG)[-20000:]}, timeout=20, tries=1)
    except Exception:
        pass


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def api(method, path, body=None, timeout=120, tries=4):
    """Call the Worker. Transient failures (network, 5xx, 429) are retried with
    backoff: one blip must not abort a job whose work is already done, or leave a
    phase half-saved. A 4xx other than 429 is a real answer and is not retried."""
    url = os.environ["STATE_URL"].rstrip("/") + path
    data = None if body is None else json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "authorization": f"Bearer {os.environ['STATE_TOKEN']}",
            "content-type": "application/json", "x-state-ns": os.environ.get("STATE_NS", "shadow"),
            # Cloudflare's bot protection answers 403 to Python's default
            # "Python-urllib" user agent, so name the client explicitly.
            "user-agent": "henrys-match-engine-runner/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode() or "null")
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429 or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
        time.sleep(2 * 2 ** attempt)


def download_code():
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main"
    with urllib.request.urlopen(url, timeout=300) as r:
        raw = r.read()
    if CODE.exists():
        shutil.rmtree(CODE)
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        top = t.getmembers()[0].name.split("/")[0]
        t.extractall(WORK, filter="data")
    (WORK / top).rename(CODE)
    log(f"code: main downloaded ({len(raw) // 1024} KB)")


def tracked_files():
    out = {}
    for root in TRACKED:
        base = CODE / root
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not (set(p.relative_to(CODE).parts) & SKIP_PARTS) \
                    and not p.name.endswith(".tmp"):
                out[p.relative_to(CODE).as_posix()] = p
    return out


def get_file(path, tries=2):
    # One chunk per request: a large cache in one response overruns the Worker. The
    # whole file is checked against its saved sha, so a file replaced while it was
    # being read is read again, never returned spliced from two versions.
    q = "/internal/state/file?path=" + urllib.parse.quote(path) + "&idx="
    for _ in range(tries):
        first = api("GET", q + "0")
        if not first or first.get("data") is None:
            return None
        parts = [first["data"]]
        for i in range(1, first["chunks"]):
            parts.append((api("GET", q + str(i)) or {}).get("data") or "")
        content = base64.b64decode("".join(parts))
        if not first.get("sha") or sha(content) == first["sha"]:
            return content
    raise RuntimeError(f"{path}: changed while it was being read")


def put_file(path, content: bytes, repo_sha: str):
    b64 = base64.b64encode(content).decode()
    size = 900_000
    chunks = [b64[i:i + size] for i in range(0, len(b64), size)] or [""]
    if len(chunks) <= 2:
        api("PUT", "/internal/state/file", {"path": path, "sha": sha(content),
                                            "repo_sha": repo_sha, "chunks": chunks}, timeout=300)
        return
    for i, c in enumerate(chunks):          # big file: staged piece by piece, then swapped in
        api("PUT", "/internal/state/chunk", {"path": path, "idx": i, "data": c})
    api("POST", "/internal/state/commit", {"path": path, "sha": sha(content), "repo_sha": repo_sha,
                                           "chunks": len(chunks), "size": len(b64)})


def overrides():
    """{path: repo sha} the owner has approved to replace the live copy (rare)."""
    try:
        return json.loads((CODE / OVERRIDES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def overlay_state():
    """Lay the saved data over the downloaded code. Returns {path: repo sha}."""
    repo_sha = {k: sha(p.read_bytes()) for k, p in tracked_files().items()}
    manifest = api("GET", "/internal/state/manifest") or []
    approved = overrides()
    used = kept_repo = refused = 0
    for m in manifest:
        path = m["path"]
        if path.startswith("cache/"):
            continue
        current = repo_sha.get(path)
        if current is not None and m.get("repo_sha") and current != m["repo_sha"]:
            if PROTECTED.search(path) and approved.get(path) != current:
                log(f"WARNING state: {path} differs in the repo but is a live record -- "
                    "keeping the live copy (list it in cloudflare/state_overrides.json to replace it)")
                refused += 1
            else:
                log(f"state: {path} changed in the repo since it was saved -- using the repo's copy")
                kept_repo += 1
                # Store it now, so the site serves it too and a later revert of the
                # repo cannot bring the old saved copy back.
                put_file(path, (CODE / path).read_bytes(), current)
                continue
        content = get_file(path)
        if content is None:
            continue
        dest = CODE / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        used += 1
    log(f"state: {used} saved file(s) applied, {kept_repo} newer in the repo"
        + (f", {refused} live record(s) kept over the repo's copy" if refused else ""))
    return repo_sha


def fingerprint(src):
    """Which files a cache holds, their sizes and modification times. Size alone would
    miss a page re-fetched at the same length; a rewrite always moves the mtime."""
    if not src.exists():
        return None
    return sorted((p.relative_to(src).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in src.rglob("*") if p.is_file())


def job_caches():
    return {k: v for k, v in CACHES.items() if k in CACHE_FOR.get(os.environ.get("JOB", ""), ())}


def restore_caches():
    for key, dest in job_caches().items():
        try:
            blob = get_file(key)
        except Exception as exc:
            log(f"cache: {key} unavailable ({type(exc).__name__})")
            continue
        if not blob:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
            t.extractall(dest, filter="data")
        RESTORED[key] = fingerprint(dest)
        log(f"cache: {key} restored ({len(blob) // 1024} KB)")


def save_caches():
    for key, src in job_caches().items():
        if not src.exists() or not any(src.rglob("*")):
            continue
        if key in RESTORED and fingerprint(src) == RESTORED[key]:
            log(f"cache: {key} unchanged, not re-saved")   # 35 MB is ~50s to upload
            continue
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for p in sorted(src.rglob("*")):
                if p.is_file():
                    t.add(p, arcname=str(p.relative_to(src)))
        blob = buf.getvalue()
        try:
            put_file(key, blob, "")
            RESTORED[key] = fingerprint(src)
            log(f"cache: {key} saved ({len(blob) // 1024} KB)")
        except Exception as exc:
            log(f"cache: {key} not saved ({type(exc).__name__})")


def snapshot():
    return {k: sha(p.read_bytes()) for k, p in tracked_files().items()}


def save_changes(before, repo_sha, only=None):
    """Save every changed file (or, after a failed gate, only those matching `only`).
    A file the job deleted is deleted from D1 too, or the next run would bring it back."""
    after = tracked_files()
    changed = [k for k, p in after.items() if before.get(k) != sha(p.read_bytes())]
    gone = [k for k in before if k not in after]
    if only is not None:
        changed = [k for k in changed if k.startswith(only)]
        gone = []
    for k in changed:
        put_file(k, after[k].read_bytes(), repo_sha.get(k, ""))
    for k in gone:
        api("POST", "/internal/state/delete", {"path": k})
    log(f"state: saved {len(changed)} changed file(s)" + (f", removed {len(gone)}" if gone else ""))
    snap = snapshot()
    if only is not None:                 # unsaved changes stay "changed" for the next phase
        snap = {k: (v if k in changed else before.get(k, "")) for k, v in snap.items()}
    return snap


def run_step(name, cmd, env):
    log(f"step: {name}")
    # Fixed commands from jobs.py, run without a shell.
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable] + shlex.split(cmd)[1:] if cmd.startswith("python ") else shlex.split(cmd),
                           cwd=CODE, env=env, capture_output=True, text=True, timeout=STEP_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        log(f"step TIMED OUT after {STEP_TIMEOUT // 60} min: {name} :: {out.strip()[-300:]}")
        return False
    log(f"step finished in {time.time() - t0:.0f}s (exit {p.returncode})")
    tail = (p.stdout + p.stderr).strip().splitlines()[-25:]
    for line in tail:
        print("    " + line, flush=True)
    if p.returncode != 0:
        # The last lines only say THAT it failed (e.g. "0/5 publishes succeeded");
        # the per-league reasons are earlier, on the ABORT / error lines.
        why = [l for l in (p.stdout + p.stderr).splitlines()
               if any(w in l for w in ("ABORT", "Error", "WARNING", "FAILED"))][-15:]
        for line in why:
            log("    " + line[:300])
        log(f"step FAILED ({p.returncode}): {name} :: {' | '.join(tail[-3:])[:400]}")
    return p.returncode == 0


def telegram(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat or os.environ.get("STATE_NS") != "live":
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=20)
    except Exception:
        pass


def main():
    job = os.environ.get("JOB", "")
    status = "failed"
    try:
        if job not in JOBS:
            raise SystemExit(f"unknown job {job!r}")
        download_code()
        repo_sha = overlay_state()
        restore_caches()
        env = {**ENV_DEFAULTS, **os.environ, **ENV, "PYTHONPATH": str(CODE)}
        before = snapshot()
        ok_all = True
        # PHASES ARE INDEPENDENT. A failed gate stops the rest of ITS phase and keeps
        # its results unsaved, but the next phase still runs -- leagues.yml ran the
        # full refresh even when the fast lock job failed (`if: always()`), after the
        # lock job's failures cost 13 hours of boards on 2026-08-30.
        for i, phase in enumerate(JOBS[job], 1):
            gate_failed = False
            for name, cmd, kind in phase:
                ok = run_step(name, cmd, env)
                if not ok and kind == "gate":
                    gate_failed = True
                    break
            if gate_failed:
                ok_all = False
                log(f"phase {i}: a gate step failed -- its results are not saved, "
                    "except the credit ledgers")
                # The credits were spent whether or not the board published: the
                # ledger is how the daily cap counts them.
                before = save_changes(before, repo_sha, only=KEEP_ON_FAILURE)
            else:
                before = save_changes(before, repo_sha)
            save_caches()
        status = "done" if ok_all else "failed"
    except Exception:
        log("runner error: " + traceback.format_exc()[-1500:])
    finally:
        try:
            api("POST", "/internal/job/finish", {"id": os.environ.get("JOB_ID"), "status": status,
                                                  "log": "\n".join(LOG)[-20000:]})
        except Exception as exc:
            print(f"could not report the result ({exc})", flush=True)
        # Alert on a CHANGE of state only: a lock failing every 10 minutes would
        # otherwise send dozens of identical messages a day.
        prev_failed = os.environ.get("PREV_FAILED") == "1"
        if status != "done" and not prev_failed:
            telegram(f"Match Engine job '{job}' failed on Cloudflare. Last lines:\n" + "\n".join(LOG[-6:]))
        elif status == "done" and prev_failed:
            telegram(f"Match Engine job '{job}' is working again on Cloudflare.")
    sys.exit(0 if status == "done" else 1)


if __name__ == "__main__":
    main()
