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
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jobs import ENV, ENV_DEFAULTS, JOBS  # noqa: E402

REPO = "guchhenry91/henrys-match-engine"
WORK = Path("/work")
CODE = WORK / "repo"
TRACKED = ("data", "data-raw")
SKIP_PARTS = {"_cache", "__pycache__", "_fdcache"}
CACHES = {"cache/soccerdata.tar.gz": Path.home() / "soccerdata",
          "cache/nflverse.tar.gz": CODE / "data-raw" / "nfl" / "_cache"}
LOG = []
RESTORED = {}                   # cache key -> fingerprint of what was restored


STEP_TIMEOUT = 20 * 60          # no single step may run longer than this
NL = chr(10)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)
    # Live progress: container output is not visible from the Worker, so every
    # line is reported as it happens (best effort -- never fails the job).
    try:
        if os.environ.get("JOB_ID"):
            api("POST", "/internal/job/progress", {"id": os.environ["JOB_ID"],
                                                   "log": NL.join(LOG)[-20000:]}, timeout=20)
    except Exception:
        pass


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def api(method, path, body=None, timeout=120):
    url = os.environ["STATE_URL"].rstrip("/") + path
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={
        "authorization": f"Bearer {os.environ['STATE_TOKEN']}",
        "content-type": "application/json", "x-state-ns": os.environ.get("STATE_NS", "shadow"),
        # Cloudflare's bot protection answers 403 to Python's default
        # "Python-urllib" user agent, so name the client explicitly.
        "user-agent": "henrys-match-engine-runner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


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


def get_file(path):
    # One chunk per request: a large cache in one response overruns the Worker.
    q = "/internal/state/file?path=" + urllib.parse.quote(path) + "&idx="
    first = api("GET", q + "0")
    if not first or first.get("data") is None:
        return None
    parts = [first["data"]] + [api("GET", q + str(i))["data"] for i in range(1, first["chunks"])]
    return base64.b64decode("".join(parts))


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


def overlay_state():
    """Lay the saved data over the downloaded code. Returns {path: repo sha}."""
    repo_sha = {k: sha(p.read_bytes()) for k, p in tracked_files().items()}
    manifest = api("GET", "/internal/state/manifest") or []
    used = kept_repo = 0
    for m in manifest:
        path = m["path"]
        if path.startswith("cache/"):
            continue
        current = repo_sha.get(path)
        if current is not None and m.get("repo_sha") and current != m["repo_sha"]:
            log(f"state: {path} changed in the repo since it was saved -- using the repo's copy")
            kept_repo += 1
            continue
        content = get_file(path)
        if content is None:
            continue
        dest = CODE / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        used += 1
    log(f"state: {used} saved file(s) applied, {kept_repo} newer in the repo")
    return repo_sha


def fingerprint(src):
    """Which files a cache holds, and how big -- enough to tell whether it changed."""
    if not src.exists():
        return None
    return sorted((p.relative_to(src).as_posix(), p.stat().st_size) for p in src.rglob("*") if p.is_file())


def restore_caches():
    for key, dest in CACHES.items():
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
    for key, src in CACHES.items():
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
            log(f"cache: {key} saved ({len(blob) // 1024} KB)")
        except Exception as exc:
            log(f"cache: {key} not saved ({type(exc).__name__})")


def snapshot():
    return {k: sha(p.read_bytes()) for k, p in tracked_files().items()}


def save_changes(before, repo_sha):
    after = tracked_files()
    changed = [k for k, p in after.items() if before.get(k) != sha(p.read_bytes())]
    for k in changed:
        put_file(k, after[k].read_bytes(), repo_sha.get(k, ""))
    log(f"state: saved {len(changed)} changed file(s)")
    return snapshot()


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
        for i, phase in enumerate(JOBS[job], 1):
            gate_failed = False
            for name, cmd, kind in phase:
                ok = run_step(name, cmd, env)
                if not ok and kind == "gate":
                    gate_failed = True
                    break
            if gate_failed:
                ok_all = False
                log(f"phase {i}: a gate step failed -- nothing from this phase is saved")
                break
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
        if status != "done":
            telegram(f"Match Engine job '{job}' failed on Cloudflare. Last lines:\n" + "\n".join(LOG[-6:]))
    sys.exit(0 if status == "done" else 1)


if __name__ == "__main__":
    main()
