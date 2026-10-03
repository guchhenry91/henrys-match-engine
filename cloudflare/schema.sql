-- Henry's Match Engine on Cloudflare D1 (database "henrys-match-engine").
-- ns = "live" (the real data the site serves) or "shadow" (a trial copy).

-- Every data file a job saved: data/*.json boards, data-raw logs, caches.
-- repo_sha = the hash of the repo's own copy when it was saved; if the repo's
-- copy later changes (a human commit), that copy wins (run_job.overlay_state).
CREATE TABLE IF NOT EXISTS state_files (
  ns TEXT NOT NULL, path TEXT NOT NULL, sha TEXT NOT NULL, repo_sha TEXT,
  chunks INTEGER NOT NULL, size INTEGER NOT NULL, updated_at INTEGER NOT NULL,
  PRIMARY KEY (ns, path)
);
-- File bodies, base64 in ~900 KB pieces (D1 rows top out at 2 MB).
CREATE TABLE IF NOT EXISTS state_chunks (
  ns TEXT NOT NULL, path TEXT NOT NULL, idx INTEGER NOT NULL, data TEXT NOT NULL,
  PRIMARY KEY (ns, path, idx)
);
-- The job queue: one container runs one job at a time, oldest first.
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, ns TEXT NOT NULL,
  status TEXT NOT NULL,            -- queued | running | done | failed
  enqueued_at INTEGER NOT NULL, started_at INTEGER, finished_at INTEGER, log TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, id);
