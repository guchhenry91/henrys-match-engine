// Henry's Match Engine on Cloudflare -- one Worker that
//   1. serves the site: pages from the `_site` assets folder, and the boards
//      (data/*.json) from D1 once the jobs run here (JOBS_MODE = "live");
//   2. answers /api/soccer/live -- in-play scores from API-Football, the key kept
//      as a Worker secret, cached 30s so every viewer shares one upstream call;
//   3. RUNS the refresh jobs itself (like Edge Engine): an every-minute cron
//      queues the jobs that are due and starts ONE container at a time
//      (Dockerfile.jobs, cloudflare/container/run_job.py). No GitHub runs.
//
// JOBS_MODE: "off" (nothing runs), "shadow" (jobs run against a trial copy of the
// data in D1 -- nothing the site shows changes), "live" (the real data).
//
// WHY A QUEUE OF ONE. The jobs share files (the soccer job grades NFL picks; the
// freeze job and the soccer job both write the soccer pick logs). Running them one
// after another means every job starts from what the previous one saved.

import { Container, getContainer } from "@cloudflare/containers";

const LIVE_TTL = 30;
const STUCK_MS = 45 * 60 * 1000;          // a job running longer than this is killed
const BOOT_MS = 3 * 60 * 1000;            // a just-started container may not report running yet

export class JobRunner extends Container {
  sleepAfter = "50m";
  onStop({ exitCode, reason }) { console.log("job container stopped", { exitCode, reason }); }
  onError(error) { console.error("job container failed", error); }
}

// ---- the stats proxy (formerly nba-stats-proxy on Render) ---------------------------
// The Henryade betting dashboard's own backend: player gamelogs (nba_api, NHL,
// StatMuse), Understat team xG/scorers, and /api/config. Run UNCHANGED from its repo
// (Dockerfile.proxy) in one always-the-same container that sleeps when idle. The keys
// it reads are this Worker's own secrets, so moving it needed none added.
export class StatsProxy extends Container {
  defaultPort = 8080;
  sleepAfter = "15m";
  constructor(ctx, env) {
    super(ctx, env);
    this.envVars = { ODDS_API_KEY: env.ODDS_API_KEY || "", API_FOOTBALL_KEY: env.API_FOOTBALL_KEY || "" };
  }
}

// /api/soccer/live is answered by this Worker itself (soccerLive), for both sites.
const PROXY_PREFIXES = ["/api/nba/", "/api/nhl/", "/api/mlb/", "/api/soccer/"];
function isProxyPath(path) {
  if (path === "/api/soccer/live") return false;
  return path === "/health" || path === "/api/config" || PROXY_PREFIXES.some((p) => path.startsWith(p));
}

// ---- live scores -------------------------------------------------------------------
const LEAGUES = { 39: "PL", 140: "LALIGA", 78: "BUNDESLIGA", 61: "LIGUE1", 135: "SERIEA", 2: "UCL" };
const IN_PLAY = new Set(["1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT", "SUSP"]);

export function shapeLive(response) {
  const out = [];
  for (const row of response || []) {
    if (!row || typeof row !== "object") continue;
    const code = LEAGUES[(row.league || {}).id];
    if (!code) continue;
    const fixture = row.fixture || {}, status = fixture.status || {};
    if (!IN_PLAY.has(status.short)) continue;
    const teams = row.teams || {}, goals = row.goals || {};
    const home = (teams.home || {}).name, away = (teams.away || {}).name;
    if (!home || !away) continue;
    out.push({ id: fixture.id, league: code, home, away,
               home_goals: goals.home ?? null, away_goals: goals.away ?? null,
               status: status.short, elapsed: status.elapsed ?? null, kickoff: fixture.date ?? null });
  }
  return out;
}

function json(body, status = 200, extra = {}) {
  return new Response(JSON.stringify(body), {
    status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...extra },
  });
}

async function soccerLive(request, env, ctx) {
  const cache = caches.default;
  const key = new Request(new URL("/api/soccer/live", request.url).toString(), { method: "GET" });
  const hit = await cache.match(key);
  if (hit) return json({ ...(await hit.json()), cached: true });
  const apiKey = env.API_FOOTBALL_KEY || "";
  if (!apiKey) return json({ matches: [], available: false, reason: "API_FOOTBALL_KEY not configured" });
  let payload;
  try {
    const r = await fetch("https://v3.football.api-sports.io/fixtures?live=all",
                          { headers: { "x-apisports-key": apiKey } });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    payload = await r.json();
  } catch (e) {
    return json({ matches: [], available: false, reason: `upstream: ${e.name || "Error"}` });
  }
  const body = { matches: shapeLive(payload.response), available: true,
                 fetched_at: new Date().toISOString(), ttl: LIVE_TTL };
  ctx.waitUntil(cache.put(key, new Response(JSON.stringify(body), {
    headers: { "content-type": "application/json", "cache-control": `public, max-age=${LIVE_TTL}` } })));
  return json({ ...body, cached: false });
}

// ---- saved data (D1) -----------------------------------------------------------------
async function readFile(env, ns, path) {
  const meta = await env.DB.prepare("SELECT chunks FROM state_files WHERE ns=? AND path=?").bind(ns, path).first();
  if (!meta) return null;
  const rows = await env.DB.prepare("SELECT data FROM state_chunks WHERE ns=? AND path=? ORDER BY idx")
    .bind(ns, path).all();
  return rows.results.map((r) => r.data);
}

function b64ToBytes(chunks) {
  const bin = atob(chunks.join(""));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function serveData(request, env) {
  const path = decodeURIComponent(new URL(request.url).pathname).replace(/^\//, "");
  if ((env.JOBS_MODE || "off") === "live") {
    const chunks = await readFile(env, "live", path);
    if (chunks) {
      return new Response(b64ToBytes(chunks), { headers: {
        "content-type": path.endsWith(".json") ? "application/json; charset=utf-8" : "application/octet-stream",
        "cache-control": "no-cache" } });
    }
  }
  return serveAsset(request, env);
}

async function serveAsset(request, env) {
  const res = await env.ASSETS.fetch(request);
  const path = new URL(request.url).pathname;
  if (path.startsWith("/data/") || path === "/" || path.endsWith(".html")) {
    const out = new Response(res.body, res);
    out.headers.set("cache-control", "no-cache");
    return out;
  }
  return res;
}

// ---- internal API (the container's only door in) -------------------------------------
function authorised(request, env) {
  const want = env.STATE_TOKEN || "";
  const got = (request.headers.get("authorization") || "").replace(/^Bearer /, "");
  return want.length >= 32 && got === want;
}

async function internal(request, env, ctx, path) {
  if (!authorised(request, env)) return json({ error: "unauthorised" }, 401);
  const ns = request.headers.get("x-state-ns") === "live" ? "live" : "shadow";
  const url = new URL(request.url);
  if (path === "/internal/state/manifest" && request.method === "GET") {
    const rows = await env.DB.prepare("SELECT path, sha, repo_sha, size, updated_at FROM state_files WHERE ns=?")
      .bind(ns).all();
    return json(rows.results);
  }
  // LARGE FILES MOVE ONE CHUNK PER REQUEST. A 35 MB cache sent as one JSON body
  // overran the Worker (HTTP 500), so: GET ?idx=N returns one piece plus the count,
  // PUT /internal/state/chunk stages pieces under "<path>#new", and
  // /internal/state/commit swaps them in at once -- a half-finished upload can
  // never be read as the file.
  if (path === "/internal/state/file" && request.method === "GET" && url.searchParams.has("idx")) {
    const p = url.searchParams.get("path") || "";
    const meta = await env.DB.prepare("SELECT chunks FROM state_files WHERE ns=? AND path=?").bind(ns, p).first();
    if (!meta) return json(null);
    const row = await env.DB.prepare("SELECT data FROM state_chunks WHERE ns=? AND path=? AND idx=?")
      .bind(ns, p, Number(url.searchParams.get("idx"))).first();
    return json({ chunks: meta.chunks, data: row ? row.data : null });
  }
  if (path === "/internal/state/chunk" && request.method === "PUT") {
    const b = await request.json();
    if (!b.path || typeof b.data !== "string") return json({ error: "bad request" }, 400);
    if (Number(b.idx) === 0)                // a new upload clears any abandoned one
      await env.DB.prepare("DELETE FROM state_chunks WHERE ns=? AND path=?").bind(ns, b.path + "#new").run();
    await env.DB.prepare("INSERT OR REPLACE INTO state_chunks (ns,path,idx,data) VALUES (?,?,?,?)")
      .bind(ns, b.path + "#new", Number(b.idx), b.data).run();
    return json({ ok: true });
  }
  if (path === "/internal/state/commit" && request.method === "POST") {
    const b = await request.json();
    const staged = await env.DB.prepare("SELECT COUNT(*) AS n FROM state_chunks WHERE ns=? AND path=?")
      .bind(ns, b.path + "#new").first();
    if (!b.path || staged.n !== Number(b.chunks)) return json({ error: `staged ${staged.n} of ${b.chunks}` }, 409);
    await env.DB.batch([
      env.DB.prepare("DELETE FROM state_chunks WHERE ns=? AND path=?").bind(ns, b.path),
      env.DB.prepare("UPDATE state_chunks SET path=? WHERE ns=? AND path=?").bind(b.path, ns, b.path + "#new"),
      env.DB.prepare("INSERT OR REPLACE INTO state_files (ns,path,sha,repo_sha,chunks,size,updated_at) VALUES (?,?,?,?,?,?,?)")
        .bind(ns, b.path, b.sha, b.repo_sha || "", Number(b.chunks), Number(b.size), Math.floor(Date.now() / 1000)),
    ]);
    return json({ ok: true });
  }
  if (path === "/internal/state/file" && request.method === "GET") {
    const chunks = await readFile(env, ns, url.searchParams.get("path") || "");
    return json(chunks ? { chunks } : null);
  }
  if (path === "/internal/state/file" && request.method === "PUT") {
    const b = await request.json();
    if (!b.path || !Array.isArray(b.chunks)) return json({ error: "bad request" }, 400);
    const now = Math.floor(Date.now() / 1000);
    const stmts = [
      env.DB.prepare("DELETE FROM state_chunks WHERE ns=? AND path=?").bind(ns, b.path),
      ...b.chunks.map((c, i) => env.DB.prepare("INSERT INTO state_chunks (ns,path,idx,data) VALUES (?,?,?,?)")
        .bind(ns, b.path, i, c)),
      env.DB.prepare("INSERT OR REPLACE INTO state_files (ns,path,sha,repo_sha,chunks,size,updated_at) VALUES (?,?,?,?,?,?,?)")
        .bind(ns, b.path, b.sha, b.repo_sha || "", b.chunks.length, b.chunks.reduce((n, c) => n + c.length, 0), now),
    ];
    await env.DB.batch(stmts);
    return json({ ok: true });
  }
  if (path === "/internal/job/finish" && request.method === "POST") {
    const b = await request.json();
    await env.DB.prepare("UPDATE jobs SET status=?, finished_at=?, log=? WHERE id=?")
      .bind(b.status === "done" ? "done" : "failed", Math.floor(Date.now() / 1000), (b.log || "").slice(-20000), Number(b.id))
      .run();
    ctx.waitUntil(pump(env));               // start the next queued job straight away
    return json({ ok: true });
  }
  if (path === "/internal/job/progress" && request.method === "POST") {
    const b = await request.json();
    await env.DB.prepare("UPDATE jobs SET log=? WHERE id=? AND status='running'")
      .bind((b.log || "").slice(-20000), Number(b.id)).run();
    return json({ ok: true });
  }
  if (path === "/internal/kill" && request.method === "POST") {
    const running = await env.DB.prepare("SELECT id FROM jobs WHERE status='running'").all();
    for (const r of running.results) {
      try { await slotFor(env, r.id).destroy(); } catch (e) { /* already gone */ }
    }
    try { await getContainer(env.JOB_RUNNER, "runner-v1").destroy(); } catch (e) { /* old slot */ }
    await env.DB.prepare("UPDATE jobs SET status='failed', finished_at=?, log=COALESCE(log,'') || ? WHERE status='running'")
      .bind(Math.floor(Date.now() / 1000), " | killed by hand").run();
    return json({ ok: true });
  }
  if (path === "/internal/jobs" && request.method === "GET") {
    // ?job=nba&ns=live narrows it: the health check reads one job's history, which
    // the every-10-minute lock runs would otherwise push out of the newest 40.
    const job = url.searchParams.get("job"), jns = url.searchParams.get("ns");
    const rows = await env.DB.prepare(
      "SELECT id, job, ns, status, enqueued_at, started_at, finished_at, substr(log, -3000) AS log FROM jobs " +
      "WHERE (?1 IS NULL OR job=?1) AND (?2 IS NULL OR ns=?2) ORDER BY id DESC LIMIT 40").bind(job, jns).all();
    return json(rows.results);
  }
  if (path === "/internal/run" && request.method === "POST") {
    const b = await request.json();
    await enqueue(env, b.job, b.ns === "live" ? "live" : "shadow");
    ctx.waitUntil(pump(env));
    return json({ ok: true, queued: b.job });
  }
  return json({ error: "not found" }, 404);
}

// ---- the schedule ----------------------------------------------------------------------
// [hour or null, minute, job, weekday or null] in UTC -- the same slots the GitHub
// workflows used (and that nfl/nba/mlb book_lines RUN_SLOTS assume).
export const SLOTS = [
  ...[...Array(12).keys()].flatMap((i) => [[11 + i, 17, "leagues"], [11 + i, 47, "leagues"]]),
  ...[0, 3, 6, 9, 12, 15, 18, 21].map((h) => [h, 15, "leagues"]),
  [6, 0, "leagues", 2],
  ...[...Array(13).keys()].flatMap((i) => [5, 15, 25, 35, 45, 55].map((m) => [11 + i, m, "lock"])),
  [5, 0, "nfl"], [9, 0, "nfl"], [16, 0, "nfl"],
  [8, 30, "nfl_backtest", 2], [8, 30, "nfl", 2],
  [15, 0, "nba"], [21, 30, "nba"],
  [14, 0, "mlb"], [17, 30, "mlb"], [22, 0, "mlb"],
  ...[0, 3, 6, 9, 12, 15, 18, 21].map((h) => [h, 17, "health"]),
];

export function due(date) {
  const h = date.getUTCHours(), m = date.getUTCMinutes(), dow = date.getUTCDay();
  const out = [];
  for (const [hour, minute, job, weekday] of SLOTS) {
    if (minute !== m || (hour !== null && hour !== h)) continue;
    if (weekday != null && weekday !== dow) continue;
    if (!out.includes(job)) out.push(job);
  }
  return out;
}

async function enqueue(env, job, ns) {
  const waiting = await env.DB.prepare("SELECT id FROM jobs WHERE job=? AND ns=? AND status='queued'").bind(job, ns).first();
  if (waiting) return false;               // already waiting: one is enough
  await env.DB.prepare("INSERT INTO jobs (job, ns, status, enqueued_at) VALUES (?,?,'queued',?)")
    .bind(job, ns, Math.floor(Date.now() / 1000)).run();
  return true;
}

function containerEnv(env, row) {
  const pass = ["API_FOOTBALL_KEY", "API_NFL_KEY", "ODDS_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"];
  const out = { JOB: row.job, JOB_ID: String(row.id), STATE_NS: row.ns,
                STATE_URL: env.STATE_URL, STATE_TOKEN: env.STATE_TOKEN };
  // SHADOW RUNS GET NO PROVIDER KEYS: a trial copy must never spend API-Football,
  // API-NFL or Odds API credits a second time alongside the real pipeline.
  if (row.ns === "live") for (const k of pass) if (env[k]) out[k] = env[k];
  // TELEGRAM WAITS FOR THE CUTOVER. While GitHub still runs the same jobs, both
  // would send every pick and failure alert -- each message twice. Set
  // TELEGRAM_FROM_CLOUDFLARE = "on" once the GitHub schedules are off.
  if (env.TELEGRAM_FROM_CLOUDFLARE !== "on") { delete out.TELEGRAM_BOT_TOKEN; delete out.TELEGRAM_CHAT_ID; }
  if (row.ns !== "live") out.ODDS_API_ENABLED = "false";
  return out;
}

// ONE CONTAINER SLOT PER JOB. Cloudflare can keep a stopped container assignment
// on a long-lived slot after an image update, so start() "succeeds" and nothing
// runs -- seen here on 2026-10-03 (0 assigned for 21 minutes), and the reason
// Edge Engine renames its slot each version. A fresh slot per job cannot inherit
// a stale assignment; max_instances = 1 still keeps it to one job at a time.
const slotFor = (env, id) => getContainer(env.JOB_RUNNER, `job-${id}`);

export async function pump(env) {
  const running = await env.DB.prepare("SELECT id, started_at FROM jobs WHERE status='running' ORDER BY id LIMIT 1").first();
  if (running) {
    const runner = slotFor(env, running.id);
    const state = await runner.getState();
    const alive = ["running", "healthy", "stopping"].includes(state.status);
    const age = Date.now() - running.started_at * 1000;
    if (!alive && age < BOOT_MS) return "booting";        // just started: give it time to come up
    if (alive && age < STUCK_MS) return "busy";
    if (alive) await runner.destroy();
    await env.DB.prepare("UPDATE jobs SET status='failed', finished_at=?, log=COALESCE(log,'') || ? WHERE id=?")
      .bind(Math.floor(Date.now() / 1000), alive ? "\nkilled: ran past the time limit" : "\ncontainer exited without reporting", running.id).run();
  }
  const next = await env.DB.prepare("SELECT id, job, ns FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").first();
  if (!next) return "idle";
  await env.DB.prepare("UPDATE jobs SET status='running', started_at=? WHERE id=?")
    .bind(Math.floor(Date.now() / 1000), next.id).run();
  try {
    await slotFor(env, next.id).start({ envVars: containerEnv(env, next), enableInternet: true,
                                        labels: { job: next.job } });
  } catch (e) {
    await env.DB.prepare("UPDATE jobs SET status='failed', finished_at=?, log=? WHERE id=?")
      .bind(Math.floor(Date.now() / 1000), `start failed: ${e}`, next.id).run();
    return "start-failed";
  }
  return `started ${next.job}`;
}

export default {
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname;
    if (path === "/api/soccer/live") return soccerLive(request, env, ctx);
    if (isProxyPath(path)) return getContainer(env.STATS_PROXY, "web").fetch(request);
    if (path === "/api/health") return json({ ok: true, mode: env.JOBS_MODE || "off", at: new Date().toISOString() });
    if (path.startsWith("/internal/")) return internal(request, env, ctx, path);
    if (path.startsWith("/data/")) return serveData(request, env);
    return serveAsset(request, env);
  },

  async scheduled(controller, env, ctx) {
    const mode = env.JOBS_MODE || "off";
    if (mode === "off") return;
    const at = new Date(Math.round(controller.scheduledTime / 60000) * 60000);
    for (const job of due(at)) await enqueue(env, job, mode === "live" ? "live" : "shadow");
    ctx.waitUntil(pump(env));
  },
};
