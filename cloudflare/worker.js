// Henry's Match Engine on Cloudflare: one Worker that
//   1. serves the static site (index.html, worldcup.html, data/*.json) from the
//      `_site` assets folder built by scripts/build_site.py,
//   2. answers /api/soccer/live -- in-play scores from API-Football, the key kept
//      as a Worker secret, cached 30s so every viewer shares one upstream call,
//   3. on its cron, STARTS the GitHub refresh workflows on time.
//
// WHY THE CRON. GitHub's own schedules were rationed and late: lock.yml's
// */10 schedule fired 3 of 97 slots (3.1%); NFL's 16:00 ran at 19:26, after
// kickoff; MLB's 14:00 never started on the first day of the playoffs. A
// workflow_dispatch from here starts within a minute or so. The SLOTS below are
// the same UTC times the workflows' RUN_SLOTS (nfl/nba/mlb book_lines.py) assume,
// so "the last run before kickoff" now means what the code thinks it means.

const REPO = "guchhenry91/henrys-match-engine";
const LIVE_TTL = 30;

// API-Football league ids: the five domestic leagues plus the Champions League
// (mirrors nba-stats-proxy/live.py, which this replaces for the site).
const LEAGUES = { 39: "PL", 140: "LALIGA", 78: "BUNDESLIGA", 61: "LIGUE1", 135: "SERIEA", 2: "UCL" };
// A ball is in play, including breaks. Finished results belong to grading, and
// postponed / not-started matches are never "live".
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
    out.push({
      id: fixture.id, league: code, home, away,
      // 0 is a real score; only null/undefined means missing.
      home_goals: goals.home ?? null, away_goals: goals.away ?? null,
      status: status.short, elapsed: status.elapsed ?? null, kickoff: fixture.date ?? null,
    });
  }
  return out;
}

function json(body, status = 200, extra = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...extra },
  });
}

async function soccerLive(request, env, ctx) {
  const cache = caches.default;
  const key = new Request(new URL("/api/soccer/live", request.url).toString(), { method: "GET" });
  const hit = await cache.match(key);
  if (hit) {
    const body = await hit.json();
    return json({ ...body, cached: true });
  }
  const apiKey = env.API_FOOTBALL_KEY || "";
  if (!apiKey) {
    // 200, not 500: the board renders normally without live data.
    return json({ matches: [], available: false, reason: "API_FOOTBALL_KEY not configured" });
  }
  let payload;
  try {
    const r = await fetch("https://v3.football.api-sports.io/fixtures?live=all", {
      headers: { "x-apisports-key": apiKey },
      cf: { cacheTtl: 0 },
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    payload = await r.json();
  } catch (e) {
    return json({ matches: [], available: false, reason: `upstream: ${e.name || "Error"}` });
  }
  const body = { matches: shapeLive(payload.response), available: true,
                 fetched_at: new Date().toISOString(), ttl: LIVE_TTL };
  ctx.waitUntil(cache.put(key, new Response(JSON.stringify(body), {
    headers: { "content-type": "application/json", "cache-control": `public, max-age=${LIVE_TTL}` },
  })));
  return json({ ...body, cached: false });
}

async function serveAsset(request, env) {
  const res = await env.ASSETS.fetch(request);
  const path = new URL(request.url).pathname;
  // Boards change every few minutes: always revalidate (as render.yaml did).
  if (path.startsWith("/data/") || path === "/" || path.endsWith(".html")) {
    const out = new Response(res.body, res);
    out.headers.set("cache-control", "no-cache");
    return out;
  }
  return res;
}

// --- the schedule ---------------------------------------------------------------
// Each slot: [hour or null (every hour), minute, workflow, inputs, weekday or null].
// Times are UTC and match the workflows' own cron lines and RUN_SLOTS.
export const SLOTS = [
  // Soccer refresh (leagues.yml): matchday window 11:00-22:59 at :17 and :47,
  // every 3 hours at :15 otherwise, and Tuesday 06:00 weekly refit.
  ...[...Array(12).keys()].flatMap((i) => [[11 + i, 17, "leagues.yml"], [11 + i, 47, "leagues.yml"]]),
  ...[0, 3, 6, 9, 12, 15, 18, 21].map((h) => [h, 15, "leagues.yml"]),
  [6, 0, "leagues.yml", null, 2],
  // Pick freezing (lock.yml), every 10 minutes through the match window -- the
  // schedule GitHub would not keep.
  ...[...Array(13).keys()].flatMap((i) => [5, 15, 25, 35, 45, 55].map((m) => [11 + i, m, "lock.yml"])),
  // NFL: 05:00, 09:00, 16:00; Tuesday 08:30 also re-runs the release gate.
  [5, 0, "nfl.yml"], [9, 0, "nfl.yml"], [16, 0, "nfl.yml"],
  [8, 30, "nfl.yml", { backtest: "true" }, 2],
  // NBA 15:00, 21:30; MLB 14:00, 17:30, 22:00.
  [15, 0, "nba.yml"], [21, 30, "nba.yml"],
  [14, 0, "mlb.yml"], [17, 30, "mlb.yml"], [22, 0, "mlb.yml"],
  // Self-check at :17 every 3 hours.
  ...[0, 3, 6, 9, 12, 15, 18, 21].map((h) => [h, 17, "health.yml"]),
];

export function due(date) {
  const h = date.getUTCHours(), m = date.getUTCMinutes(), dow = date.getUTCDay();
  const out = new Map();
  for (const [hour, minute, wf, inputs, weekday] of SLOTS) {
    if (minute !== m || (hour !== null && hour !== h)) continue;
    if (weekday != null && weekday !== dow) continue;
    out.set(wf, { ...(out.get(wf) || {}), ...(inputs || {}) });   // one dispatch per workflow
  }
  return out;
}

async function dispatch(env, workflow, inputs) {
  const token = env.GITHUB_DISPATCH_TOKEN;
  if (!token) {
    console.warn(`no GITHUB_DISPATCH_TOKEN; ${workflow} not started`);
    return false;
  }
  const r = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${workflow}/dispatches`, {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, accept: "application/vnd.github+json",
               "user-agent": "henrys-match-engine-cron", "x-github-api-version": "2022-11-28" },
    body: JSON.stringify({ ref: "main", inputs: inputs || {} }),
  });
  if (r.status !== 204) console.error(`dispatch ${workflow} failed: HTTP ${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.status === 204;
}

export default {
  async fetch(request, env, ctx) {
    const path = new URL(request.url).pathname;
    if (path === "/api/soccer/live") return soccerLive(request, env, ctx);
    if (path === "/api/health") return json({ ok: true, at: new Date().toISOString() });
    return serveAsset(request, env);
  },

  async scheduled(controller, env, ctx) {
    // The cron fires every minute ("* * * * *"); round to the minute it is for.
    const at = new Date(Math.round(controller.scheduledTime / 60000) * 60000);
    for (const [wf, inputs] of due(at)) ctx.waitUntil(dispatch(env, wf, inputs));
  },
};
