// Runs the real Worker against a real SQLite (node:sqlite) standing in for D1.
import { DatabaseSync } from "node:sqlite";
import { readFileSync } from "node:fs";

const [workerUrl, schemaPath] = process.argv.slice(2);
function makeD1() {
  const db = new DatabaseSync(":memory:");
  db.exec(readFileSync(schemaPath, "utf8"));
  const stmt = (sql, args = []) => ({
    bind: (...a) => stmt(sql, a),
    first: async () => { const r = db.prepare(sql).get(...args); return r === undefined ? null : { ...r }; },
    all: async () => ({ results: db.prepare(sql).all(...args).map((r) => ({ ...r })) }),
    run: async () => { db.prepare(sql).run(...args); return { success: true }; },
    _exec: () => db.prepare(sql).run(...args),
  });
  return { raw: db, prepare: (sql) => stmt(sql), batch: async (list) => { for (const s of list) s._exec(); } };
}

const cacheStore = new Map();
globalThis.caches = { default: { match: async (r) => cacheStore.get(r.url)?.clone(), put: async (r, res) => cacheStore.set(r.url, res) } };
let upstream = [], tg = [];
globalThis.fetch = async (u, init) => {
  u = String(u);
  if (u.includes("api.telegram.org")) { tg.push(JSON.parse(init.body).text); return new Response("{}"); }
  if (u.includes("raw.githubusercontent.com")) return new Response(globalThis.NEWS[u.split("/main/")[1]] || "{}");
  upstream.push(u);
  const regions = new URL(u).searchParams.get("regions"), markets = new URL(u).searchParams.get("markets");
  const cost = u.includes("/events?") || u.endsWith("/events") ? 0 : regions.split(",").length * markets.split(",").length;
  return new Response("[]", { headers: { "content-type": "application/json", "x-requests-last": String(cost), "x-requests-used": "1000" } });
};
globalThis.NEWS = {};

const W = await import(workerUrl);
const worker = W.default;
const ctxs = [];
const ctx = { waitUntil: (p) => ctxs.push(p) };
const settle = async () => { await Promise.all(ctxs.splice(0)); };
const env = (db, extra = {}) => ({ ODDS_API_KEY: "SECRET", DB: db, TELEGRAM_FROM_CLOUDFLARE: "on",
  TELEGRAM_BOT_TOKEN: "t", TELEGRAM_CHAT_ID: "c", ...extra });
let pass = 0, fail = 0;
const ok = (cond, name, extra = "") => { cond ? pass++ : fail++; console.log(cond ? "PASS" : "FAIL", name, extra); };
const get = (p, e) => worker.fetch(new Request("https://x.dev" + p, { headers: { "cf-connecting-ip": "1.2.3.4" } }), e, ctx);
const today = new Date().toISOString().slice(0, 10);

// ---- relay ----
{
  const db = makeD1(), e = env(db);
  let r = await get("/api/odds/v4/sports/baseball_mlb/events/abc123/odds?regions=us&markets=batter_hits,batter_rbis&apiKey=EVIL", e); await settle();
  ok(r.status === 200 && upstream.at(-1).includes("apiKey=SECRET") && !upstream.at(-1).includes("EVIL"), "relay adds the key, drops the caller's");
  ok(db.raw.prepare("SELECT credits FROM odds_relay WHERE day=?").get(today).credits === 2, "credits recorded as actually charged");
  r = await get("/api/odds/v4/sports/baseball_mlb/events/abc123/odds?regions=us&markets=player_xyz", e);
  ok(r.status === 400, "unknown market refused", r.status);
  r = await get("/api/odds/v4/sports/baseball_mlb/events/abc123/odds?regions=uk,eu,au,us&markets=h2h,totals,btts", e);
  ok(r.status === 400, "12-credit call refused (max 10)", r.status);
  r = await get("/api/odds/v4/sports/soccer_epl/odds/?regions=xx&markets=h2h", e);
  ok(r.status === 400, "unknown region refused", r.status);
  const before = upstream.length;
  r = await get("/api/odds/v4/sports/baseball_mlb/events/abc123/odds?regions=us&markets=batter_hits,batter_rbis&oddsFormat=american", e);
  ok(upstream.length === before, "format variations cannot bust the cache");
}
{ // reservation: 100 simultaneous 10-credit calls against a 600 cap
  const db = makeD1(), e = env(db);
  const rs = await Promise.all([...Array(100).keys()].map((i) =>
    get(`/api/odds/v4/sports/baseball_mlb/events/e${i.toString(16)}/odds?regions=us&markets=batter_hits,batter_rbis,batter_walks,batter_home_runs,batter_total_bases,batter_strikeouts,batter_runs_scored,batter_hits_runs_rbis,pitcher_strikeouts,pitcher_earned_runs`, e)));
  await settle();
  const passed = rs.filter((r) => r.status === 200).length;
  ok(passed === 60, "parallel burst stops exactly at the 600 cap", `${passed} x 10 credits got through`);
}
{ // month stop includes this call's cost
  const db = makeD1(), e = env(db);
  db.raw.prepare("INSERT INTO odds_relay VALUES (?, 0, 39995)").run(today);
  const r = await get("/api/odds/v4/sports/baseball_mlb/events/ff/odds?regions=us&markets=batter_hits,batter_rbis,batter_walks,batter_home_runs,batter_total_bases,batter_strikeouts", e);
  ok(r.status === 429, "a call that would cross 40,000 is refused", r.status);
  const r2 = await get("/api/odds/v4/sports/soccer_epl/events", e);
  ok(r2.status === 200, "free events call still allowed at the stop", r2.status);
}
{ // one daily limit for the account: the engine's ledger counts against the relay
  const db = makeD1(), e = env(db);
  const ledger = Buffer.from(JSON.stringify({ days: { [today]: { credits: 995 } } })).toString("base64");
  db.raw.prepare("INSERT INTO state_files VALUES ('live','data-raw/odds_api/ledger.json','x','',1,?,0)").run(ledger.length);
  db.raw.prepare("INSERT INTO state_chunks VALUES ('live','data-raw/odds_api/ledger.json',0,?)").run(ledger);
  const r = await get("/api/odds/v4/sports/baseball_mlb/events/ab/odds?regions=us&markets=batter_hits,batter_rbis,batter_walks,batter_home_runs,batter_total_bases,batter_strikeouts", e);
  ok(r.status === 429, "engine 995 + relay 6 > 1,000 refused", r.status);
  const r2 = await get("/api/odds/v4/sports/baseball_mlb/events/ab/odds?regions=us&markets=batter_hits", e);
  ok(r2.status === 200, "engine 995 + 1 still allowed", r2.status);
  const t = "x".repeat(40);
  const sp = await worker.fetch(new Request(`https://x.dev/internal/relay-spend?day=${today}`,
    { headers: { authorization: "Bearer " + t } }), { ...e, STATE_TOKEN: t }, ctx);
  ok((await sp.json()).credits === 1, "relay spend reported to the engine");
}
{ // rate limit binding honoured
  const db = makeD1(); let n = 0;
  const e = env(db, { RELAY_LIMIT: { limit: async () => ({ success: ++n <= 2 }) } });
  const a = await get("/api/odds/v4/sports/rl_a/events", e), b = await get("/api/odds/v4/sports/rl_b/events", e);
  const c = await get("/api/odds/v4/sports/rl_c/events", e);
  ok(a.status === 200 && b.status === 200 && c.status === 429, "per-visitor rate limit", [a.status, b.status, c.status].join(","));
}

// ---- queue ----
{
  const db = makeD1();
  const started = [];
  const slots = {};
  const e = env(db, { JOB_RUNNER: {}, STATE_URL: "u", STATE_TOKEN: "x".repeat(40) });
  // pump uses getContainer from the stub: started / alive controlled here
  globalThis.__containers = { start: (name, opts) => started.push([name, opts.envVars.JOB, opts.envVars.PREV_FAILED]),
                              state: (name) => slots[name] || "running" };
  const now = Math.floor(Date.now() / 1000);
  db.raw.prepare("INSERT INTO jobs (job, ns, status, enqueued_at) VALUES ('leagues','live','queued',?)").run(now - 60);
  db.raw.prepare("INSERT INTO jobs (job, ns, status, enqueued_at) VALUES ('lock','live','queued',?)").run(now - 30);
  await W.pump(e);
  ok(started[0][1] === "lock", "lock jumps the queue", started[0][1]);
  const r1 = await W.pump(e);
  ok(r1 === "busy" && started.length === 1, "only one job at a time", r1);
  // the running lock dies without reporting, past boot time
  db.raw.prepare("UPDATE jobs SET started_at=? WHERE job='lock'").run(now - 600);
  slots["job-2"] = "stopped";
  tg = [];
  await W.pump(e);
  ok(db.raw.prepare("SELECT status FROM jobs WHERE job='lock'").get().status === "failed", "dead container marked failed");
  ok(tg.length === 1 && tg[0].includes("lock"), "worker alerts on a dead container", tg[0]);
  ok(started[1][1] === "leagues" && started[1][2] === "0", "next job starts; its previous-failed flag is right", JSON.stringify(started[1]));
  // a leagues run at 50 min is NOT killed (limit 110), at 120 it is
  db.raw.prepare("UPDATE jobs SET started_at=? WHERE job='leagues'").run(now - 50 * 60);
  ok((await W.pump(e)) === "busy", "a 50-minute leagues run is left alone");
  db.raw.prepare("UPDATE jobs SET started_at=? WHERE job='leagues'").run(now - 120 * 60);
  await W.pump(e);
  ok(db.raw.prepare("SELECT status FROM jobs WHERE job='leagues'").get().status === "failed", "a 120-minute leagues run is killed");
  // unknown job names refused
  const bad = await worker.fetch(new Request("https://x.dev/internal/run", { method: "POST",
    headers: { authorization: "Bearer " + "x".repeat(40) }, body: JSON.stringify({ job: "rm -rf" }) }), e, ctx);
  ok(bad.status === 400, "unknown job name refused", bad.status);
  const wrong = await worker.fetch(new Request("https://x.dev/internal/jobs", { headers: { authorization: "Bearer " + "y".repeat(40) } }), e, ctx);
  ok(wrong.status === 401, "wrong token refused", wrong.status);
}

// ---- watchdog ----
{
  const db = makeD1(), e = env(db); tg = [];
  const now = Date.UTC(2026, 9, 4, 16, 7);
  const t = Math.floor(now / 1000);
  for (const [job, hAgo] of [["lock", 1], ["leagues", 1], ["health", 2], ["nfl", 5], ["nba", 5], ["mlb", 30]])
    db.raw.prepare("INSERT INTO jobs (job, ns, status, enqueued_at, finished_at) VALUES (?,'live','done',?,?)").run(job, t - hAgo * 3600, t - hAgo * 3600);
  let stale = await W.watchdog(e, now);
  ok(stale.length === 1 && stale[0].startsWith("mlb") && tg.length === 1, "watchdog names only the stale job", JSON.stringify(stale));
  await W.watchdog(e, now + 15 * 60000);
  ok(tg.length === 1, "and does not repeat it within 12 hours", tg.length);
  stale = await W.watchdog(e, Date.UTC(2026, 9, 4, 8, 7));
  ok(!stale.some((s) => s.startsWith("lock")), "lock is not judged overnight");
}

// ---- news poll ----
{
  const db = makeD1(), e = env(db);
  globalThis.NEWS = { "data-raw/nba/news.json": "a" };
  let q = await W.newsPoll(e);
  ok(q.length === 0, "first poll only records hashes", JSON.stringify(q));
  globalThis.NEWS = { "data-raw/nba/news.json": "b" };
  q = await W.newsPoll(e);
  ok(q.length === 1 && q[0] === "nba" && db.raw.prepare("SELECT COUNT(*) n FROM jobs WHERE job='nba' AND status='queued'").get().n === 1,
     "changed NBA news queues the NBA job", JSON.stringify(q));
}
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
