# Presentation-only patch — proposed, NOT applied (revised)

**The previous version of this document was wrong and has been replaced.** It was
drafted against `openMatch()` (the single-match detail modal) using local variables
(`sc`, `bestPct`, `condDiffers`) and a `.dhead` container that belong to that function's
scope. The user correctly flagged this doesn't match where fixtures are actually
rendered. There is no function named `renderLeague()` in the current working tree —
the real one is **`viewLeague(k)`** (`index.html`, current lines 477–495), which builds
the per-fixture row list. This patch targets that function instead, verified against
the actual current working-tree code (re-read immediately before drafting this).

## What this does NOT do

No probability, lambda, rho, or grid value changes — display only. Does not manipulate
or suppress 1-1 or any other score. `.psc`'s microcopy already correctly identifies the
headline number as "the largest individual cell" territory; the new per-card sentence
below makes that explicit in words too, rather than leaving it implicit.

## Verified-real symbols only (per instruction — checked against the current file, not memory)

- `esc()` — global helper, line 225: `const esc=s=>String(s==null?"":s).replace(...)`
- `pc()` — global helper, line 227: `const pc=p=>Math.round((p||0)*100)`. **Takes a raw
  0–1 probability and returns a rounded percentage. The correct call is `pc(p.p_pick)`,
  not `pct(100 * p.p_pick)`** — there is no function named `pct` anywhere in this file,
  and even substituting `pc`, multiplying by 100 first would double-scale the result
  (`pc(0.65)` → `65`, correct; `pc(100*0.65)` → `pc(65)` → `6500`, wrong). This is
  flagged rather than followed literally because implementing it as specified would be
  a real bug. Every existing call site (`viewToday()` line 324, `viewBest()` line 340,
  `viewLeague()` line 492) already uses the bare `pc(p.p_pick)` form.
- `p.pick`, `p.p_pick`, `p.score`, `p.top_scores` — all real fields, confirmed against
  live `data/leagues/pl.json` output (checked directly, not assumed):
  `{"score": "2-0", "top_scores": [{"score":"2-0","pct":13.9,...}, ...3 entries total], "pick": "Arsenal", "p_pick": 0.7741}`

## The four required elements, mapped to `viewLeague()`'s actual per-row scope

1. **Match Pick and p_pick** — already displayed today: `pick ${esc(p.pick)}` in `.info`,
   and `${pc(p.p_pick)}%` in `.conf`. No change needed; not touched by this diff.
2. **Highest-Probability Exact Score (`p.top_scores[0]`)** — already displayed via the
   existing `.psc` chip (`best.score` + percentage). Only its `<small>` label text
   changes, from "guess" to "top score" — a one-word correction, not a rebuild.
3. **Projected Score If Pick Lands (`p.score`)** — new. "Lands" instead of "Wins," a
   fixed label with no team name folded into a conjugated verb, so it reads correctly
   whether the pick is a team or a draw (no "If Draw wins" problem).
4. **Top Exact Scores (the complete `top_scores` list)** — new. Currently 3 entries per
   fixture (`top_scorelines(grid, n=3)` in `leagues/model.py`); rendered as a compact
   inline list, not a repeat of the headline chip. Named "Top Exact Scores," not "Other
   Plausible Scores," because the headline score is NOT excluded from this list (per
   instruction #7's own rule).

Per-card short text (instruction #5, exact wording): *"The highest-probability exact
score can differ from the most likely match result."* The longer explanation lives in
ONE `<details>` guide above the fixture list, not repeated per card (instruction #6).

## Exact minimal diff (verified by rendering — see below, not applied to `index.html`)

```diff
 function viewLeague(k){
   const d=DATA[FILES[k]]; if(!d) return emptyState();
   const ms=(d.matches||[]).filter(m=>!m.result);
   if(!ms.length) return emptyState();
   // Fixtures only -- the full standings live in the Tables tab, not repeated here.
-  let h=`<div class="eyebrow" style="color:${LC[k]}">${esc(LG[k])} · upcoming fixtures</div><div class="glass rowc">${ms.slice(0,16).map(m=>{const p=m.prediction||{};
+  let h=`<div class="eyebrow" style="color:${LC[k]}">${esc(LG[k])} · upcoming fixtures</div>
+    <details class="scoreguide"><summary>What do these scores mean?</summary>
+      <p>Match Pick is which result (win / draw / win) the model rates most likely, with its probability.
+      Highest-Probability Exact Score is the single most likely FINAL SCORE among every possible scoreline --
+      a different, narrower question. Projected Score If Pick Lands is the most likely exact score GIVEN that
+      the Match Pick comes true. These can genuinely disagree: in a close match many similarly-likely low
+      scores compete for the top spot, so the match-result favourite and the single most likely scoreline
+      often aren't the same score. Top Exact Scores lists the leading alternatives and their own chances, so
+      you can see how thin the favourite really is.</p></details>
+    <div class="glass rowc">${ms.slice(0,16).map(m=>{const p=m.prediction||{};
     // The BEST single exact-score guess is top_scores[0] -- the true highest-
     // probability scoreline, unconditional. p.score is a DIFFERENT number (the
     // most likely score GIVEN the win/draw/loss pick) and is deliberately not
     // shown here, since the two can disagree and only one is the model's actual
     // best guess at the final score.
     const best=(p.top_scores||[])[0];
+    const topScores=(p.top_scores||[]).map(t=>`${esc(t.score)} ${Math.round(t.pct||0)}%`).join(" · ");
     return `<div class="rw" onclick="openMatch('${k}',${m.id})"><span class="lgtag" style="background:${LC[k]}"></span>
-      <div class="info"><b>${esc(m.home)} v ${esc(m.away)}</b><span>MW${m.matchweek} · pick ${esc(p.pick)}</span></div>
-      <span class="psc">${esc(best?best.score:"–")}<small>${best?Math.round(best.pct)+"% guess":"score"}</small></span>
+      <div class="info"><b>${esc(m.home)} v ${esc(m.away)}</b><span>MW${m.matchweek} · pick ${esc(p.pick)}</span>
+        <div class="scoredetail">
+          <span>Projected Score If Pick Lands: <b>${esc(p.score||"–")}</b></span>
+          ${topScores?`<span>Top Exact Scores: ${topScores}</span>`:""}
+        </div>
+        <p class="cardnote">The highest-probability exact score can differ from the most likely match result.</p></div>
+      <span class="psc">${esc(best?best.score:"–")}<small>${best?Math.round(best.pct)+"% top score":"score"}</small></span>
       <div class="conf" style="min-width:48px"><b>${pc(p.p_pick)}%</b></div><span class="chev">›</span></div>`;}).join("")}</div>
     <p class="note">Best single-score guess and its own chance, plus the win %. No single score is likely -- see each match for the full spread. Full standings are in the Tables tab.</p>`;
   return h;
 }
```

Plus new CSS (inserted after the existing `.note{...}` rule, no existing rule changed):

```diff
   .note{font-size:12.5px;color:var(--sub);font-weight:600;text-align:center;padding:10px 24px 0;line-height:1.55}
+  .scoreguide{margin:0 0 8px;font-size:12px;color:var(--sub)}
+  .scoreguide summary{cursor:pointer;font-weight:700;color:var(--cyan);list-style:none;padding:2px 0}
+  .scoreguide summary::-webkit-details-marker{display:none}
+  .scoreguide summary::before{content:"▸ ";display:inline-block;transition:transform .15s}
+  .scoreguide[open] summary::before{transform:rotate(90deg)}
+  .scoreguide p{margin-top:6px;line-height:1.5}
+  .scoredetail{display:flex;flex-direction:column;gap:2px;margin-top:4px;font-size:11px;color:var(--sub);font-weight:600}
+  .cardnote{font-size:10.5px;color:var(--sub);font-weight:550;margin-top:4px;line-height:1.4;opacity:.8}
```

## Rendered verification

Applied to a throwaway, untracked local copy (`index_preview.html`, never committed,
deleted after verification — `git status` confirms zero changes to the tracked
`index.html` throughout). Served over a local static file server so the real
`data/leagues/*.json` loaded normally, then exercised via `render('PL')`.

**Pixel screenshots were not obtainable** — the Browser pane was not displayed on the
user's side during this session, and screenshot/click actions time out without a
composited frame. Verified instead via the actual rendered DOM and extracted text at
both target widths, plus explicit overflow checks:

| Width | Content order/completeness | Horizontal overflow | Row height |
|---|---|---|---|
| 1280×900 (desktop) | Correct — all 10 PL fixtures, all 4 new fields present in spec order | None (`scrollWidth` 1265 < 1280) | 162px |
| 375×812 (mobile) | Correct — identical content and order | None (`scrollWidth` 375 = viewport) | 239px |

Sample of one rendered card (extracted `get_page_text`, both widths identical in
content):
```
Arsenal v Coventry
MW1 · pick Arsenal
Projected Score If Pick Lands: 2-0
Top Exact Scores: 2-0 14% · 3-0 12% · 1-0 11%
The highest-probability exact score can differ from the most likely match result.
2-0
14% TOP SCORE
77%
```

**Honest tradeoff, not hidden**: row height roughly 1.5× taller than today at both
widths (desktop 162px vs. an unmeasured but visibly shorter current row; mobile
239px). Ten fixtures now means noticeably more scrolling than before. This is a
direct consequence of fitting four data points instead of two into each card — if
that height cost isn't acceptable, the alternative is moving items 3–4 behind a
per-card expand/collapse instead of always-visible, which would need a different,
larger diff. Flagged for a decision, not resolved unilaterally here.

## Status

Prepared for review. Not applied to `index.html`. `git status` on the real repo shows
zero changes from this exercise. Will not be deployed without explicit approval, and is
independent of the market-blend research track.
