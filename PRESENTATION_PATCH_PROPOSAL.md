# Presentation-only patch — proposed, NOT applied (v3, final scope)

**Target confirmed**: `C:\Users\John\worldcup`, remote `henrys-match-engine.git`, branch
`main`, HEAD `d8c541904c24c47c300c968be07c066fc1d95108`. Codex was examining a different
repository entirely (`worldcup.git`, branch `codex/fix-audit-and-rosters`) — resolved,
not relevant to this repo.

**v2 (expanded `viewLeague()` rows) was rejected**: 162px desktop / 239px mobile row
height created too much scrolling. This version keeps `viewLeague()` rows compact and
moves all the new content into `openMatch()` (the match-detail view) instead, per
explicit instruction.

## What this does NOT do

No probability, lambda, rho, or grid value changes — display only. Does not manipulate
or suppress 1-1 or any other score.

## Scope

**`viewLeague()`** (fixture list) — one-word label change only. No new elements, no new
CSS, no height change (confirmed: 70px per row, identical before/after).

**`openMatch()`** (match detail) — Match Pick line added, "best guess" renamed to
"Highest-Probability Exact Score," the pick-conditional score always shown (previously
hidden when it agreed with the raw mode) under the fixed label "Projected Score If Pick
Lands" (no team-name grammar problem for draws), the top-3 section renamed to "Top
Exact Scores," one explanation sentence added. Unused `condDiffers` variable removed
(dead code once its only use is gone).

## Exact diff (verified by rendering, not applied to `index.html`)

```diff
--- index.html
+++ index.html (proposed)
@@ -488,7 +488,7 @@
     const best=(p.top_scores||[])[0];
     return `<div class="rw" onclick="openMatch('${k}',${m.id})"><span class="lgtag" style="background:${LC[k]}"></span>
       <div class="info"><b>${esc(m.home)} v ${esc(m.away)}</b><span>MW${m.matchweek} · pick ${esc(p.pick)}</span></div>
-      <span class="psc">${esc(best?best.score:"–")}<small>${best?Math.round(best.pct)+"% guess":"score"}</small></span>
+      <span class="psc">${esc(best?best.score:"–")}<small>${best?Math.round(best.pct)+"% top score":"score"}</small></span>
       <div class="conf" style="min-width:48px"><b>${pc(p.p_pick)}%</b></div><span class="chev">›</span></div>`;}).join("")}</div>
     <p class="note">Best single-score guess and its own chance, plus the win %. No single score is likely -- see each match for the full spread. Full standings are in the Tables tab.</p>`;
   return h;
@@ -517,17 +517,19 @@
   const props=(m.props||[]).slice(0,5).map(pr=>`<div class="leg"><div class="av" style="width:36px;height:36px;font-size:13px;background:${tc(pr.team)}">${esc(initials(pr.player))}</div>
     <div class="li2"><b>${esc(pr.player)}</b><span>${pr.anytime_pct}% anytime · ${num(pr.exp_shots)} attempts · ${num(pr.exp_sot)} on tgt</span></div></div>`).join("");
   const ua=p.unmodeled_absences||{}; const uaAll=[...(ua.home||[]),...(ua.away||[])];
-  const condDiffers = p.score && best && p.score !== best.score;
   const app=document.getElementById("app");
   app.innerHTML=`<button class="back" type="button" onclick="render(cur)">‹ Back</button>
     <div class="glass dhead"><div class="lg">${esc(LG[lk])} · MW${m.matchweek}</div><div class="vs">${esc(m.home)} v ${esc(m.away)}</div>
-      <div class="score"><b>${esc(sc[0])}</b><span>best guess${bestPct!=null?" · "+bestPct+"%":""}</span><b>${esc(sc[1]||"")}</b></div>
+      <div style="font-size:12px;font-weight:700;color:var(--sub)">Match Pick</div>
+      <div style="font-size:15px;font-weight:800">${esc(p.pick||"–")} <span style="color:var(--sub);font-weight:600">(${pc(p.p_pick)}%)</span></div>
+      <div class="score"><b>${esc(sc[0])}</b><span>Highest-Probability Exact Score${bestPct!=null?" · "+bestPct+"%":""}</span><b>${esc(sc[1]||"")}</b></div>
       <div class="split"><div class="sh" style="flex:${ph||1}">${ph}%</div><div class="sd" style="flex:${pd||1}">${pd}%</div><div class="sa" style="flex:${pa||1}">${pa}%</div></div>
       <div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:8px">${esc(m.home)} win · draw · ${esc(m.away)} win</div>
-      ${condDiffers?`<div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:4px">If ${esc(p.pick)} win${p.pick===m.home||p.pick===m.away?"s":""}, most likely: ${esc(p.score)}</div>`:""}</div>
+      <div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:4px">Projected Score If Pick Lands: ${esc(p.score||"–")}</div>
+      <p style="font-size:10.5px;color:var(--sub);font-weight:550;margin-top:6px;line-height:1.4">The highest-probability exact score is one individual scoreline. It can differ from the most likely match result because each result contains many possible scores.</p></div>
     ${uaAll.length?`<div class="glass warnbox">Confirmed defensive/keeper absence not priced by the shot-based model: ${esc(uaAll.join(", "))}. Read the win probability with that in mind.</div>`:""}
-    <div class="glass dsec"><h3>Most likely scorelines</h3>${tops}
-      <p class="note" style="padding-top:8px">No single exact score is likely in football -- even the best guess above is usually right well under 1 time in 5. Treat this as the model's honest best shot, not a confident call.</p></div>
+    <div class="glass dsec"><h3>Top Exact Scores</h3>${tops}
+      <p class="note" style="padding-top:8px">No single exact score is likely in football -- even the highest-probability score above is usually right well under 1 time in 5. Treat this as the model's honest best shot, not a confident call.</p></div>
     <div class="glass dsec"><h3>Why this pick</h3>${reasons}</div>
     ${props?`<div class="glass dsec"><h3>Top players in this match</h3>${props}</div>`:""}`;
   window.scrollTo(0,0);
```

One extra, unrequested-but-consistency-preserving tweak included: the existing note
paragraph under "Top Exact Scores" said "even the best guess above" — updated to "even
the highest-probability score above" so it doesn't reference a label that no longer
exists on the page. Flagged explicitly, not silently folded in.

## Safety checklist

- `pc(p.p_pick)` used exactly as-is — no `pct()`, no pre-multiplication. `pc` already
  handles a missing/undefined `p_pick` safely (`Math.round((p||0)*100)` → `0`).
- `p.score` guarded: `esc(p.score||"–")`.
- `p.pick` guarded: `esc(p.pick||"–")`.
- `p.top_scores` already guarded pre-existing (`(p.top_scores||[])`) — untouched.
- Every dynamic value passed through `esc()`.
- No model, lambda, rho, or probability computation touched — grep confirms zero
  changes outside `viewLeague()`'s one label and `openMatch()`'s render block.

## Verification (rendered on a throwaway, untracked copy — `git status` confirms zero
changes to the real `index.html` before, during, or after)

| Check | Result |
|---|---|
| 1280×900, fixture list | 10 rows, 70px each (unchanged), label reads "N% top score" |
| 375×812, fixture list | 10 rows, 70px each, no horizontal overflow |
| 1280×900, differing fixture (Hull v Man Utd: `p.score`="0-2", `top_scores[0]`="1-1") | Match Pick: Manchester United (57%) · Highest-Probability Exact Score: 1-1 · 12% · Projected Score If Pick Lands: 0-2 — both numbers shown, correctly different |
| 375×812, same fixture | Identical content; "Highest-Probability Exact Score · 12%" label fits on **one line** (verified: span `scrollWidth` 304 = `clientWidth` 304, no internal overflow) — the wrapping risk flagged in the prior round did not materialize |
| Agreeing fixture (Arsenal v Coventry: both "2-0") | Both lines correctly show 2-0; explanation sentence still shown (not hidden just because they agree, per "always display") |
| Draw pick (none exists in current real data — synthetic fixture injected into in-memory `DATA` only, popped immediately after reading, never written to any file) | Match Pick: **Draw (31%)** · Projected Score If Pick Lands: 1-1 — reads correctly, no "If Draw wins" grammar problem |
| Body-level and `.score`-level overflow, both widths | None anywhere |

## Status

Prepared for review. Not applied to `index.html`. Repository confirmed as
`C:\Users\John\worldcup` @ `d8c541904c24c47c300c968be07c066fc1d95108` throughout this
verification. Waiting for approval before touching production.
