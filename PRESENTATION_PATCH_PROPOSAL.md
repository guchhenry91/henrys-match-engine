# Presentation-only patch — proposed, NOT applied

Separate from the model patch (`CORRECT_SCORE_AUDIT_REPORT.md` §8). This changes
**display only** — no probability, lambda, rho, or grid value changes. Nothing here
touches `leagues/model.py`, `leagues/publish.py`, or any `data/leagues/*.json` field;
it only changes how `index.html` renders fields that already exist in those files.

## What this does NOT do

**This does not manipulate or suppress 1-1, or any other score, mathematically.** The
grid, the rho correction, and the argmax are untouched. This is a labeling change only:
making explicit, in the copy itself, that "Most Likely Raw Score" means *the single
largest cell in the exact-score probability grid* — a statement about which specific
scoreline has the highest individual probability among many similarly-sized ones — and
is **not** a claim that a draw is the most likely 1X2 outcome. Those are answering
different questions (§1 of the main report covers why they can legitimately disagree),
and the copy needs to say that outright rather than let a reader infer it.

## The four sections, as specified

1. **Match Pick** — the highest of the three 1X2 probabilities, with its percentage.
2. **Most Likely Raw Score** — the unrestricted grid mode (`top_scores[0]`), clearly
   labeled as a raw individual-cell probability, with explanatory microcopy.
3. **Projected Score If Pick Wins** — the score conditional on the Match Pick
   (`score_for_outcome`), always shown, not hidden when it happens to agree with #2.
4. **Other Plausible Scores** — the next-ranked alternatives (already exists as "Most
   likely scorelines," cosmetic rename only if desired).

## Current state (`index.html`, `openMatch()`, lines 497–534)

Already implements about 80% of this, post the earlier `09e48fc` fix in this session:
the true grid mode is the headline (labeled "best guess"), the pick-conditional score
shows as a note only when it *differs* from the grid mode, and a top-3 spread exists.
Three gaps close the rest of the way:

1. No explicit "Match Pick: Team (P%)" line in the detail view — only the H/D/A split
   bar, no text statement of the pick itself.
2. "Projected Score If Pick Wins" is conditional today (`condDiffers`) — hidden when it
   agrees with the raw mode. Spec wants it always shown.
3. Headline label "best guess" → "Most Likely Raw Score," with explanatory microcopy
   making the individual-cell-vs-1X2-outcome distinction explicit.

## Proposed diff (reference only — not applied to `index.html`)

```diff
@@ index.html openMatch(), inside the .dhead block @@
     <div class="glass dhead"><div class="lg">${esc(LG[lk])} · MW${m.matchweek}</div><div class="vs">${esc(m.home)} v ${esc(m.away)}</div>
+      <div style="font-size:12px;font-weight:700;color:var(--sub)">Match Pick</div>
+      <div style="font-size:15px;font-weight:800">${esc(p.pick)} <span style="color:var(--sub);font-weight:600">(${pc(p.p_pick)}%)</span></div>
-      <div class="score"><b>${esc(sc[0])}</b><span>best guess${bestPct!=null?" · "+bestPct+"%":""}</span><b>${esc(sc[1]||"")}</b></div>
+      <div class="score"><b>${esc(sc[0])}</b><span>Most Likely Raw Score${bestPct!=null?" · "+bestPct+"%":""}</span><b>${esc(sc[1]||"")}</b></div>
       <div class="split"><div class="sh" style="flex:${ph||1}">${ph}%</div><div class="sd" style="flex:${pd||1}">${pd}%</div><div class="sa" style="flex:${pa||1}">${pa}%</div></div>
       <div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:8px">${esc(m.home)} win · draw · ${esc(m.away)} win</div>
-      ${condDiffers?`<div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:4px">If ${esc(p.pick)} win${p.pick===m.home||p.pick===m.away?"s":""}, most likely: ${esc(p.score)}</div>`:""}</div>
+      <div style="font-size:11px;color:var(--sub);font-weight:650;margin-top:4px">Projected Score If Pick Wins: ${esc(p.score)}${condDiffers?" (differs from the raw score above)":""}</div>
+      <div style="font-size:10.5px;color:var(--sub);font-weight:550;margin-top:6px;line-height:1.4">"Most Likely Raw Score" is the single largest cell in the full scoreline grid -- it is not a claim that a draw is the most likely match result. The Match Pick above answers who's more likely to win; the raw score answers which exact tally is individually most probable, and the two can honestly disagree.</div></div>
```

@@ existing "Most likely scorelines" section header, optional cosmetic rename only @@
```diff
-    <div class="glass dsec"><h3>Most likely scorelines</h3>${tops}
+    <div class="glass dsec"><h3>Other Plausible Scores</h3>${tops}
```

## Status

Prepared for review. Not applied to `index.html`. Will not be deployed without explicit
approval, and is independent of the market-blend model patch — this can be approved and
shipped on its own regardless of what happens with the market-blend research track.
