# SOLUTION NOTES

## Issue #11 — Compare lanes must run on the prompt baseline, not the session prompt's roll-forward

Implemented per the spec (resolves #5). Glossary terms follow `CONTEXT.md`:
*session prompt* (rolling buffer), *prompt baseline* (`prompt_before` — the
server-known prompt/completion boundary), *roll-forward*, *Compare*, *lane*.

### Root cause

The session prompt is a rolling continuation buffer by design: after a
single-pane run the server appends the completion
(`workbench_generations._run_generation_task`:
`chat.prompt_text = prompt + state.completion_text`, in the same transaction as
the `ChatSnapshot` write). Both panes rendered that one buffer — the Compare
shared prompt included — so a single-pane run silently glued its completion
onto the comparison input, whether you arrived via the mode toggle or by
opening `?mode=compare` directly.

### Fix shape (targeted, per spec — no schema changes)

1. **Server render rule** (`routes/workbench.py::_render_chat`, the one tested
   seam). The compare textarea renders `compare_prompt`, computed as: the
   latest roll-forward snapshot's `prompt_before` **when — and only when** —
   the session prompt exactly equals that snapshot's `prompt_before +
   completion_text` (the unmodified roll-forward); otherwise the session
   prompt as-is; a chat with no roll-forward snapshot renders unchanged. The
   snapshot list is already loaded in `chat_open` (`_session_snapshots`,
   newest first), so no extra query. Anchor deliberately on the latest
   *snapshot*, not the latest generation: failed runs write neither snapshot
   nor roll-forward, and cancelled-with-partial runs write both — so the
   snapshot is the only state that always holds the exact boundary. Restores
   write an older baseline into the session prompt, which fails the equality
   check and renders as-is (story 16); a Compare run's write-back
   (`compare_start` sets `chat.prompt_text = submitted prompt`) likewise fails
   the check and renders untouched (story 12 kept verbatim).
2. **Boundary exposure for the toggle guard** (`workbench_compare.html`): the
   rendered page embeds `<script id="cmp-roll-forward">` with
   `{baseline, completion}` (null when not an unmodified roll-forward). The
   server is the only place that knows the split exactly (story 3).
3. **Mode toggle, forward** (Single → Compare, `workbench.html`): when the
   single textarea is *exactly* `baseline + completion`, the copy substitutes
   the baseline; any other content (post-run edits, fresh typing) copies
   as-is. The existing mid-stream skip is preserved (story 10).
4. **Mode toggle, reverse** (Compare → Single): the copy back skips while the
   single textarea still holds the unmodified roll-forward, so returning never
   discards the completion tail (story 9); once the single pane is edited, the
   compare text copies back as before. The skip also advances `lastSynced` to
   the compare value so a later reverse sync can't resurrect a stale compare
   edit over newer single-pane content.
5. **Freshness without reload**: after a completed single-pane run the client
   does NOT reload (ACS-187), so the pre-run embedded boundary would be stale.
   `refreshSnapshots()` — already fired on the `done` frame, already fetching
   the full page HTML — now also swaps the fresh `#cmp-roll-forward` script in
   place. The toggle parses it fresh on every sync (no cached state, no
   cross-IIFE event wiring). The commit that writes the new snapshot happens
   before the done frame broadcasts, so the refreshed fetch always sees the
   committed boundary.
6. **Compare submission unchanged**: `compare_start` still writes the submitted
   prompt back into the session prompt (ACS-254 invariant, kept by an existing
   test). Post-fix the write-back lands clean baselines because the compare
   textarea itself prefills with the baseline.

### Tests

`tests/test_workbench_compare_render.py` — the spec's one seam (DB-gated e2e
HTTP render of the Compare-mode page), per-file preamble copied from
`test_workbench_generations.py` prior art; whole file is run, never node ids:

- completed run → shared prompt == baseline, single pane still == roll-forward
  (story 9's server side), embedded boundary `{baseline, completion}` present —
  **fails pre-fix** (regression for story 15);
- no generations → session prompt unchanged, boundary null (stories 2/6);
- cancelled-with-partial → snapshot carries the exact partial boundary,
  session prompt == baseline + partial, compare renders baseline (story 13);
- restore-older-snapshot → restored baseline renders as-is, NOT the latest
  generation's baseline — this pins the design choice of the equality check
  over latest-generation anchoring (story 16); boundary null;
- post-run edit → edited text renders verbatim, boundary null (story 18).

Toggle guards are browser script with no JS test infrastructure in this repo —
manually verified per the spec's accepted gap (the server embeds everything
they need, which is what the render seam asserts).

### Known corners (deliberate, per spec)

- **Append to a roll-forward without running, then toggle**: carried whole
  (baseline + completion + suffix). Appending is the Continue workflow; running
  refreshes the boundary. Documented accepted residual, not guarded.
- **Byte-identical re-creation of the roll-forward** (e.g. manually retyping
  `baseline + completion`, or a Compare run submitted with exactly that text):
  the equality check can't distinguish provenance, so it treats it as the
  unmodified roll-forward and renders the baseline. Story 18's rule is
  explicitly text-based ("isn't *exactly* an unmodified roll-forward"); the
  server cannot know history from text alone, and continuation-compare is
  explicitly not a feature.
- **Cancelled-with-partial then toggle without reload**: the optimistic Stop
  path aborts the SSE reader (no `done` frame → no `refreshSnapshots()`), so
  the in-page boundary can stay stale until the next refresh/reload; the
  render seam is correct the moment the page is re-rendered. Same fail-soft
  class as the accepted residual above.

### Out of scope (per spec)

The root refactor (stop rolling forward entirely), Loom/snapshot-restore sweeps
for new fixes, continuation-compare as a feature, JS test infrastructure, and
changes to the Compare submission contract. Nothing else in the Loom/snapshot
surfaces shares this exact carry-over seam (they don't render
`chat.prompt_text`), so no follow-up issues were generated from the eyeball
pass.
