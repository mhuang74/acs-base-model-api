# Spec: Compare lanes must run on the prompt baseline, not the session prompt's roll-forward

Resolves #5.

## Problem Statement

I generated a completion in the single-pane Workbench, then switched to Compare to try the same prompt on two models. The lanes ran on my prompt **plus the completion I had just generated**, not on my prompt. I had to notice the contamination in the output and manually clean the shared prompt before Compare was usable.

## Solution

Compare always starts from the **prompt baseline** — the text as it stood *before* my last single-pane completion — no matter how I arrive at Compare: flipping the mode toggle, or opening/refreshing a chat directly in Compare mode. My completion is never silently glued onto the comparison input. If I edited my prompt after the run, or typed fresh text without running, that text carries over whole. Switching back to Single never discards my continuation text.

## User Stories
1. As a Workbench user, I want the Compare shared prompt to contain only my prompt after a single-pane completion, so that the lanes compare what I actually asked for.
2. As a Workbench user who just ran a completion, I want the Compare prompt prefilled with the text as it stood before the run (the prompt baseline), so that I can immediately compare models on the same prompt without manual cleanup.
3. As a Workbench user, I want the prompt boundary to come from the server's own records (the snapshot written with each roll-forward), so that the prompt/completion split is exact rather than guessed in the browser.
4. As a Workbench user, I want the Compare prefill to be correct when I arrive via the mode toggle, so that the most common path is fixed.
5. As a Workbench user, I want the Compare prefill to be correct when I open or refresh a chat directly in Compare mode (link, bookmark, reload), so that no entry path leaks the completion.
6. As a Workbench user with a brand-new chat (nothing generated yet), I want Compare prefilled with the current session prompt, so that nothing is lost when there is no baseline to recover.
7. As a Workbench user who edited the prompt after a completion, I want my edited text carried into Compare as-is, so that my edits are never mistaken for a completion tail.
8. As a Workbench user who typed fresh text and switched to Compare without running, I want that text carried whole, so that I can compare on it immediately.
9. As a Workbench user who switches from Compare back to Single, I want my single-pane continuation text (prompt + completion) still present, so that mode-switching never silently discards the completion.
10. As a Workbench user streaming a completion in Single, I want the mode toggle to keep skipping the prompt sync mid-stream, so that partial completions never contaminate Compare.
11. As a Compare user, I want each lane to run on exactly the text shown in the shared prompt, so that what I see is what every model received.
12. As a Compare user whose run finishes, I want the session prompt to become the submitted Compare prompt, so that Single and Compare stay coherent (existing intended behavior, kept).
13. As a Compare user whose last single-pane run was cancelled mid-completion, I want the same baseline semantics, so that partial completions do not leak either.
14. As a Compare user restoring a snapshot, I want restore to keep putting the snapshot's prompt baseline into the session, so that restore semantics remain predictable and unaffected by this fix.
15. As a developer reviewing this fix, I want a regression test that fails whenever the Compare page renders prompt + completion, so that this class of bug cannot silently return.
16. As a Workbench user who restored an older snapshot and then switched to Compare, I want Compare prefilled with the restored baseline, so that my restore point is not silently replaced by the latest generation's prompt.
17. As a Workbench user whose last run failed before producing a completion, I want Compare prefilled with my session prompt as-is, so that a failed run never hides my baseline.
18. As a Workbench user, I want the Compare prefill rule to treat any state that isn't exactly an unmodified roll-forward as intentional, so that edits, restores, and re-submissions are never second-guessed.

## Implementation Decisions

- **Vocabulary**: this spec uses the project glossary (CONTEXT.md): *session prompt* (the rolling text buffer a single-pane chat carries), *prompt baseline* (the text a generation ran on, before its completion was appended — the authoritative prompt/completion boundary, known only server-side), *roll-forward* (appending a completion to the session prompt), *Compare*, *lane*.
- **Fix shape: targeted.** The session prompt keeps its rolling-buffer role for the single-pane Continue workflow (an existing, tested behavior). No schema changes: the prompt/completion boundary is recovered from state already stored per chat — the latest *roll-forward snapshot* (written in the same transaction as the roll-forward), not the latest generation (a failed or cancelled-after-restore generation can lack/hold the wrong boundary).
- **Compare-mode page render**: the shared prompt renders the prompt baseline when — and only when — the session prompt exactly equals the latest roll-forward snapshot's baseline + completion (the unmodified roll-forward); otherwise it renders the session prompt as-is. This covers every clean state for free: snapshot restore (session prompt is an older baseline), post-run edits, fresh typing, and a Compare run's write-back all fail the equality check and render untouched. When the chat has no roll-forward snapshot, it renders the session prompt unchanged. The render context additionally exposes the baseline and completion as separate values so client script can recognize the unmodified roll-forward.
- **Mode toggle, forward (Single → Compare)**: when the single textarea exactly equals baseline + last completion (the unmodified roll-forward), the copy into Compare substitutes the baseline; any other content (post-completion edits, freshly typed text) copies as-is. The existing mid-stream skip is preserved.
- **Mode toggle, reverse (Compare → Single)**: the copy back skips when the single textarea still holds the unmodified roll-forward, so returning from Compare never discards the completion tail; when the single pane was edited since the run, the compare text copies back as before.
- **Compare submission**: unchanged — the submitted shared prompt goes to every lane verbatim, and the submitted prompt is written back into the session prompt (existing intended behavior, codified by an existing invariant test). Post-fix this write-back lands clean baselines.
- **Accepted residual**: text appended to the rolled-forward single textarea *without running*, then switching to Compare, carries whole (baseline + completion + suffix). Appending to a roll-forward is the Continue workflow; running it refreshes the baseline. Documented corner, deliberately not guarded.
- **Continuation-compare is explicitly not a feature**: comparing how two models continue a text must, if ever wanted, be an explicit control — never a side effect of carrying state between modes.

## Testing Decisions

- Good tests assert external behavior only: what the rendered Compare page shows in the shared prompt, and what persisted state a Compare run leaves — never internal helpers or template internals.
- **One seam**: the DB-gated e2e HTTP render of the Compare-mode page. Cases: (a) after a completed single-pane run, the rendered shared prompt equals the prompt baseline, not baseline + completion — this fails pre-fix; (b) a chat with no generations renders the session prompt unchanged; (c) after a cancelled-with-partial run, the same baseline semantics hold; (d) after restoring an older snapshot, the restored baseline renders as-is — not the latest generation's baseline, so a restore point is never discarded by a switch to Compare.
- Existing submission-side coverage (lane fan-out, per-lane error isolation, the submitted-prompt invariant, snapshots) already guards the server side and is untouched.
- **Prior art**: the existing Compare generation tests — per-file database-URL preamble with its skip marker, a lazily-built test client fixture, fresh uuid rows per test, and monkeypatched model-spawn seams. Whole files are run, never single node ids.
- The toggle guards are browser script; the repo has no JS test infrastructure, so they stay manually verified (accepted gap). The server embeds everything the guards need, which is what the seam above asserts.

## Out of Scope

- The root refactor: stopping the roll-forward entirely so the session prompt never contains a completion, making Continue an explicit composition action. Cleaner long-term model (it dissolves the accepted residual and an existing streaming workaround), but a feature-level change to the single-pane workflow with its own test updates.
- Sweeping Loom and snapshot-restore *for new fixes* for the same carry-over class: eyeballed during implementation; findings become follow-up issues, not part of this change. (Snapshot restore itself is load-bearing for the render rule above — restore leaves a clean state the equality check must not "fix".)
- A continuation-compare feature (deliberately running lanes on prompt + completion).
- JavaScript test infrastructure.
- Changes to the Compare submission contract or validation.

## Further Notes

- Root cause: the session prompt is a rolling continuation buffer by design (the Continue workflow depends on it), and both panes render from it; the exact prompt/completion boundary exists only server-side on the last generation record. The fix reuses the boundary-recovery pattern the single-pane auto-resume path already uses, instead of inventing a new one.
- Related tickets: ACS-163 (unified Workbench), ACS-254 (Compare persists the submitted prompt — behavior kept), ACS-186 (server-side Compare run identity).
- The equality-check render rule came out of design review: anchoring on the latest generation's baseline instead would break restore-then-Compare (restore writes an older baseline into the session prompt while the latest generation's baseline stays newer) and failed/cancelled runs that never wrote a roll-forward snapshot.
- A project glossary (CONTEXT.md) was created during design; the session-prompt / prompt-baseline split it records is the conceptual heart of this fix.