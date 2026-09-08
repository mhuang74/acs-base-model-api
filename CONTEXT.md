# ACS Base-Model API

OpenAI-compatible API wrapper in front of a Modal/vLLM base-model endpoint, with auth, key/budget management, usage accounting, and a server-rendered web UI (Workbench, Loom, Compare).

## Language

### Workbench sessions

**Session prompt** (`prompt_text`):
The rolling text buffer carried by a Workbench chat session: the last submitted prompt, plus the last completion when the session continues from it. What the single-pane composer renders. Not a pure prompt — treat as prompt-or-continuation-context depending on the last action.
_Avoid_: "the prompt" (ambiguous), prompt history

**Prompt baseline** (`prompt_before`):
The text a generation ran on, before its completion was appended. The authoritative prompt/completion boundary; only the server knows it.
_Avoid_: original prompt

**Roll-forward**:
The act of appending a completion to the session prompt (server-side after a completed single-pane run, client-side while streaming).
_Avoid_: chaining, chaining carryover

**Continue**:
Running a new generation on the roll-forward. The only intended path where a completion joins the next prompt.

### Workbench modes

**Single pane**:
The Workbench mode holding the session prompt textarea and one output box.

**Compare**:
A Workbench mode that fans one shared prompt out to two or more model lanes in one batch. Inheriting anything beyond the prompt baseline from a single-pane session is a defect, not a feature.
_Avoid_: A/B mode, dual lane

**Lane**:
One model's generation inside a Compare batch; lanes share a compare run identity and are isolated from each other's failures.

### Snapshots

**Snapshot** (`ChatSnapshot`):
A saved prompt-before/completion pair from a single-pane run, used for restore and the heatmap.

**Restore**:
Putting a snapshot's prompt baseline back into the session prompt. Restore means the baseline, not the roll-forward.

### Usage accounting

**Usage month**:
A UTC calendar month; the window over which usage is totaled, budgeted, and reported.
_Avoid_: billing cycle, statement period

**Usage rollup**:
Per-key token and request totals maintained as usage is committed, at monthly and daily grain.
_Avoid_: usage counters, usage tables

**Request log**:
The persisted metadata record of one API request (model, token counts, status) — never prompt or completion text.
_Avoid_: audit log, request history

**Usage commit**:
The act of adding one request's token counts to the rollups.
_Avoid_: usage write

### Reporting

**Usage report**:
A downloadable monthly token report for one user, broken out per key and model.
_Avoid_: export, statement
