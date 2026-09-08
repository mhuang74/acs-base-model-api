# ACS Base Model Platform

An OpenAI-compatible API wrapper in front of a Modal/vLLM base-model endpoint, plus auth, key/budget management, usage accounting, and a server-rendered web UI. This glossary records the canonical domain language.

## Language

### Workbench & reproduction

**Session**:
A user-owned, server-side container of Runs with a stable URL, holding the working prompt and the Runs performed against it.
_Avoid_: chat, conversation, workspace

**Run**:
One completion request and its recorded outcome: the sampling settings used, the prompt prefix, the completion, the model identity, and the timestamps.
_Avoid_: snapshot, generation, completion (record)

**Sampling settings**:
The parameter set chosen for a Run: temperature, top_p, top_k, min_p, presence/frequency/repetition penalties, seed, stop sequences, and the logprobs toggle.
_Avoid_: knobs, hyperparameters, recipe, config

**Draft**:
The URL-encoded sampling settings carried in the workbench URL; authoritative for the next Run while present.
_Avoid_: pending settings, staged params

**Compare lane**:
A Run produced as one of several alternatives within a single compare action.
_Avoid_: branch, variant

**Compare snapshot**:
The grouping of lanes produced by one compare action.
_Avoid_: diff, matrix

**Reproducible Run**:
A Run recorded with an effective seed, or with temperature 0. Only Reproducible Runs back the reproduction promise.
_Avoid_: deterministic run

**Reproduction**:
Re-running a Run's sampling settings against the upstream to obtain the same outcome; guaranteed only for temperature 0, best-effort otherwise (upstream deployments may differ).
_Avoid_: replay, rerun

**Export document**:
The versioned JSON file describing one Session and its Runs; the unit of interchange between users.
_Avoid_: backup, dump, transcript file

**Import**:
Bringing an Export document in as a new Session owned by the importing user, with sampling settings normalized to this deployment's parameter ranges.
_Avoid_: restore, merge

**Restore chain**:
The order of authority for composer settings on page load: Draft, then the Session's most recent successful Run, then built-in defaults.
_Avoid_: fallback logic, hydration

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
