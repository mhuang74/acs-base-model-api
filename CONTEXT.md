# ACS Base-Model API

OpenAI-compatible API wrapper in front of Modal/vLLM base-model backends, with key/budget management, usage accounting, and a server-rendered web UI.

## Language

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