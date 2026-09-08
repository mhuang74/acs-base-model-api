# Anonymize identity: scrub `<owner>` / `<owner>` from repo files

## Context

User wants their identity removed from all files in `/Users/<owner>/Projects/interviews/acsinfra/acs-base-model-api`, except hidden directories (`.git`, `.venv`, and other dot-dirs), per explicit instruction. User separately approved deleting cached files.

Exhaustive byte-level scan this session (516 files, case-insensitive, hidden dirs skipped) found the string in exactly two places:

1. **9 occurrences across 6 markdown files** — all are the GitHub owner component in slugs/URLs (`<owner>`, `<owner>-learning-playground`). No source code, config, lockfile, env file, or test file contains it. No other `<owner>` variants; nothing in filenames.
2. **179 `__pycache__/*.pyc` compiled caches across 10 directories** — each embeds absolute source paths `/Users/<owner>/...`. Regenerable artifacts, gitignored.

Replacement chosen by user: **`<owner>` placeholder** (longest token first):
- `<owner>-learning-playground` → `<owner>-learning-playground`
- `<owner>` → `<owner>`

Angle brackets are kept verbatim as shown to the user (e.g. `https://github.com/<owner>/acs-base-model-api/issues/10`), including in bare markdown URLs.

## Approach

### Step 1 — Scrub the 6 markdown files (9 occurrences on 7 lines)

Apply the replacement rule to these exact locations (re-read each file before editing; the edit tool requires a fresh snapshot tag):

| File | Line | Occurrences |
|---|---|---|
| `AGENTS.md` | 147 | 1 (`<owner>-learning-playground/acs-base-model-api` in backticks) |
| `SOLUTION-NOTES.md` | 3 | 1 (`https://github.com/<owner>/acs-base-model-api`) |
| `docs/agents/issue-tracker.md` | 3 | 1 (`<owner>-learning-playground/acs-base-model-api`) |
| `docs/agents/issue-tracker.md` | 14 | 1 (`origin` → `<owner>-learning-playground/acs-base-model-api`) |
| `specs/handover-issue-12-workbench-run-persistence.md` | 4 | 1 (`<owner>-learning-playground/acs-base-model-api`) |
| `specs/monthly_token_report.md` | 3 | 2 (`https://github.com/<owner>/acs-base-model-api/issues/10` and `/issues/2` in link targets) |
| `specs/workbench_import_export.md` | 3 | 2 (`[<owner>/acs-base-model-api#12]` text + link target `https://github.com/<owner>/acs-base-model-api/issues/12`) |

Only the token changes; the rest of each line is preserved byte-for-byte (em-dashes, arrows, backticks included). Use the `edit` tool, one hunk per changed line (`PUT N.=N:` with the full replacement line). These are doc-only changes; nothing references these slugs programmatically (`docs/agents/issue-tracker.md:14` itself says the repo is inferred from `git remote -v`).

### Step 2 — Delete the 10 `__pycache__` directories (user-approved)

From repo root:

```bash
find . \( -name '.*' -prune \) -o -type d -name __pycache__ -print0 | xargs -0 rm -rf
```

The prune protects `.git`/`.venv`/`.pytest_cache`/`.ruff_cache`. Expected targets (verified this session via dry-run): `./serving`, `./base_model_wrapper/tests`, `./base_model_wrapper/cli`, `./base_model_wrapper/alembic/versions`, `./base_model_wrapper/alembic`, `./base_model_wrapper/src/wrapper`, `./base_model_wrapper/src/wrapper/routes/admin`, `./base_model_wrapper/src/wrapper/routes`, `./base_model_wrapper/src/wrapper/services`, `./acs_model_registry` (each +`/__pycache__`).

They regenerate on the next `pytest`/`python` run — expected, not a failure.

## Verification

1. **Byte-level rescan** (eval, Python, skipping dot-dirs — same method as discovery): expect zero files containing `<owner>` (case-insensitive) outside hidden dirs, zero `__pycache__` dirs remaining, no filename containing `<owner>`.
2. **Built-in `grep`**, case-insensitive, `gitignore: false`, over repo root: remaining matches must be only under hidden dirs (`.venv/`).
3. **Targeted confirmation**: grep `<owner>` over the 6 files → exactly 9 occurrences; spot-read one rewritten line per file matches the before/after mapping above.
4. **Git sanity** (read-only): `git diff --stat` → exactly the 6 markdown files modified; `git status --porcelain` → only those 6 `M` entries (pycache deletions are gitignored-invisible).

## Assumptions & contingencies

- **`.git` is untouched per user instruction.** Committed history still contains the string in blobs (e.g. previously committed versions of these files), and `.git/config` holds the real remote URL. Sharing the repo as-is would leak identity via history; history rewrite/re-init is out of scope unless the user asks.
- **`.venv` is a hidden dir → excluded per instruction.** It embeds `/Users/<owner>/...` paths in venv scripts/`pyvenv.cfg`/editable-install finder, but it regenerates via `uv sync` and is never part of a shared repo.
- If an edit-tool snapshot tag is stale → re-read the file and re-apply the same replacement; do not guess.
- If the `find` invocation misbehaves on macOS → delete the 10 enumerated directories directly via eval `shutil.rmtree` and re-run the rescan.
