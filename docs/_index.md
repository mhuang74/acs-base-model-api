---
title: Docs map
status: current
updated: 2026-06-23
owner: platform@example.org
---

# `docs/` — what lives where

One git-tracked tree for everything that isn't co-located code (module READMEs stay
next to their code). Each subfolder has its own `_index.md`. The taxonomy:

| Path | Role | Update style |
|------|------|--------------|
| [`architecture/`](architecture/_index.md) | **How it works now** — canonical system reference. | Living — overwrite in place. |
| [`runbooks/`](runbooks/_index.md) | **How to operate it** — smoke / triage / stress procedures. | Living — overwrite in place. |

## Where do I put a new doc?

- Describes how the system *currently* works → `architecture/`.
- A procedure someone runs (smoke, triage, deploy) → `runbooks/`.

## Related, but not here

- `../README.md`, `../serving/README.md`, `../base_model_wrapper/README.md` — module quickstarts, co-located with code.
- `../base_model_wrapper/src/wrapper/docs/` — the live in-app **tutorial** content (served at `/tutorial`, tested by the wrapper suite). Not part of this tree.
