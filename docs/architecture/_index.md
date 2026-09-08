---
title: Architecture
status: current
updated: 2026-07-17
owner: platform@example.org
---

# architecture/ — how the system works now

Living reference. Overwrite in place; keep `updated:` frontmatter current.

- [`modal-app.md`](modal-app.md) — end-to-end walkthrough of how a `/v1/completions` request is served on Modal + vLLM (every layer from laptop to generated continuation).
- [`modal-phases.md`](modal-phases.md) — why the deploy is split into `stage_weights` → `preshard` → `serve`, plus a status check against current Modal/vLLM docs.
- [`cost-monitoring.md`](cost-monitoring.md) — GPU cost sampler + spike alert, and how the wrapper reads live container counts from Modal (the internal-client requirement + the `gpu_shape_label` "×" gotcha).
- [`activation-harvesting.md`](activation-harvesting.md) — the two activation-capture paths (inline probing/steering vs SAE-scale bulk harvest), the HF-forward-loop pipeline (batching, storage/upload overlap, Activault shard layout), the self-serve `/v1/harvest` API, and the measured bottleneck profile (upload 23.8 MB/s).
