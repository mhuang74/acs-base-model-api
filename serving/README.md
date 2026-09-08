# Serving Helpers

Root `modal_app.py` is intentionally still the canonical Modal serving
entrypoint. Railway packages that file into the wrapper container and the admin
Deploy button runs `modal deploy modal_app.py` from there.

This folder holds secondary Modal/operator scripts that are not part of the
normal live inference deploy path:

- `capacity_check.py` - manual 8-GPU availability probe.
- `capacity_scheduled.py` - HTTP-triggered capacity probe used by the wrapper.
- `prestage_trinity_weights.py` - one-time Trinity Volume prestage helper.
- `experiments/clustered_smoke.py` - multi-node `@modal.experimental.clustered` smoke test.
- `modal_config.py` - static model registry imported by root `modal_app.py`.
