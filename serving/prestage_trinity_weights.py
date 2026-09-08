"""
Prestage Trinity-Large-TrueBase weights into a Modal Volume.

Downloads the 398B/13B-active MoE BF16-native weights (~797 GB, 31 shards)
for ``arcee-ai/Trinity-Large-TrueBase`` into the ``acs-trinity-cache`` Volume
on a cheap CPU container, so the 8xH200 serving Function can later mount the
Volume and load from disk instead of burning ~$36/hr on idle GPUs while HF
downloads stream in over the network.

One-time job. Subsequent serving cold boots read the weights directly from
the Volume — no re-download needed.

Cost: ~$1, ~60 min wall time (CPU-only container at ~$0.20/hr).

Usage:
    modal run serving/prestage_trinity_weights.py::main
"""

import os
import subprocess

import modal

APP_NAME = "acs-trinity-prestage"
VOLUME_NAME = "acs-trinity-cache"

HF_REPO = "arcee-ai/Trinity-Large-TrueBase"
CACHE_DIR = "/cache"
LOCAL_DIR = f"{CACHE_DIR}/trinity-base"

MINUTES = 60

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=2,
    memory=16384,
    volumes={CACHE_DIR: vol},
    timeout=2 * 60 * MINUTES,
)
def prestage():
    """Download Trinity-Large-TrueBase into the Volume. Idempotent (snapshot_download skips existing files)."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from huggingface_hub import snapshot_download

    print(f"[prestage] downloading {HF_REPO} -> {LOCAL_DIR}", flush=True)
    path = snapshot_download(
        repo_id=HF_REPO,
        local_dir=LOCAL_DIR,
        max_workers=8,
    )
    print(f"[prestage] download complete: {path}", flush=True)

    vol.commit()
    print(f"[prestage] volume {VOLUME_NAME!r} committed", flush=True)

    du = subprocess.run(
        ["du", "-sh", LOCAL_DIR],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"[prestage] final size: {du.stdout.strip()}", flush=True)


@app.local_entrypoint()
def main():
    prestage.remote()
    print()
    print(f"Done. Weights live in Modal Volume {VOLUME_NAME!r} at {LOCAL_DIR}.")
    print("Next steps:")
    print(f"  - Mount the {VOLUME_NAME!r} Volume at {CACHE_DIR} in the 8xH200 serve Function")
    print(f"  - Point vLLM at {LOCAL_DIR} instead of the HF repo id")
