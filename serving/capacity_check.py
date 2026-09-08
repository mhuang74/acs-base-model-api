import time
import modal

from serving.capacity_probe import query_gpu_inventory

app = modal.App("capacity-check")

image = modal.Image.debian_slim()

@app.function(image=image, gpu="H200:8", timeout=600)
def probe_h200():
    return query_gpu_inventory()

@app.function(image=image, gpu="H100:8", timeout=600)
def probe_h100():
    return query_gpu_inventory()

@app.local_entrypoint()
def main(gpu: str = "H200"):
    probe = probe_h200 if gpu == "H200" else probe_h100
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] requesting 8x{gpu}...")
    try:
        output = probe.remote()
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] container started in {elapsed:.1f}s")
        print(output)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
