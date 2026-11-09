"""Run the Phase 3 benchmark on several Modal GPU types, to choose the container type by cost per game (protocol §7).

    uv run modal run scripts/modal_bench.py
    uv run modal run scripts/modal_bench.py --gpus L4,A10

The container installs exactly the versions in uv.lock (exported for Linux). Results and estimated cost per
1,000 games go to docs/gates/phase-3-bench-modal.json.
"""
import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CPU_CORES, MEMORY_GIB = 4, 16

# Modal list prices, USD per hour (modal.com/pricing, read 2026-11-08).
GPU_PRICE = {"T4": 0.590, "L4": 0.799, "A10": 1.102, "L40S": 1.951, "A100-40GB": 2.099, "A100-80GB": 2.498,
             "H100": 3.949, "H200": 4.540, "B200": 6.250}
CORE_PRICE, GIB_PRICE = 0.0472, 0.0080


REQUIREMENTS = ROOT / "build" / "requirements-modal.txt"

if modal.is_local():  # this file is also imported inside the containers, where there is no repo or uv
    REQUIREMENTS.parent.mkdir(exist_ok=True)
    REQUIREMENTS.write_text(subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-dev", "--no-emit-project", "--format", "requirements-txt"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements(str(REQUIREMENTS))
         .add_local_python_source("ws_zero")
         .add_local_file(str(ROOT / "scripts" / "bench_phase3.py"), "/root/bench_phase3.py"))
app = modal.App("ws-zero-bench", image=image)


@app.function(gpu="L4", cpu=CPU_CORES, memory=MEMORY_GIB * 1024, timeout=1800)
def bench(label: str, batches: str) -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    subprocess.run(["python", "/root/bench_phase3.py", "--batches", batches, "--label", label,
                    "--out", "/tmp/bench.json"], check=True)
    result = json.loads(Path("/tmp/bench.json").read_text())
    result["nvidia_smi"] = gpu
    return result


@app.local_entrypoint()
def main(gpus: str = "T4,L4,A10,L40S,A100-40GB,H100", batches: str = "100,400,1600,6400"):
    unknown = [g for g in gpus.split(",") if g not in GPU_PRICE]
    if unknown:  # before the paid runs: their results would be lost when the price is looked up
        raise SystemExit(f"no price for GPU {', '.join(map(repr, unknown))}; known: {', '.join(GPU_PRICE)}")
    calls = {g: bench.with_options(gpu=g).spawn(g, batches) for g in gpus.split(",")}
    results = {}
    for g, call in calls.items():
        try:
            results[g] = call.get()
        except Exception as e:  # report and keep the other GPU types
            results[g] = {"error": repr(e)}
            print(f"{g}: failed: {e!r}")
    print(f"\n{'GPU':10s} {'$/hour':>7s}  {'games in flight':>15s}  {'games/hour':>10s}  {'$ per 1,000 games':>17s}")
    for g, r in results.items():
        if "error" in r:
            continue
        hourly = GPU_PRICE[g] + CPU_CORES * CORE_PRICE + MEMORY_GIB * GIB_PRICE
        r["usd_per_hour"] = hourly
        for e in r["estimates"]:
            e["usd_per_1000_games"] = 1000 * hourly / e["games_per_hour"]
            if e["chunk_plies"] == 512:
                print(f"{g:10s} {hourly:7.2f}  {e['batch']:15d}  {e['games_per_hour']:10,.0f}  "
                      f"{e['usd_per_1000_games']:17.2f}")
    out = ROOT / "docs" / "gates" / "phase-3-bench-modal.json"
    out.write_text(json.dumps({"cpu_cores": CPU_CORES, "memory_gib": MEMORY_GIB, "gpu_price_per_hour": GPU_PRICE,
                               "results": results}, indent=1))
    print(f"\nwrote {out}")
