"""Run the Phase 4 self-play speed measurement (scripts/selfplay_speed.py) on one Modal GPU.

    uv run modal run scripts/modal_selfplay_speed.py
    uv run modal run scripts/modal_selfplay_speed.py --gpu A10 --plies 20
    uv run modal run scripts/modal_selfplay_speed.py --plies 20 --prior-peaks 0,0.002,0.01,0.05   # deeper trees

The container installs exactly the versions in uv.lock (exported for Linux) and stops after at most 20 minutes
(about $0.40 on an L4), however many prior peaks it runs. Results and the cost per 1,000 games go to
docs/gates/phase-4-selfplay-modal.json, or with --prior-peaks to docs/gates/phase-4-selfplay-modal-depth.json.
"""
import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CPU_CORES, MEMORY_GIB = 4, 16
TIMEOUT_S = 1200

# Modal list prices, USD per hour (modal.com/pricing, read 2026-11-08), as in modal_bench.py.
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
         .add_local_file(str(ROOT / "scripts" / "selfplay_speed.py"), "/root/selfplay_speed.py"))
app = modal.App("ws-zero-selfplay-speed", image=image)


@app.function(gpu="L4", cpu=CPU_CORES, memory=MEMORY_GIB * 1024, timeout=TIMEOUT_S)
def measure(label: str, games: int, plies: int, prior_peaks: list[float]) -> list[dict]:
    """One measurement per prior peak, one after the other in this container (so the time limit covers them all)."""
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    results = []
    for peak in prior_peaks:
        print(f"== prior peak {peak}", flush=True)
        subprocess.run(["python", "/root/selfplay_speed.py", "--games", str(games), "--plies", str(plies),
                        "--prior-peak", str(peak), "--label", label, "--out", "/tmp/speed.json"], check=True)
        results.append({**json.loads(Path("/tmp/speed.json").read_text()), "nvidia_smi": gpu})
    return results


@app.local_entrypoint()
def main(gpu: str = "L4", games: int = 100, plies: int = 40, prior_peaks: str = "0"):
    if gpu not in GPU_PRICE:  # before the paid run: its result would be lost when the price is looked up
        raise SystemExit(f"no price for GPU {gpu!r}; known: {', '.join(GPU_PRICE)}")
    peaks = [float(p) for p in prior_peaks.split(",")]
    results = measure.with_options(gpu=gpu).remote(gpu, games, plies, peaks)
    hourly = GPU_PRICE[gpu] + CPU_CORES * CORE_PRICE + MEMORY_GIB * GIB_PRICE
    for result in results:
        result["usd_per_hour"] = hourly
        depth = f"deepest line {int(sorted(result['deepest_line'])[len(result['deepest_line']) // 2])} moves"
        for e in result["estimates"]:
            e["usd_per_1000_games"] = 1000 * hourly / e["games_per_hour"]
            print(f"{gpu}, prior peak {result['prior_peak']}, {depth}, {result['median_ply_s']:.2f} s per ply: "
                  f"chunk of {e['chunk_plies']} plies: {e['games_per_hour']:,.0f} games/hour, "
                  f"${e['usd_per_1000_games']:.2f} per 1,000 games")
    header = {"cpu_cores": CPU_CORES, "memory_gib": MEMORY_GIB, "gpu_price_per_hour": GPU_PRICE, "gpu": gpu}
    if peaks == [0.0]:  # the plain measurement, with the untrained network's own move probabilities
        out, body = ROOT / "docs" / "gates" / "phase-4-selfplay-modal.json", {"result": results[0]}
    else:  # the same measurement at several tree depths
        out, body = ROOT / "docs" / "gates" / "phase-4-selfplay-modal-depth.json", {"results": results}
    out.write_text(json.dumps({**header, **body}, indent=1))
    print(f"wrote {out}")
