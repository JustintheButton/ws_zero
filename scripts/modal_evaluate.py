"""Evaluate agents against the ladder (scripts/evaluate_agent.py) on Modal: one L4 GPU and 8 processor cores per job.

Each job is agent:seed and runs in its own container, side by side. A container installs the pinned Stockfish
(checksum, bench number, the moves pinned on the Mac), then plays the agent against every opponent. Each stops after
at most 40 minutes (about $0.87). Results go to docs/gates/phase-5-eval-<agent>-seed<seed>.json.

    uv run modal run scripts/modal_evaluate.py --jobs untrained:0,untrained:1,material:0,material:1
"""
import base64
import shlex
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU, CPU_CORES, MEMORY_GIB, TIMEOUT_S = "L4", 8, 16, 2400
GPU_PRICE, CORE_PRICE, GIB_PRICE = 0.799, 0.0472, 0.0080  # Modal list prices, USD per hour (read 2026-11-08)

REQUIREMENTS = ROOT / "build" / "requirements-modal.txt"

if modal.is_local():  # this file is also imported inside the containers, where there is no repo or uv
    REQUIREMENTS.parent.mkdir(exist_ok=True)
    REQUIREMENTS.write_text(subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-dev", "--no-emit-project", "--format", "requirements-txt"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements(str(REQUIREMENTS))
         .add_local_python_source("ws_zero")
         .add_local_file(str(ROOT / "scripts" / "install_stockfish.py"), "/root/install_stockfish.py")
         .add_local_file(str(ROOT / "scripts" / "evaluate_agent.py"), "/root/evaluate_agent.py")
         .add_local_file(str(ROOT / "data" / "openings.json"), "/data/openings.json"))
app = modal.App("ws-zero-evaluate", image=image)


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_GIB * 1024, timeout=TIMEOUT_S)
def evaluate(agent: str, seed: int, extra: list[str]) -> dict:
    import time

    from ws_zero.stockfish import check_pinned_moves
    start = time.time()
    install = subprocess.run(["python", "/root/install_stockfish.py"], capture_output=True, text=True)
    print(install.stdout + install.stderr, flush=True)
    install.check_returncode()
    check_pinned_moves()
    work = Path("/tmp/job")
    work.mkdir()
    name = f"phase-5-eval-{agent}-seed{seed}.json"
    subprocess.run(["python", "/root/evaluate_agent.py", "--agent", agent, "--seed", str(seed), "--threads",
                    str(CPU_CORES), *extra, "--out", str(work / name)], check=True)
    return {"files": {p.name: base64.b64encode(p.read_bytes()).decode("ascii") for p in work.iterdir()},
            "seconds": time.time() - start}


@app.local_entrypoint()
def main(jobs: str = "untrained:0,untrained:1", args: str = "", out_dir: str = "docs/gates"):
    pairs = [(agent, int(seed)) for agent, seed in (job.split(":") for job in jobs.split(","))]
    calls = [evaluate.spawn(agent, seed, shlex.split(args)) for agent, seed in pairs]
    hourly = GPU_PRICE + CPU_CORES * CORE_PRICE + MEMORY_GIB * GIB_PRICE
    for (agent, seed), call in zip(pairs, calls):
        try:
            result = call.get()
        except Exception as e:  # report and keep the other jobs
            print(f"{agent}:{seed} failed: {e!r}")
            continue
        print(f"{agent}:{seed}: {result['seconds']:.0f} s in the container, about ${hourly * result['seconds'] / 3600:.2f}")
        for name, data in result["files"].items():
            (ROOT / out_dir / name).write_bytes(base64.b64decode(data))
            print(f"wrote {out_dir}/{name}")
