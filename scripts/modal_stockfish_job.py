"""Run one of the project's Stockfish jobs on Modal, on processor cores only (Justin, 2026-11-09: not on the Mac).

The container installs the pinned Stockfish with scripts/install_stockfish.py (checksum and bench number), checks
that it plays the moves pinned on the Mac (`ws_zero.stockfish.check_pinned_moves`), then runs the script with
`--workers 16 --out <file>` and sends back every file the script wrote. It stops after at most 30 minutes (about
$0.45).

    uv run modal run scripts/modal_stockfish_job.py --script ladder_matches \
        --args "--players random,grabber,sf1320,sf1" --out docs/gates/ladder.json
    uv run modal run scripts/modal_stockfish_job.py --script make_openings --out data/openings.json
"""
import base64
import shlex
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CPU_CORES, MEMORY_GIB, TIMEOUT_S = 16, 8, 1800
CORE_PRICE, GIB_PRICE = 0.0472, 0.0080  # Modal list prices, USD per hour (modal.com/pricing, read 2026-11-08)
SCRIPTS = ("ladder_matches", "make_openings")  # each takes --workers and --out

REQUIREMENTS = ROOT / "build" / "requirements-modal.txt"

if modal.is_local():  # this file is also imported inside the containers, where there is no repo or uv
    REQUIREMENTS.parent.mkdir(exist_ok=True)
    REQUIREMENTS.write_text(subprocess.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-dev", "--no-emit-project", "--format", "requirements-txt"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements(str(REQUIREMENTS))
         .add_local_python_source("ws_zero"))
for _name in ("install_stockfish",) + SCRIPTS:
    image = image.add_local_file(str(ROOT / "scripts" / f"{_name}.py"), f"/root/{_name}.py")
if (ROOT / "data" / "openings.json").exists() or not modal.is_local():
    image = image.add_local_file(str(ROOT / "data" / "openings.json"), "/data/openings.json")
app = modal.App("ws-zero-stockfish-job", image=image)


@app.function(cpu=CPU_CORES, memory=MEMORY_GIB * 1024, timeout=TIMEOUT_S)
def run(script: str, args: list[str], out_name: str) -> dict:
    import time

    from ws_zero.stockfish import check_pinned_moves
    start = time.time()
    install = subprocess.run(["python", "/root/install_stockfish.py"], capture_output=True, text=True)
    print(install.stdout + install.stderr, flush=True)
    install.check_returncode()
    check_pinned_moves()
    print("the engine plays the moves pinned on the Mac", flush=True)
    work = Path("/tmp/job")
    work.mkdir()
    subprocess.run(["python", f"/root/{script}.py", *args, "--workers", str(CPU_CORES), "--out", str(work / out_name)],
                   check=True)
    return {"files": {p.name: base64.b64encode(p.read_bytes()).decode("ascii") for p in work.iterdir()},
            "seconds": time.time() - start}


@app.local_entrypoint()
def main(script: str, out: str, args: str = ""):
    if script not in SCRIPTS:
        raise SystemExit(f"unknown script {script!r}; known: {', '.join(SCRIPTS)}")
    result = run.remote(script, shlex.split(args), Path(out).name)
    hourly = CPU_CORES * CORE_PRICE + MEMORY_GIB * GIB_PRICE
    print(f"{result['seconds']:.0f} s in the container, about ${hourly * result['seconds'] / 3600:.2f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        (Path(out).parent / name).write_bytes(base64.b64decode(data))
        print(f"wrote {Path(out).parent / name}")
