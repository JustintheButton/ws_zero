"""Play the ladder's opponents against each other (scripts/ladder_matches.py) on Modal, on processor cores only.

The container installs the pinned Stockfish with scripts/install_stockfish.py (checksum and bench number), checks
that it plays the moves pinned on the Mac (`ws_zero.stockfish.check_pinned_moves`), and then plays the matches.
It stops after at most 30 minutes (about $0.45 at 16 cores).

    uv run modal run scripts/modal_ladder_matches.py --players random,grabber,sf1320,sf1 --out docs/gates/ladder.json
"""
import base64
import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CPU_CORES, MEMORY_GIB, TIMEOUT_S = 16, 8, 1800
CORE_PRICE, GIB_PRICE = 0.0472, 0.0080  # Modal list prices, USD per hour (modal.com/pricing, read 2026-11-08)

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
         .add_local_file(str(ROOT / "scripts" / "ladder_matches.py"), "/root/ladder_matches.py"))
app = modal.App("ws-zero-ladder-matches", image=image)


@app.function(cpu=CPU_CORES, memory=MEMORY_GIB * 1024, timeout=TIMEOUT_S)
def play(players: str, openings: int) -> dict:
    import time

    from ws_zero.stockfish import check_pinned_moves
    start = time.time()
    install = subprocess.run(["python", "/root/install_stockfish.py"], capture_output=True, text=True)
    print(install.stdout + install.stderr, flush=True)
    install.check_returncode()
    check_pinned_moves()
    print("the engine plays the moves pinned on the Mac", flush=True)
    subprocess.run(["python", "/root/ladder_matches.py", "--players", players, "--openings", str(openings),
                    "--workers", str(CPU_CORES), "--out", "/tmp/ladder.json"], check=True)
    return {"summary": json.loads(Path("/tmp/ladder.json").read_text()),
            "games_gz": base64.b64encode(Path("/tmp/ladder-games.json.gz").read_bytes()).decode("ascii"),
            "install": install.stdout.strip().splitlines(), "seconds": time.time() - start}


@app.local_entrypoint()
def main(players: str = "random,grabber,sf1320,sf1", openings: int = 100, out: str = ""):
    result = play.remote(players, openings)
    hourly = CPU_CORES * CORE_PRICE + MEMORY_GIB * GIB_PRICE
    print(f"{result['seconds']:.0f} s in the container, about ${hourly * result['seconds'] / 3600:.2f}")
    if out:
        summary = {**result["summary"], "machine": "Modal, " + "; ".join(result["install"][:1]),
                   "seconds": result["seconds"]}
        Path(out).write_text(json.dumps(summary, indent=1))
        Path(out.removesuffix(".json") + "-games.json.gz").write_bytes(base64.b64decode(result["games_gz"]))
        print(f"wrote {out}")
