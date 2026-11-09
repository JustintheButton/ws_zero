"""Install the project's pinned Stockfish (decision 2026-11-09: Stockfish 19) and check that it is the right engine.

Downloads the official release file for this machine from github.com/official-stockfish/Stockfish, refuses it unless
its SHA-256 is the one pinned here (the digests GitHub listed for release sf_19 on 2026-11-09), unpacks it into
build/stockfish/sf_19/ and runs Stockfish's built-in `bench`: a fixed set of positions searched to a fixed depth.
The number of nodes it searches is the same for every correct build of one version, on any machine, so it is the
check that the Mac and Modal play with the same engine. Prints the engine's path.

    uv run python scripts/install_stockfish.py
"""
import argparse
import hashlib
import platform
import re
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "sf_19"
RELEASE = f"https://github.com/official-stockfish/Stockfish/releases/download/{TAG}/"
ASSETS = {  # (system, machine) -> (file, SHA-256)
    ("Darwin", "arm64"): ("stockfish-macos-universal.tar.gz",
                          "a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77"),
    ("Darwin", "x86_64"): ("stockfish-macos-universal.tar.gz",
                           "a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77"),
    ("Linux", "x86_64"): ("stockfish-linux-x86-64-universal.tar.gz",
                          "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"),
}
BENCH_NODES = 2497913  # "Bench: 2497913" in the release commit (edb0d9db67); measured the same on the Mac, 2026-11-09


def download(url: str, sha256: str, path: Path) -> None:
    """Fetch `url` to `path`, keeping it only if its SHA-256 is `sha256`."""
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ws-zero-setup"})) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(f"checksum mismatch for {url}: got {digest}, pinned {sha256}; nothing was kept")
    path.write_bytes(data)


def parse_bench(output: str) -> int:
    """The node count from the output of `stockfish bench`."""
    match = re.search(r"Nodes searched\s*:\s*(\d+)", output)
    if not match:
        raise SystemExit("no node count in the bench output:\n" + output[-500:])
    return int(match.group(1))


def find_engine(folder: Path) -> Path:
    """The engine binary among the unpacked files: the largest file whose name starts with "stockfish"."""
    files = [p for p in folder.rglob("stockfish*") if p.is_file() and not p.name.endswith(".tar.gz")]
    if not files:
        raise SystemExit(f"no stockfish binary under {folder}")
    return max(files, key=lambda p: p.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "stockfish" / TAG)
    args = parser.parse_args()
    key = (platform.system(), platform.machine())
    if key not in ASSETS:
        raise SystemExit(f"no pinned Stockfish file for {key}; known: {sorted(ASSETS)}")
    name, sha256 = ASSETS[key]
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / name
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != sha256:
        print(f"downloading {RELEASE + name}", flush=True)
        download(RELEASE + name, sha256, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(args.out, filter="data")
    engine = find_engine(args.out)
    engine.chmod(0o755)
    run = subprocess.run([str(engine), "bench"], capture_output=True, text=True, timeout=600)
    nodes = parse_bench(run.stdout + run.stderr)
    banner = subprocess.run([str(engine)], input="uci\nquit\n", capture_output=True, text=True, timeout=60).stdout
    print(next((line for line in banner.splitlines() if line.startswith("id name")), "id name ?"))
    print(f"bench: {nodes} nodes")
    if nodes != BENCH_NODES:
        raise SystemExit(f"bench gave {nodes} nodes, not the pinned {BENCH_NODES}: this is not the project's engine")
    print(engine)


if __name__ == "__main__":
    main()
