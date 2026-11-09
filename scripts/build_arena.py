"""Build the arena page (arena/index.html) and its files into build/arena/, ready to publish.

- models/<id>.json: each network, batch norm folded (ws_zero.export), its float32 arrays as base64 (artifacts
  serve text, not raw binary files);
- data/models.json: the opponents, each with the positions its browser copy must reproduce before it is offered;
- data/games.json: the game archive and score tables (for now the stand-ins' demo games, arena/data/);
- js/: the browser code; index.html: the page.
Also downloads chess.js and TensorFlow.js into build/vendor/ (once) for tests/test_arena_js.py.

    uv run python scripts/build_arena.py
"""
import argparse
import base64
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from ws_zero.arena import network_cases, network_lines, pawn_pusher, untrained
from ws_zero.export import model_file, write_model

ROOT = Path(__file__).resolve().parents[1]
VENDOR = {"chess.min.js": "https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js",
          "tf.min.js": "https://cdnjs.cloudflare.com/ajax/libs/tensorflow/4.22.0/tf.min.js"}

STAND_INS = "Stand-ins (before training)"
MODELS = [  # id, name, group, about, params (None: not a network)
    ("untrained", "Untrained network", STAND_INS,
     "The random-start network before any training: every move equally likely and every position judged equal. "
     "With search it still finds mates and sees material won or lost at the end of a line.", untrained),
    ("pawn-pusher", "Hand-written pawn-pusher", STAND_INS,
     "Phase 3's hand-written network: three nonzero weights out of 571,276, which make it want to push a pawn two "
     "squares. It judges every position equal.", pawn_pusher),
    ("material", "Material counter", STAND_INS,
     "The Phase 4 gate's stand-in, not a network: every move equally likely, and a position is worth its material "
     "balance (pawn 1, knight and bishop 3, rook 5, queen 9). With search it grabs material and avoids losing its own.",
     None),
]


def fetch_vendor():
    out = ROOT / "build" / "vendor"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in VENDOR.items():
        if not (out / name).exists():
            urllib.request.urlretrieve(url, out / name)


def commit_label() -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + (" (with uncommitted changes)" if git("status", "--porcelain") else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "arena")
    parser.add_argument("--games", type=Path, default=ROOT / "arena" / "data" / "demo-games.json")
    args = parser.parse_args()
    fetch_vendor()
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for sub in ("models", "data", "js"):
        (out / sub).mkdir(parents=True)

    manifest = []
    for model_id, name, group, about, make in MODELS:
        entry = {"id": model_id, "name": name, "group": group, "about": about}
        if make is None:
            entry["kind"] = "material"
        else:
            params = make()
            path = out / "models" / model_id
            write_model(path, params, {"id": model_id, "name": name})
            model = json.loads(model_file(path, ".json").read_text())
            model["data"] = base64.b64encode(model_file(path, ".bin").read_bytes()).decode("ascii")
            model_file(path, ".json").write_text(json.dumps(model, separators=(",", ":")))
            model_file(path, ".bin").unlink()
            entry.update(kind="network", file=model_id, check=network_cases(params, network_lines()))
        manifest.append(entry)
    (out / "data" / "models.json").write_text(json.dumps(manifest, separators=(",", ":")))
    shutil.copy(args.games, out / "data" / "games.json")
    for js in (ROOT / "arena" / "js").glob("*.js"):
        shutil.copy(js, out / "js" / js.name)
    page = (ROOT / "arena" / "index.html").read_text().replace("__COMMIT__", commit_label())
    (out / "index.html").write_text(page)

    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "index.html")
    print(f"wrote {out}: index.html and {len(files)} files")
    print(json.dumps({f: str(out / f) for f in files}, indent=1))


if __name__ == "__main__":
    main()
