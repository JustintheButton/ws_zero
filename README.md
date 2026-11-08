# ws_zero

A from-scratch, reduced-scale implementation of AlphaZero for chess (Silver et al., 2017/2018), used to test
whether **Weight-Space Programming** (WSP; Takehi, Teoh, Zhang & Khattab, MIT CSAIL, Nov 2026), in which a
frontier LLM writes network weights directly with no gradient descent, gives a better starting point for
AlphaZero training than random initialization.

The research question is **sample efficiency in early training**: how many self-play games each starting point
needs to reach a fixed strength. Final superhuman strength is not a goal.

- Plan and progress: [`docs/checklist.md`](docs/checklist.md)
- Pre-registered protocol: [`docs/protocol.md`](docs/protocol.md)
- Decisions and deviations: [`docs/decisions.md`](docs/decisions.md)
- Gate reports: [`docs/gates/`](docs/gates)
- Lab notebook (visual progress page): `uv run python scripts/build_lab.py` writes `lab/build/lab.html`

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 and the exact package versions in `uv.lock`.

**macOS** (development, small runs):

```bash
uv sync
uv run pytest
```

**Windows PC** (training and evaluation runs): use WSL2 with Ubuntu. Install the current NVIDIA Windows
driver; do not install a CUDA toolkit inside WSL. Then, inside WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/JustintheButton/ws_zero.git && cd ws_zero
uv sync            # installs JAX with CUDA 13 and the CUDA 13.2 build of PyTorch
uv run pytest
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Weights & Biases**: run `uv run wandb login` once per machine. The key is stored in `~/.netrc`, never in
this repository. Every run also writes complete local logs (see below), which are the source of truth.

## Runs

Each run writes a folder under `runs/` (not committed) containing `config.yaml`, `meta.json` (git commit,
dirty flag, seed, host, package versions) and `metrics.jsonl`, and mirrors metrics to W&B when enabled.
