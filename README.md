# Memory Machines

A research codebase for a cheap LLM — a causal encoder-decoder transformer built
around **Compressed Sparse Attention 2 (CSA2)** and a handful of auxiliary memory
mechanisms (GDN-2, mHC, Engram, Titans, Continuum Memory, LIV). This is an alpha
research project, not a product: expect rough edges, and treat the default config
as a starting point rather than a recommendation.

## What's here

```
src/memory_machines/
├── models/            # backbones, blocks, model I/O, registry
├── blocks/            # CSA2, GDN-2, mHC, Engram, Titans, CMS, LIV, FFNs
├── modules/           # PyTorch Lightning wrapper (LitResearchSystem)
├── data/              # dataset + Lightning DataModule
├── ops/               # dispatcher + reference kernels (Taichi + CPU)
├── kernels/           # Taichi kernels + CPU loader
├── losses/            # classification losses
├── training/          # training engine
├── callbacks/         # logging callbacks
├── integrations/      # Hugging Face transformers compatibility
└── utils/             # config schema + validation
```

Models are picked by name through a registry (`register_model` / `get_model`).
Registered: `ExampleBackbone` (scaffold / smoke-test) and `CheapLLM` (causal
encoder-decoder LLM that owns the CED encoder and composes `CausalDecoder`).

## Installation

Python 3.11+. PyTorch resolves to CPU wheels by default; on a CUDA machine
override the index (no file edits, no extras):

```bash
# CPU (default)
uv sync --group dev

# NVIDIA GPU (CUDA cu128)
uv sync --group dev \
        --no-sources-package torch --no-sources-package torchvision \
        --no-sources-package torchaudio \
        --index https://download.pytorch.org/whl/cu128
```

List all three torch packages — a partial list leaves the rest on CPU wheels and
produces a mixed build. Use `--index`, not `--default-index` (which replaces PyPI
and breaks non-torch resolution). After syncing on a GPU machine, run
`uv lock` once before committing so the committed lock keeps the marker-split form.

### Library consumers

```bash
pip install MemoryMachines   # or: uv add MemoryMachines
```

torch is a normal dependency, so the standard install works with no flags and the
consumer's own package manager resolves the right wheel for their hardware.

## Verify

```python
import torch

print(torch.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Usage

Training and evaluation are Hydra-driven; configs live in `configs/`.

```bash
# 1 train + 1 val step (fast dev run)
uv run python scripts/train.py trainer.fast_dev_run=true

# full run
uv run python scripts/train.py

# evaluate a checkpoint on the test set
uv run python scripts/eval.py checkpoint=path/to/ckpt.ckpt

# export trained weights to Hugging Face format
uv run python scripts/export_hf.py --checkpoint path/to/ckpt.ckpt --output-dir ./hf_model
```

The default dataset is synthetic (`SimpleTensorDataset`) — swap in real ingestion
by editing `configs/data/`. The Lightning wrapper (`LitResearchSystem`) owns loss,
step metrics, and optimizer config only; the model itself is injected and never
knows Lightning exists.

## Configuration

Everything is configured through `MemoryMachineConfig` in
`src/memory_machines/utils/config_schema.py`. It validates cross-group
invariants at boot (before GPU init), so a bad config fails fast with a clear
message rather than blowing up deep inside a layer. Notable knobs:

- **Attention**: `ced_attention_variant` (`dense` / `compressed` /
  `compressed_sparse` / `heavily_compressed`), sliding-window sizes, CSA2
  compression ratios, top-K, hierarchical indexer pool.
- **Encoder**: parallel attention ∥ GDN-2 via Single-Pass mHC
  (`ced_attention_gdn2_parallel`, default on). GDN-2 mixes the full encoder
  sequence — never windowed.
- **Decoder**: always plain CSA2, inherits global context through the encoder's
  KV (Eq. 1). Optional Looped Language Model mode
  (`ced_decoder_looping`) with per-position Q-exit.
- **Memory**: Engram hashed n-gram tables (RAM / SSD / auto offload),
  Self-Referential Titans fast weights, Continuum Memory System levels.
- **Asymmetry**: `ced_decoder_hidden_size` gives a wide encoder / narrow decoder
  (the global KV projection doubles as the cross-attention downsample).

## Testing

```bash
uv run --no-sync pytest                       # full suite
uv run --no-sync pytest tests/test_ced_loop.py -q   # focused
```

The suite is 748 tests covering block shapes, CSA2 modes, GDN-2 reference
parity, Engram offload, CED memory/looping, and config-schema coverage (every
`config.X` a block reads is a field — no dead fields). Doctests in
`src/memory_machines/models/blocks` and `src/memory_machines/utils` run as part
of the default `pytest` invocation via `--doctest-modules` (they are listed in
`testpaths`).

## License

MIT — Copyright (c) 2026 João Gabriel Souza Amaral