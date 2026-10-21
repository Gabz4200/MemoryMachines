# Memory Machines

A research codebase for a cheap LLM: a causal encoder-decoder transformer built
around **Compressed Sparse Attention 2 (CSA2)** with auxiliary memory mechanisms
(GDN-2, mHC, Engram, Titans, Continuum Memory, LIV). Alpha research project, not
a product: expect rough edges, and treat the default config as a starting point
rather than a recommendation.

## What's here

Three models, picked by name through a registry:

- `example`: minimal flat scaffold: token embedding, grouped-query attention
  + SwiGLU blocks, a norm, and an LM head. Good for smoke tests.
- `cheapllm`: the causal encoder-decoder LLM. Owns a CED encoder (parallel
  attention ∥ memory mixer per layer, or plain CSA2) and composes a `CausalDecoder`
  seeded with the encoder's global KV. Optional asymmetric decoder width and
  Looped Language Model mode with per-position Q-exit.
- `blt` / `ByteLatentTransformer`: byte-level model: local byte encoders and
  decoders (hash n-gram embeddings, causal byte transformers) wrapped around the
  CED latent stack operating on patches, with Perceiver-AR masked-patch
  generation.

Under the hood: CSA2 attention variants (dense, compressed, compressed-sparse,
heavily compressed), sliding windows, GDN-2 memory mixers, Engram hashed n-gram
tables, Self-Referential Titans fast weights, Continuum Memory System levels, DyT
and layer norms, SwiGLU FFNs, and Taichi-backed kernels.

## Installation

Python 3.11+. PyTorch resolves to CPU wheels by default; on a CUDA machine, point
`uv` at the CUDA index (no file edits, no extras):

```bash
# CPU (default)
uv sync --group dev

# NVIDIA GPU (CUDA cu128)
uv sync --group dev \
        --no-sources-package torch --no-sources-package torchvision \
        --no-sources-package torchaudio \
        --index https://download.pytorch.org/whl/cu128
```

List all three torch packages. A partial list leaves the rest on CPU wheels and
produces a mixed build. Use `--index`, not `--default-index` (which replaces PyPI
and breaks non-torch resolution). After syncing on a GPU machine, run
`uv lock` once before committing so the committed lock keeps the marker-split
form.

Library consumers can install with no flags:

```bash
pip install MemoryMachines   # or: uv add MemoryMachines
```

torch is a normal dependency, so the standard install resolves the right wheel
for the consumer's hardware.

## Usage

Training and evaluation are Hydra-driven; configs live in `configs/`.

```bash
# 1 train step + 1 val step (fast dev run)
uv run python scripts/train.py trainer.fast_dev_run=true

# full run
uv run python scripts/train.py

# evaluate a checkpoint on the test set
uv run python scripts/eval.py checkpoint=path/to/ckpt.ckpt

# export trained weights to Hugging Face format
uv run python scripts/export_hf.py --checkpoint path/to/ckpt.ckpt --output-dir ./hf_model
```

The default dataset is synthetic. To use real data, edit `configs/data/`.
The Lightning wrapper owns only loss, step metrics, and optimizer config;
the model itself never knows Lightning exists.

## Configuration

Everything goes through `MemoryMachineConfig` in
`src/memory_machines/utils/config_schema.py`, which validates cross-group
invariants at boot (before GPU init), so a bad config fails fast with a clear
message. Common knobs:

- **Models**: `model.name` picks `example`, `cheapllm`, or `blt`;
  `vocab_size`, `hidden_size`, `num_layers`.
- **Encoder**: parallel attention ∥ memory mixer (GDN-2, or Titans on alternating
  layers) behind one pre-norm, fused by a `2d -> d` down-projection. Layer 0's
  attention branch is plain SWA with the smallest window; later layers run CSA2.
  The memory branch mixes the full sequence — never windowed — so it supplies
  global context instead of a widened attention window.
- **Decoder**: always plain CSA2, inheriting global context through the encoder's
  KV. Optional asymmetric width (`ced_decoder_hidden_size`) and Looped Language
  Model mode with per-position Q-exit.
- **Attention**: `ced_attention_variant` (dense / compressed / compressed_sparse /
  heavily_compressed), sliding-window sizes, CSA2 compression ratios, top-K,
  hierarchical indexer pool.
- **Memory**: Engram hashed n-gram tables (RAM / SSD / auto offload),
  Self-Referential Titans fast weights, Continuum Memory System levels.

## Testing

```bash
uv run --no-sync pytest                       # full suite
uv run --no-sync pytest tests/test_ced_loop.py -q   # focused run
```

Doctests in `src/memory_machines/models/blocks` and `src/memory_machines/utils`
run as part of the default `pytest` invocation.

## License

MIT — Copyright (c) 2026 João Gabriel Souza Amaral
