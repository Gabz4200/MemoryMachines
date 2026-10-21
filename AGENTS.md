# AGENTS.md

## Project Overview

MemoryMachines is a research codebase for a cheap LLM: a causal encoder-decoder transformer built around **Compressed Sparse Attention 2 (CSA2)** with auxiliary memory mechanisms (GDN-2, mHC, Engram, Titans, Continuum Memory, LIV). Alpha research project, not a product: expect rough edges, treat the default config as a starting point rather than a recommendation.

- **Version:** 0.1.0
- **License:** MIT (Copyright 2026 João Gabriel Souza Amaral)
- **Python:** 3.11+
- **Framework:** PyTorch, PyTorch Lightning, Hydra, einops, omegaconf, transformers, safetensors, taichi
- **Package layout:** `src/` layout; package import root is `memory_machines`
- **Build system:** hatchling (no `setup.py`)
- **Package manager:** `uv` (never `pip install` directly into this project)

Three models, picked by name through the registry (`src/memory_machines/models/registry.py`):

1. `example` — minimal flat scaffold: token embedding, grouped-query attention + SwiGLU blocks, a norm, and an LM head. Good for smoke tests.
2. `cheapllm` — the causal encoder-decoder LLM. Owns a CED encoder (parallel attention ∥ memory mixer per layer, or plain CSA2) and composes a `CausalDecoder` seeded with the encoder's global KV (Eq. 1). Optional asymmetric decoder width and Looped Language Model mode with per-position Q-exit.
3. `blt` / `ByteLatentTransformer` — byte-level model: local byte encoders and decoders (hash n-gram embeddings, causal byte transformers) wrapped around the CED latent stack operating on patches, with Perceiver-AR masked-patch generation.

Everything is driven by `MemoryMachineConfig` in `src/memory_machines/utils/config_schema.py`, which validates cross-group invariants at boot **before GPU init**, so a bad config fails fast with a clear message.

## Setup Commands

Python 3.11+. PyTorch resolves to CPU wheels by default; on a CUDA machine, point `uv` at the CUDA index (no file edits, no extras):

```bash
# CPU (default)
uv sync --group dev

# NVIDIA GPU (CUDA cu128)
uv sync --group dev \
        --no-sources-package torch --no-sources-package torchvision \
        --no-sources-package torchaudio \
        --index https://download.pytorch.org/whl/cu128
```

List all three torch packages. A partial list leaves the rest on CPU wheels and produces a mixed build. Use `--index`, not `--default-index` (which replaces PyPI and breaks non-torch resolution). After syncing on a GPU machine, run `uv lock` once before committing so the committed lock keeps the marker-split form.

Library consumers can install with no flags:

```bash
pip install MemoryMachines   # or: uv add MemoryMachines
```

`torch` is a normal dependency, so the standard install resolves the right wheel for the consumer's hardware.

The repo is committed with a marker-split `uv.lock`. Do not commit a lock written under the CUDA override; `uv lock` without the override restores the marker-split form.

## Development Workflow

Training and evaluation are Hydra-driven; configs live in `configs/`. The Lightning wrapper owns only loss, step metrics, and optimizer config; the model itself never knows Lightning exists.

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

Configs:
- `configs/default.yaml` — top-level Hydra compose (model, system, data, callbacks, trainer)
- `configs/model/default.yaml` — currently `ExampleBackbone` (switch to `CheapLLM` or `BLTBackbone` to change model family)
- `configs/system/default.yaml` — `LitResearchSystem` (learning_rate, weight_decay, ignore_index)
- `configs/trainer/default.yaml` — `pytorch_lightning.Trainer`
- `configs/data/default.yaml` — `ResearchDataModule` (data_dir, batch_size, num_workers)
- `configs/callbacks/default.yaml` — side-effect callbacks (commented out by default)

Notebooks use jupytext (`notebooks/01_quickstart.py`, synced to `.ipynb` via `.jupytext.toml`). The pre-commit jupytext hook uses `args: [--sync --pre-commit]` with `pass_filenames: false`.

## Testing Instructions

Run the full suite:

```bash
uv run --no-sync pytest
```

Run a focused test file:

```bash
uv run --no-sync pytest tests/test_ced_loop.py -q
```

Run doctests in `src/memory_machines/models/blocks` and `src/memory_machines/utils` (these run as part of the default `pytest` invocation via `addopts="--doctest-modules"`):

```bash
pytest --doctest-modules src/memory_machines/models/blocks src/memory_machines/utils
```

- `tests/test_blocks_shapes.py::tiny_cfg()` is the shared tiny config (hidden 32, heads 4, kv 2, csa_top_k 2, csa_num_output_groups 2). Legacy modules import it from `tests.test_blocks_shapes`; new tests should use `tests/conftest.py` (`make_tiny_cfg`, `make_inputs`).
- Coverage: `uv run --no-sync pytest --cov=memory_machines -q` (line coverage; ~89% for the full suite). If the C tracer segfaults, retry with `COVERAGE_CORE=sysmon` on Python 3.12+.
- Doctests are 137 and must pass (quality gate).
- Suite is 1147 tests.
- No sleeps, no network, no wall-clock dependence in unit tests. `tmp_path` for files; `monkeypatch`/`mocker` only at true externals. `pytest.raises` for expected exceptions. `parametrize` with descriptive `ids`. Mirror source layout (`src/foo/bar.py` -> `tests/foo/test_bar.py`).

## Code Style Guidelines

- **Ruff:** line-length 200, target py311, src=["src"].
  - Check: `uv run --no-sync ruff check --output-format concise <files>`
  - Format check: `uv run --no-sync ruff format --check`
  - Format (apply): `uv run --no-sync ruff format <files>`
  - Ruff bypasses `pyproject exclude` for explicit CLI filenames; block excluded files with hook-level `exclude:` if needed.
- **Pyrefly:** strict type checking (`check-unannotated-defs = true`, `untyped-def-behavior = "check-and-infer-return-type"`).
  - Check: `uv run --no-sync pyrefly check` (quality gate: 0 errors).
- **aislop:** `uv run --no-sync aislop scan .` (quality gate: 0 errors; 24 complexity warnings intentionally LEFT per user — do not disable rules to pass).
- **Docstrings:** PyTorch style (Sphinx/reStructuredText: signature line first, `Args:`/`Returns:`/`Raises:`/`Examples::`/`.. note::`/`.. warning::`, raw strings `r"""..."""`, `:class:`/`:func:`/`:meth:` cross-references, LaTeX `:math:` for tensor shapes). One-line imperative summary first.
- **Comments:** only non-obvious intent (constraints, invariants, compatibility requirements, reason for an unusual choice). Never restate names, narrate history, or leave commented-out code.
- **Fail fast:** let programmer errors crash loudly; handle only expected failures at genuine external boundaries (network, DB, filesystem, subprocess, user file, third-party payload). No silent recovery, default value, alternate path, or fabricated success for a programming error.
- **ML:** keep the math model independent of config/ingestion/loops/logging/checkpointing/distribution. CPU first; GPU via override on the GPU host. Reference path runs everywhere; optimized kernels sit behind the same contract. Never hard-code one accelerator as default; never hide allocation in `forward`; never catch kernel errors to mask wrong shapes.

## Build and Deployment

Build:
```bash
uv build
```

This is a library (`src/memory_machines`) + scripts, not a deployable service. There is no Dockerfile, no CI pipeline file beyond pre-commit, and no production deployment steps documented.

Artifacts are gitignored: `data/`, `runs/`, `outputs/`, `wandb/`, `*.ckpt`, `*.pt`, `*.pth`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.pyrefly/`, `__pycache__/`, `.venv/`, `uv.lock.tmp`.

Pre-commit (`.pre-commit-config.yaml`, `fail_fast: true`). No `.git/hooks/pre-commit` and pre-commit can't install the pyrefly hook env → run hooks manually:

```bash
uv run --no-sync pre-commit run --all-files
```

Individual hooks (from `.pre-commit-config.yaml`):
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files` (pre-commit-hooks v6.0.0)
- `ruff-check` (args: [--fix]), `ruff-format` (ruff-pre-commit v0.17.0)
- `jupytext` (args: [--sync, --pre-commit], pass_filenames: false) (jupytext v1.19.6)
- `pyrefly-check` (entry: `uv run --no-sync pyrefly check`, language: system, always_run: true) (pyrefly-pre-commit v1.3.2)
- `pytest` (entry: `uv run --no-sync pytest -q`, language: system, always_run: true) (pytest-pre-commit v1.0.0)
- `aislop` (aislop v0.18.1)

Because `fail_fast: true`, pre-commit stops at the first failing hook.

## Pull Request Guidelines

- Title format: conventional commits (type(scope): description).
- Required checks before submission: `uv run --no-sync ruff check --output-format concise`, `uv run --no-sync ruff format --check`, `uv run --no-sync pyrefly check`, `uv run --no-sync pytest -q`, `uv run --no-sync aislop scan .`.
- One composition root. No service locators, no import-time side effects.
- Interface only where implementations vary or a test seam earns it. No interface with one implementation, no factory for one product, no config for a changeless value.
- State flows through arguments and returns. Adapters return typed batches. Composition root wires models, losses, optimizers, datasets, trainers.
- Fixes must show red-green: reproduce the symptom before, passing after.
- Behavior-preserving refactors must not require test changes.
- Add or update tests for the code you change.
- Do not hardcode, print, commit, or transmit secrets. Never log credentials, tokens, or unrequested personal data.

## Debugging and Troubleshooting

- **VIRTUAL_ENV mismatch warning:** harmless when running `uv run --no-sync pytest`.
- **`--cov` segfault (historical):** did not reproduce on Python 3.13 / coverage 7.16; retry with `COVERAGE_CORE=sysmon` if the C tracer crashes again.
- **`data/datamodule.py` fails to collect:** unrelated to this project; skip.
- **Pre-commit pyrefly hook env can't install:** run `uv run --no-sync pyrefly check` manually instead of `pre-commit run`.
- **Mixed torch build:** installing only some torch packages under `--index` leaves the rest on CPU wheels. List all three (torch, torchvision, torchaudio).
- **SparseIndexer.topk warning:** silenced via `filterwarnings` in `pyproject.toml` (`ignore:SparseIndexer.topk:UserWarning`); expected for tiny doctest configs.
- **Lock file under CUDA override:** do not commit a lock written under the torch override; `uv lock` without the override restores the marker-split form.

## Key Architecture Reference

- `src/memory_machines/utils/config_schema.py` — `MemoryMachineConfig` (single config interface; validates at boot before GPU init). Common knobs: `model.name` (`example`, `cheapllm`, `blt`), `vocab_size`, `hidden_size`, `num_layers`, `ced_*` (CED encoder/decoder), `ced_attention_variant` (`dense`/`compressed`/`compressed_sparse`/`heavily_compressed`), `ced_attention_gdn2_parallel` (encoder parallel-hybrid on/off), `ced_gdn2_*` (GDN-2 knobs), `titan_*` (Titans fast weights), `engram_*` (Engram hashed n-gram tables), `cms_*` (Continuum Memory System levels), `blt_*` (BLT byte model).
- `src/memory_machines/models/backbone.py` — `ExampleBackbone`, `CheapLLM`, `BLTBackbone` (registered via `@register_model`).
- `src/memory_machines/models/blocks/ced.py` — `CausalEncoderDecoderStack`, `CausalDecoder`, `CEDGlobalKVProjection`, `VocabularyUpProjection`, `AdaptiveExitGate`. Decoder is plain CSA2; global context comes from the encoder's global KV only (Eq. 1).
- `src/memory_machines/models/blocks/parallel_hybrid.py` — `ParallelAttentionGDN2`, `ParallelAttentionTitan` (parallel attention ∥ memory mixer).
- `src/memory_machines/models/blocks/sparse.py` — `CompressedSparseAttention2` (CSA2 Full/Reindex/Reuse cycle).
- `src/memory_machines/models/blocks/gdn2.py` — `GatedDeltaNet2`, `GatedDeltaNet2Mixer`, `gated_delta_rule_2`.
- `src/memory_machines/models/blocks/ffn.py` — `SwiGLU`, `ClassicMLP`, `DenseFFN` (pre-norm residual).
- `src/memory_machines/models/io.py` — `ModelInput`, `ModelOutput`, `ModelState` (typed forward contracts).
- `src/memory_machines/training/engine.py` — `create_trainer` (imperative Trainer factory).
- `src/memory_machines/modules/lightning_module.py` — `LitResearchSystem` (PL shell; owns loss, step metrics, optimizer config only).
- `src/memory_machines/integrations/transformers/` — HF adapter (`CustomResearchConfig`, `CustomModelForCausalLM`).
- `src/memory_machines/ops/` — op dispatch, causal conv1d, attention references.
- `src/memory_machines/kernels/` — Taichi and native (torch) kernels.
- `scripts/train.py` — Hydra entry point (instantiate model, system, data, callbacks, trainer; `trainer.fit`).
- `scripts/eval.py` — load checkpoint, rebuild model from config, `trainer.test`.
- `scripts/export_hf.py` — export to HF format (infers architecture from checkpoint tensors).
- `notebooks/01_quickstart.py` — load package, build model, run one forward pass.
- `configs/` — Hydra config compose root (`default.yaml` + `model/`, `system/`, `data/`, `trainer/`, `callbacks/`).

## Additional Notes

- Default config values: `hidden_size=1024`, `intermediate_size=2816`, `num_layers=40`, `num_attention_heads=16`, `num_key_value_heads=4`, `vocab_size=32768`, `max_position_embeddings=32768`, `window_size=256`, `initializer_range=0.006`, `ced_num_encoder_layers=20`, `ced_num_decoder_layers=20`, `ced_attention_variant=compressed_sparse`, `ced_encoder_norm=dyT`, `ced_encoder_ffn=swiglu`, `ced_decoder_ffn=swiglu`.
- Preset sizes via `MemoryMachineConfig.from_pretrained_size("350M" | "700M" | "1.2B")`.
- BLT: byte vocab `blt_vocab_size` (0 = vocab_size), patch size `blt_patch_size` (default 4), local layers `blt_local_layers` (default 2), local window `blt_local_window_size` (default 512), n-gram sizes `blt_ngram_sizes` (default (3,4,5,6,7,8)), `blt_max_patch_bytes` (default 8).
- ModelInput fields: `input_ids`, `attention_mask`, `patch_ids` (BLT), `state`, `inputs_embeds`, `pixel_values`, `num_masked_tokens` (Perceiver-AR). ModelOutput fields: `logits`, `new_state`, `aux_loss`, `attn_weights`.
- CED decoder mode cycle (non-looped): `full` (layer 0), `reindex` (i % 4 == 1), `reuse` (rest). Encoder: layer 0 plain SWA, layer 1 `full`, rest `reuse`.
- SWA Bounded Replay: decoder prefill replays only the last `window_size` tokens when `swa_bounded_replay` is set; tail states stay out of the global compressed store (projected from encoder-final states, Eq. 1).
- Looped Language Model: `ced_decoder_looping=True` collapses repeated CSA2 -> DenseFFN into one parameter-shared layer applied `ced_decoder_loop_steps` times (default 16) with Q-exit via `AdaptiveExitGate`. Looped CSA2 stage per step: `full` at 0, `reindex` at steps 4/8/12/... (step % 4 == 0), `reuse` otherwise; TRM and plain LoopLM share the schedule.
- Perceiver-AR: encoder contextualizes appended mask latents; decoder queries = those masked latents, K/V = full encoder-final latent stream (SWA), compressed store = encoder global KV. `CausalDecoder.decode_full_context(..., kv_states=..., kv_position_ids=...)`; wired by `CheapLLM`/`BLTBackbone` masked paths. Backbones add `decoder_in_proj` (identity when symmetric) so asymmetric `ced_decoder_hidden_size` decoders receive downsampled queries and K/V. The masked count is free per forward pass: one table entry is shared for all masks; a shorter count takes the highest-index entries in order; a longer count interpolates between neighbouring entries (first entry anchors slot 0, last entry anchors the final slot).
- TRM mechanism (arXiv:2510.04871): `ced_decoder_trm=True` adds `[X, Y, Z]` three-stream input to the looped decoder.
