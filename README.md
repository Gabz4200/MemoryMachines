# Memory Machines

Research project combining my previous experiments into a cheap LLM.

## Installation

PyTorch resolves to CPU wheels by default, if you need GPU you need to manually set the index.

Default Installation:

```bash
uv sync --group dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### NVIDIA GPU (CUDA cu128)

On a CUDA machine, override the index — no file edits, no extras:

```bash
uv sync --group dev \
        --no-sources-package torch --no-sources-package torchvision \
        --no-sources-package torchaudio \
        --index https://download.pytorch.org/whl/cu128
```

Equivalent env form for a shell:

```bash
UV_NO_SOURCES_PACKAGE="torch torchvision torchaudio" \
UV_INDEX=https://download.pytorch.org/whl/cu128 uv sync --group dev
```

List all three torch packages. A partial list leaves the rest on CPU wheels and produces a mixed build.
Use `--index`, not `--default-index`, which replaces PyPI and breaks non-torch resolution.
Note that `UV_INDEX=<url>` above works because it is a _priority index_;
enabling the declared index by name (`UV_INDEX_pytorch-cu128=1`) or `--extra-index-url` does NOT,
because `[tool.uv.sources]` takes precedence over index selection.

After syncing on a GPU machine, run `uv lock` once before committing,
so the committed lock is the marker-split form rather than a single all-platform entry.

### End users / library consumers

```bash
pip install project # or: uv add project
```

torch is a normal dependency, so the standard install works with no flags and the
consumer's own package manager resolves the right wheel for their hardware. For a
one-off manual install, uv can also pick the backend itself:

```bash
uv pip install --torch-backend=auto torch
```

## Verify

```python
import torch

print(torch.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Usage

```bash
uv run python scripts/train.py trainer.fast_dev_run=true # 1 train + 1 val step
uv run python scripts/train.py # full run
uv run python scripts/eval.py checkpoint=path/to/ckpt.ckpt
```

## License

MIT
