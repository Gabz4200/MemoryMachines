# Notebooks

- `01_quickstart.py` is the source of truth (py:percent). Edit it, then sync:
  `uv run jupytext --sync notebooks/`
- `01_quickstart.ipynb` is generated; commit it if you upload to Kaggle.
- On Kaggle, install deps inline, matching the kernel's accelerator:

```ipython
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
!pip install omegaconf hydra-core pytorch-lightning transformers
```

Use `/whl/cpu` when the kernel has no GPU.
