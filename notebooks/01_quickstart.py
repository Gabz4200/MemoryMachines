# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart
#
# Loads the package, builds a model, runs one forward pass.

# %%
import torch

from memory_machines.models.io import ModelInput
from memory_machines.models.registry import get_model
from memory_machines.ops.dispatch import dispatch_dw_scale

# %%
model = get_model("example", embed_dim=256, num_layers=2, vocab_size=1000)
# %%
x = torch.randn(4, 16, 256)
y = model(ModelInput(inputs_embeds=x))
print(y.logits.shape)
# %% [markdown]
# ### Primitive op via backend: "naive" (default, working) | "cpu" (compiled C++) | "gpu" (Taichi)

# %%
a = torch.randn(4, 16)
out = dispatch_dw_scale(a, torch.randn_like(a), backend="naive")
print(out.shape)
