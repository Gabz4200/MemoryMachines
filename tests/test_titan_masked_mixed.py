"""Mixed-batch padding regression tests for SelfReferentialTitanMixer.

A batch mixing fully-masked (empty) rows with live rows must not crash: empty
rows come back as zeros with the full sequence width, matching the GDN2 masked
convention. Fully-empty batches keep the legacy collapse to ``(B, 0, D)``.
"""

import torch

from memory_machines.utils.config_schema import MemoryMachineConfig


def _cfg(**kw):
    base = {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "csa_num_output_groups": 2,
    }
    base.update(kw)
    return MemoryMachineConfig(**base)


def _mixer(**kw):
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    args = {"memory_depth": 1, "use_short_conv": False}
    args.update(kw)
    torch.manual_seed(0)
    return SelfReferentialTitanMixer(_cfg(), **args).eval()


def test_masked_mixed_batch_empty_row_is_zeros():
    mixer = _mixer()
    torch.manual_seed(1)
    x = torch.randn(2, 4, 16)
    mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    with torch.no_grad():
        out, weights, cache = mixer(x, attention_mask=mask)
        ref, _, _ = mixer(x[1:2])
    assert tuple(out.shape) == (2, 4, 16)
    assert weights is None and cache is None
    assert out[0].abs().amax().item() == 0.0
    torch.testing.assert_close(out[1:2], ref, atol=1e-6, rtol=1e-5)


def test_masked_all_empty_batch_collapses():
    mixer = _mixer()
    torch.manual_seed(1)
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        out, _, _ = mixer(x, attention_mask=torch.zeros(2, 4))
    assert tuple(out.shape) == (2, 0, 16)
    assert out.numel() == 0
