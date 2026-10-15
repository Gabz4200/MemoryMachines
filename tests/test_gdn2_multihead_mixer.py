"""RED: config-driven MultiHead mixer wrapping GatedDeltaNet2.

Interface parity with the other token mixers (SelfReferentialTitanMixer):
  forward(hidden_states, attention_mask=None, past_key_values=None,
          use_cache=False, output_attentions=False, cu_seqlens=None,
          chunk_size=None) -> (out, None, cache|None)
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


def test_mixer_class_exists_and_builds_from_config():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False)
    assert mixer.hidden_size == 16
    assert mixer.num_heads == 2
    assert mixer.head_dim == 8


def test_mixer_forward_triple_and_shapes():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False).eval()
    x = torch.randn(1, 6, 16)
    out, weights, cache = mixer(x)
    assert tuple(out.shape) == (1, 6, 16)
    assert weights is None
    assert cache is None


def test_mixer_use_cache_and_continuation_matches_full():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 7, 16)
    with torch.no_grad():
        full, _, _ = mixer(x)
        _, _, state = mixer(x[:, :4], use_cache=True)
        part, _, _ = mixer(x[:, 4:], past_key_values=state)
    torch.testing.assert_close(full[:, 4:], part, atol=1e-5, rtol=1e-5)


def test_mixer_chunk_backend_matches_naive():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    chunked_mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="chunk", chunk_size=4).eval()
    naive_mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="naive").eval()
    naive_mixer.layer.load_state_dict(chunked_mixer.layer.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 9, 16)
    with torch.no_grad():
        chunked, _, _ = chunked_mixer(x)
        naive, _, _ = naive_mixer(x)
    torch.testing.assert_close(naive, chunked, atol=1e-5, rtol=1e-5)


def test_mixer_call_chunk_size_override():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False, backend="chunk", chunk_size=2).eval()
    assert mixer.chunk_size == 2
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        defaulted, _, _ = mixer(x)
        overridden, _, _ = mixer(x, chunk_size=6)
    # Same backend/math, different chunk widths: outputs agree.
    torch.testing.assert_close(defaulted, overridden, atol=1e-5, rtol=1e-5)


def test_mixer_parity_with_raw_gdn2_shared_weights():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2, GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False).eval()
    raw = GatedDeltaNet2(hidden_size=16, num_heads=2, head_dim_k=8, head_dim_v=8, use_short_conv=False, norm_eps=_cfg().rms_norm_eps).eval()
    raw.load_state_dict(mixer.layer.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out_mixer, _, _ = mixer(x)
        out_raw, _, _ = raw(x)
    torch.testing.assert_close(out_mixer, out_raw, atol=1e-6, rtol=1e-5)


def test_mixer_causality_no_future_leak():
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    torch.manual_seed(0)
    mixer = GatedDeltaNet2Mixer(_cfg(), use_short_conv=False).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        base, _, _ = mixer(x)
        x2 = x.clone()
        x2[0, 7] += 10.0
        out2, _, _ = mixer(x2)
    delta = (out2 - base).abs().amax(-1)[0]
    assert delta[:7].max().item() == 0.0


def _mixer(**kw):
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    args = {"use_short_conv": False}
    args.update(kw)
    torch.manual_seed(0)
    return GatedDeltaNet2Mixer(_cfg(), **args).eval()


def test_mixer_mask_scatters_packed_results():
    torch.manual_seed(1)
    x = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    mixer = _mixer()
    with torch.no_grad():
        out, weights, cache = mixer(x, attention_mask=mask)
    assert weights is None and cache is None
    assert out[0, 3:].abs().amax().item() == 0.0
    with torch.no_grad():
        ref, _, _ = mixer(x[0:1, :3])
    torch.testing.assert_close(out[0:1, :3], ref, atol=1e-5, rtol=1e-5)


def test_mixer_packed_matches_fresh_segments():
    torch.manual_seed(1)
    x = torch.randn(1, 7, 16)
    cu = torch.tensor([0, 3, 7])
    mixer = _mixer()
    with torch.no_grad():
        packed, _, _ = mixer(x, cu_seqlens=cu)
        seg0, _, _ = mixer(x[:, :3])
        seg1, _, _ = mixer(x[:, 3:])
    torch.testing.assert_close(packed[:, :3], seg0, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(packed[:, 3:], seg1, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        full, _, _ = mixer(x)
    assert (packed[:, 3:] - full[:, 3:]).abs().amax().item() > 1e-6


def test_mixer_packed_error_paths():
    import pytest

    mixer = _mixer()
    x = torch.randn(1, 4, 16)
    with pytest.raises(ValueError, match="batch 1"):
        mixer(torch.randn(2, 4, 16), cu_seqlens=torch.tensor([0, 4, 8]))
    with pytest.raises(ValueError, match="cu_seqlens must end"):
        mixer(x, cu_seqlens=torch.tensor([0, 2]))
    with pytest.raises(ValueError, match="use_cache"):
        mixer(x, cu_seqlens=torch.tensor([0, 4]), use_cache=True)
    with pytest.raises(ValueError, match="same time"):
        mixer(x, attention_mask=torch.ones(1, 4), cu_seqlens=torch.tensor([0, 4]))


def test_mixer_constructor_and_call_validation():
    import pytest

    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    with pytest.raises(ValueError, match="chunk_size"):
        GatedDeltaNet2Mixer(_cfg(), chunk_size=0)
    mixer = _mixer()
    with pytest.raises(ValueError, match=r"\[B, T, D="):
        mixer(torch.randn(1, 4, 15))
    with pytest.raises(ValueError, match="chunk_size"):
        mixer(torch.randn(1, 4, 16), chunk_size=0)


def test_mixer_reset_and_package_export():
    from memory_machines.models import blocks
    from memory_machines.models.blocks.gdn2 import GatedDeltaNet2Mixer

    assert blocks.GatedDeltaNet2Mixer is GatedDeltaNet2Mixer
    mixer = _mixer()
    before = [p.detach().clone() for p in mixer.parameters()]
    mixer.reset_parameters()
    after = [p.detach() for p in mixer.parameters()]
    assert any((b - a).abs().amax().item() > 0 for b, a in zip(before, after))
    torch.manual_seed(1)
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        out, weights, _ = mixer(x, output_attentions=True)
    assert tuple(out.shape) == (1, 3, 16) and weights is None
