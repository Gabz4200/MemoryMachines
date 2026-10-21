"""RED: config-driven MultiHead mixer wrapping SelfReferentialTitanMemory.

Interface parity with GatedDeltaNet2 (the other token mixers):
  forward(hidden_states, attention_mask=None, past_key_values=None,
          use_cache=False, output_attentions=False, cu_seqlens=None,
          chunk_size=None) -> (out, None, cache|None)
"""

import pytest
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
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    mixer = SelfReferentialTitanMixer(_cfg())
    assert mixer.hidden_size == 16
    assert mixer.num_heads == 2
    assert mixer.head_dim == 8


def test_mixer_forward_triple_and_shapes():
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    torch.manual_seed(0)
    mixer = SelfReferentialTitanMixer(_cfg()).eval()
    x = torch.randn(1, 6, 16)
    out, weights, cache = mixer(x)
    assert tuple(out.shape) == (1, 6, 16)
    assert weights is None
    assert cache is None


def test_mixer_use_cache_and_continuation_matches_full():
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    torch.manual_seed(0)
    mixer = SelfReferentialTitanMixer(_cfg(), memory_depth=1, use_short_conv=False).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 7, 16)
    with torch.no_grad():
        full, _, _ = mixer(x)
        _, _, state = mixer(x[:, :4], use_cache=True)
        part, _, _ = mixer(x[:, 4:], past_key_values=state)
    torch.testing.assert_close(full[:, 4:], part, atol=1e-5, rtol=1e-5)


def test_mixer_chunked_equals_sequential():
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    torch.manual_seed(0)
    mixer = SelfReferentialTitanMixer(_cfg(), memory_depth=1, use_short_conv=False).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 9, 16)
    with torch.no_grad():
        full, _, _ = mixer(x)
        chunked, _, _ = mixer(x, chunk_size=4)
    torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-5)


def test_mixer_parity_with_raw_selfref_shared_weights():
    from memory_machines.models.blocks.fast_weights import (
        SelfReferentialTitanMemory,
        SelfReferentialTitanMixer,
    )

    torch.manual_seed(0)
    mixer = SelfReferentialTitanMixer(_cfg(), memory_depth=1, use_short_conv=False).eval()
    raw = SelfReferentialTitanMemory(hidden_size=16, num_heads=2, memory_depth=1, use_short_conv=False).eval()
    # Copy mixer inner weights into raw module exactly.
    raw.load_state_dict(mixer.memory.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out_mixer, _, _ = mixer(x)
        out_raw, _ = raw(x)
    torch.testing.assert_close(out_mixer, out_raw, atol=1e-6, rtol=1e-5)


def test_mixer_causality_no_future_leak():
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    torch.manual_seed(0)
    mixer = SelfReferentialTitanMixer(_cfg(), memory_depth=1, use_short_conv=False).eval()
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
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    args = {"memory_depth": 1, "use_short_conv": False}
    args.update(kw)
    torch.manual_seed(0)
    return SelfReferentialTitanMixer(_cfg(), **args).eval()


def test_mixer_mask_scatters_packed_results():
    torch.manual_seed(1)
    x = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    mixer = _mixer()
    with torch.no_grad():
        out, weights, cache = mixer(x, attention_mask=mask)
    assert weights is None and cache is None
    # Padded slots stay zero.
    assert out[0, 3:].abs().amax().item() == 0.0
    # Valid slots equal a dense run over the packed prefix.
    with torch.no_grad():
        ref, _, _ = mixer(x[0:1, :3])
    torch.testing.assert_close(out[0:1, :3], ref, atol=1e-5, rtol=1e-5)


def test_mixer_mask_all_padded_row_is_zeros():
    mixer = _mixer()
    torch.manual_seed(1)
    x = torch.randn(1, 4, 16)
    mask = torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        out, _, _ = mixer(x, attention_mask=mask)
    # Fully-padded rows return full-width zeros (canonical masked convention).
    assert tuple(out.shape) == (1, 4, 16)
    assert out.abs().amax().item() == 0.0


def test_mixer_mask_error_paths():
    import pytest

    mixer = _mixer()
    x = torch.randn(1, 4, 16)
    mask = torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        _, _, state = mixer(x, use_cache=True)
    with pytest.raises(ValueError, match="past_key_values"):
        mixer(x, attention_mask=mask, past_key_values=state)
    with pytest.raises(ValueError, match="use_cache"):
        mixer(x, attention_mask=mask, use_cache=True)
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        mixer(x, attention_mask=torch.ones(1, 3, dtype=torch.long))


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
    # Packed segments start fresh, so they differ from one continuous run.
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

    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    with pytest.raises(ValueError, match="chunk_size"):
        SelfReferentialTitanMixer(_cfg(), chunk_size=0)
    mixer = _mixer()
    with pytest.raises(ValueError, match=r"\[B, T, D="):
        mixer(torch.randn(1, 4, 15))
    with pytest.raises(ValueError, match="chunk_size"):
        mixer(torch.randn(1, 4, 16), chunk_size=0)


def test_mixer_short_conv_cache_continuation():
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    torch.manual_seed(0)
    mixer = SelfReferentialTitanMixer(_cfg(), memory_depth=1, use_short_conv=True).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 7, 16)
    with torch.no_grad():
        full, _, _ = mixer(x)
        _, _, state = mixer(x[:, :4], use_cache=True)
        assert state is not None and state["conv_state"] is not None
        part, _, _ = mixer(x[:, 4:], past_key_values=state)
    torch.testing.assert_close(full[:, 4:], part, atol=1e-5, rtol=1e-5)


def test_mixer_deep_continuation_and_chunk_parity():
    mixer = _mixer(memory_depth=2)
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        full, _, _ = mixer(x)
        chunked, _, _ = mixer(x, chunk_size=2)
        _, _, state = mixer(x[:, :2], use_cache=True)
        part, _, _ = mixer(x[:, 2:], past_key_values=state)
    torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(full[:, 2:], part, atol=1e-5, rtol=1e-5)


def test_mixer_dot_objective_parity_with_raw():
    from memory_machines.models.blocks.fast_weights import (
        SelfReferentialTitanMemory,
        SelfReferentialTitanMixer,
    )

    torch.manual_seed(0)
    mixer = SelfReferentialTitanMixer(_cfg(), memory_depth=1, objective="dot", use_short_conv=False).eval()
    raw = SelfReferentialTitanMemory(hidden_size=16, num_heads=2, memory_depth=1, objective="dot", use_short_conv=False).eval()
    raw.load_state_dict(mixer.memory.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out_mixer, _, _ = mixer(x)
        out_raw, _ = raw(x)
    torch.testing.assert_close(out_mixer, out_raw, atol=1e-6, rtol=1e-5)


def test_mixer_ctor_chunk_default_and_override():
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    torch.manual_seed(0)
    mixer = SelfReferentialTitanMixer(_cfg(), memory_depth=1, use_short_conv=False, chunk_size=2).eval()
    assert mixer.chunk_size == 2
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        defaulted, _, _ = mixer(x)
        full, _, _ = mixer(x, chunk_size=6)
    torch.testing.assert_close(defaulted, full, atol=1e-5, rtol=1e-5)
    # Unset per-call chunk resolves from config: nonzero is used verbatim, 0 means whole sequence.
    assert SelfReferentialTitanMixer(_cfg(titan_chunk_size=3), memory_depth=1, use_short_conv=False).chunk_size == 3
    assert SelfReferentialTitanMixer(_cfg(titan_chunk_size=0), memory_depth=1, use_short_conv=False).chunk_size is None


def test_mixer_reset_fast_init_and_package_export():
    from memory_machines.models import blocks
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    assert blocks.SelfReferentialTitanMixer is SelfReferentialTitanMixer
    mixer = _mixer()
    assert tuple(mixer.fast_init.keys()) == ("k", "v", "q", "eta", "alpha", "memory")
    before = [p.detach().clone() for p in mixer.memory.parameters()]
    mixer.reset_parameters()
    after = [p.detach() for p in mixer.memory.parameters()]
    assert any((b - a).abs().amax().item() > 0 for b, a in zip(before, after))
    torch.manual_seed(1)
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        out, weights, _ = mixer(x, output_attentions=True)
    assert tuple(out.shape) == (1, 3, 16) and weights is None


@pytest.mark.parametrize("objective", ["l2", "dot"], ids=["l2", "dot"])
@pytest.mark.parametrize("memory_depth", [1, 2], ids=["depth_1", "depth_2"])
def test_mixer_chunk_backend_matches_naive_at_chunk_size_one(memory_depth: int, objective: str) -> None:
    """TNT chunk path with one-token chunks equals the sequential backend (depth-1 closed form and deep autograd arms)."""
    from memory_machines.models.blocks.fast_weights import (
        HOPE_BOXES,
        SelfReferentialTitanMixer,
    )

    torch.manual_seed(0)
    naive = SelfReferentialTitanMixer(_cfg(titan_backend="naive"), memory_depth=memory_depth, objective=objective, use_short_conv=False).eval()
    torch.manual_seed(0)
    chunk = SelfReferentialTitanMixer(_cfg(titan_backend="chunk"), memory_depth=memory_depth, objective=objective, use_short_conv=False, chunk_size=1).eval()
    chunk.load_state_dict(naive.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 7, 16)
    with torch.no_grad():
        out_naive, _, state_naive = naive(x, use_cache=True)
        out_chunk, _, state_chunk = chunk(x, use_cache=True)
    assert state_naive is not None and state_chunk is not None
    torch.testing.assert_close(out_chunk, out_naive, atol=1e-6, rtol=1e-5)
    for box in HOPE_BOXES:
        for got, ref in zip(state_chunk["fast_weights"][box], state_naive["fast_weights"][box]):
            torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-5)


def test_mixer_qk_projection_changes_retrieval_and_stays_causal() -> None:
    """TNT query projection (arXiv:2511.07343 Sec. 4.1.2) rewrites queries yet keeps strict causality."""
    from memory_machines.models.blocks.fast_weights import SelfReferentialTitanMixer

    torch.manual_seed(0)
    proj = SelfReferentialTitanMixer(_cfg(titan_backend="chunk", titan_use_qk_projection=True), memory_depth=1, use_short_conv=False, chunk_size=1).eval()
    torch.manual_seed(0)
    plain = SelfReferentialTitanMixer(_cfg(titan_backend="chunk"), memory_depth=1, use_short_conv=False, chunk_size=1).eval()
    plain.load_state_dict(proj.state_dict())
    torch.manual_seed(1)
    x = torch.randn(1, 6, 16)
    with torch.no_grad():
        out_proj, _, _ = proj(x)
        out_plain, _, _ = plain(x)
        x2 = x.clone()
        x2[0, 5] += 10.0
        out_perturbed, _, _ = proj(x2)
    assert (out_proj - out_plain).abs().amax().item() > 1e-5
    delta = (out_perturbed - out_proj).abs().amax(-1)[0]
    assert delta[:5].max().item() == 0.0
    assert delta[5].item() > 0.0


def test_mixer_packed_empty_segment_passes_through() -> None:
    """A zero-length packed segment contributes no tokens and does not disturb its neighbours."""
    mixer = _mixer()
    torch.manual_seed(1)
    x = torch.randn(1, 7, 16)
    cu = torch.tensor([0, 3, 3, 7])
    with torch.no_grad():
        packed, _, _ = mixer(x, cu_seqlens=cu)
        first, _, _ = mixer(x[:, :3])
        last, _, _ = mixer(x[:, 3:])
    assert tuple(packed.shape) == (1, 7, 16)
    torch.testing.assert_close(packed[:, :3], first, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(packed[:, 3:], last, atol=1e-6, rtol=1e-5)


def test_mixer_public_head_dims_and_backend() -> None:
    mixer = _mixer(backend="chunk")
    assert mixer.backend == "chunk"
    assert mixer.head_dim_k == mixer.head_dim_v == mixer.head_dim == 8
    assert mixer.memory.head_k_dim == mixer.memory.head_v_dim == 8
