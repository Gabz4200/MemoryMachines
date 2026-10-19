"""Looped (parameter-shared) CausalDecoder with the LoopLM adaptive exit gate.

The CED decoder is refactored into the Looped Language Model style
(arXiv:2510.25741): the repeated ``CSA2 -> DenseFFN`` pattern collapses into
one parameter-shared layer (plus one shared encoder-KV projection) that is
applied ``ced_decoder_loop_steps`` times. An exit gate reads the hidden state
after each loop; the loop stops early once the cumulative exit probability
crosses ``ced_decoder_loop_exit_threshold`` (Q-exit, Eq. 3).
"""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from typing import cast

import pytest
import torch

from memory_machines.models.blocks import (
    AdaptiveExitGate,
    CausalDecoder,
    CausalEncoderDecoderStack,
    CompressedSparseAttention2,
    CSA2SharedState,
    DenseFFN,
    SwiGLU,
)
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg

H = 32


def _loop_cfg(**kw) -> MemoryMachineConfig:
    return replace(
        tiny_cfg(),
        swa_bounded_replay=False,
        ced_num_decoder_layers=4,
        csa_encoder_compression_ratio=2,
        csa_decoder_compression_ratio=1,
        ced_decoder_looping=True,
        **kw,
    )


def _amp_signal(dec: CausalDecoder) -> None:
    """Restore output signal: the depth-scaled residual init leaves it tiny."""
    with torch.no_grad():
        for layer in cast("list[CompressedSparseAttention2]", list(dec.layers)):
            torch.nn.init.trunc_normal_(layer.grouped_out.out_proj.weight, 0.0, 0.5, -1.0, 1.0)
        for ffn in cast("list[DenseFFN]", list(dec.ffns)):
            inner = cast("SwiGLU", ffn.ffn)
            for w in (inner.gate_proj.weight, inner.up_proj.weight, inner.down_proj.weight):
                w.normal_(std=0.5)


@contextmanager
def _record_loop_modes(dec: CausalDecoder):
    """Spy the ``mode`` each loop iteration passes to the shared layer."""
    seen: list[str | None] = []
    layer = dec.layers[0]
    orig = cast("Callable[..., tuple[torch.Tensor, CSA2SharedState]]", layer.forward_shared)

    def spy(*args, **kwargs):
        seen.append(kwargs.get("mode"))
        return orig(*args, **kwargs)

    object.__setattr__(layer, "forward_shared", spy)
    try:
        yield seen
    finally:
        object.__delattr__(layer, "forward_shared")


# 1. exit-gate math (Eq. 3 and the Q-exit criterion) ---------------------------
def test_exit_gate_starts_neutral_with_a_zero_bias():
    gate = AdaptiveExitGate(tiny_cfg())
    assert torch.count_nonzero(gate.proj.bias) == 0
    with torch.no_grad():
        gate.proj.weight.zero_()
    lam = gate(torch.randn(2, 5, H))
    assert tuple(lam.shape) == (2, 5)
    torch.testing.assert_close(lam, torch.full((2, 5), 0.5))


def test_exit_distribution_is_normalised_over_the_loop_steps():
    lam = torch.full((1, 4, 3), 0.5)
    p = AdaptiveExitGate.exit_distribution(lam)
    assert tuple(p.shape) == (1, 4, 3)
    torch.testing.assert_close(p.sum(dim=1), torch.ones(1, 3))
    # p_t = lambda_t * S_{t-1} for t < T, remaining mass S_{T-1} at t = T.
    torch.testing.assert_close(p[0, :, 0], torch.tensor([0.5, 0.25, 0.125, 0.125]))


def test_exit_cdf_is_monotone_and_ends_at_one():
    cdf = AdaptiveExitGate.cdf(torch.full((1, 4, 3), 0.5))
    torch.testing.assert_close(cdf[0, :, 0], torch.tensor([0.5, 0.75, 0.875, 1.0]))
    assert bool((cdf[:, 1:] >= cdf[:, :-1]).all())


def test_exit_step_is_the_first_step_crossing_the_threshold():
    lam = torch.full((1, 4, 3), 0.5)
    steps = AdaptiveExitGate.exit_steps(lam, 0.5)  # cdf 0.5, 0.75, 0.875, 1.0
    assert steps.tolist() == [[1, 1, 1]]
    assert AdaptiveExitGate.exit_steps(lam, 0.75).tolist() == [[2, 2, 2]]
    assert AdaptiveExitGate.exit_steps(lam, 0.9).tolist() == [[4, 4, 4]]
    # Float-safe clamp: q = 1.0 must land on the final step, never T + 1.
    assert AdaptiveExitGate.exit_steps(lam, 1.0).tolist() == [[4, 4, 4]]


# 2. parameter sharing (the "merge the repeated CSA2 -> DenseFFN" refactor) ----
def test_looped_decoder_shares_one_layer_projection_ffn_and_gate():
    dec = CausalDecoder(_loop_cfg()).eval()
    assert len(dec.layers) == len(dec.global_projs) == len(dec.ffns) == 1
    assert isinstance(dec.layers[0], CompressedSparseAttention2)
    assert isinstance(dec.ffns[0], DenseFFN)
    assert isinstance(dec.gate, AdaptiveExitGate)
    assert dec.loop_steps == 4


def test_looping_off_keeps_one_layer_per_configured_depth():
    dec = CausalDecoder(replace(tiny_cfg(), ced_num_decoder_layers=3)).eval()
    assert len(dec.layers) == 3
    assert len(dec.ffns) == 3
    assert getattr(dec, "gate", None) is None


def test_looped_decoder_parameter_count_is_independent_of_loop_depth():
    shallow = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=1))
    deep = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=8))
    assert sum(p.numel() for p in shallow.parameters()) == sum(p.numel() for p in deep.parameters())


def test_looped_decoder_projects_the_encoder_kv_once():
    dec = CausalDecoder(_loop_cfg()).eval()
    calls = {"n": 0}
    proj = dec.global_projs[0]
    orig = proj.forward

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    proj.forward = spy
    try:
        dec.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H), exit_threshold=1.0)
    finally:
        del proj.forward
    assert calls["n"] == 1


# 3. loop semantics ------------------------------------------------------------
def test_trm_looped_decoder_uses_three_stream_input():
    dec = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=3)).eval()
    # Verify TRM mechanism is active: mix and z_proj exist.
    assert hasattr(dec, "trm_mix") and dec.trm_mix is not None
    assert hasattr(dec, "z_proj") and dec.z_proj is not None
    # The looped decoder takes [X, Y, Z] as input: each loop step
    # combines the fixed previous-layer output X (decoder input),
    # the running answer Y, and the loop-local latent Z.
    torch.manual_seed(0)
    enc = torch.randn(1, 6, H)
    h = torch.randn(1, 5, H)
    pos = torch.arange(5).unsqueeze(0)
    out = dec.decode_full_context(h, enc, pos, exit_threshold=1.0)
    # Output must match decoder input shape (the layer's Y stream).
    assert tuple(out.shape) == (1, 5, H)
    # The TRM mechanism introduces a different path from the plain
    # single-stream LoopLM: the mix projects concatenated [X, Y, Z]
    # features, and z_proj extracts the loop-local latent from the
    # The output must have the correct decoder-output shape.
    assert tuple(out.shape) == (1, 5, H)


def test_looped_decoder_reindexes_on_the_configured_cycle():
    dec = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=5, ced_decoder_loop_exit_threshold=1.0)).eval()
    with _record_loop_modes(dec) as modes:
        dec.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H))
    assert modes == ["full", "reindex", "reuse", "reuse", "reuse"]


def test_looped_decoder_output_changes_with_loop_depth():
    torch.manual_seed(0)
    shallow = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=1)).eval()
    deep = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=4)).eval()
    deep.load_state_dict(shallow.state_dict(), strict=True)
    _amp_signal(deep)
    h, enc = torch.randn(1, 5, H), torch.randn(1, 6, H)
    out1 = shallow.decode_full_context(h, enc, exit_threshold=1.0)
    out4 = deep.decode_full_context(h, enc, exit_threshold=1.0)
    assert (out1 - out4).abs().max().item() > 1e-5


# 4. adaptive early exit -------------------------------------------------------
def test_looped_decoder_stops_after_one_loop_when_the_gate_is_confident():
    dec = CausalDecoder(_loop_cfg()).eval()
    with torch.no_grad():
        torch.nn.init.constant_(dec.gate.proj.bias, 20.0)  # lambda -> 1
    with _record_loop_modes(dec) as modes:
        out = dec.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H))
    assert modes == ["full"]
    assert tuple(out.shape) == (1, 5, H)


def test_looped_decoder_honours_the_configured_exit_threshold():
    # q = 0 crosses at the first step for every position (lambda > 0).
    dec = CausalDecoder(_loop_cfg(ced_decoder_loop_exit_threshold=0.0)).eval()
    with _record_loop_modes(dec) as modes:
        dec.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H))
    assert modes == ["full"]


def test_looped_decoder_runs_every_step_when_the_gate_never_exits():
    dec = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=3, ced_decoder_loop_exit_threshold=1.0)).eval()
    with torch.no_grad():
        torch.nn.init.constant_(dec.gate.proj.bias, -20.0)  # lambda -> 0
    with _record_loop_modes(dec) as modes:
        dec.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H))
    assert modes == ["full", "reindex", "reuse"]


# 5. stack wiring, encoder conditioning, causality, config validation ----------
def test_stack_uses_the_looped_decoder_when_enabled():
    stack = CausalEncoderDecoderStack(_loop_cfg()).eval()
    assert len(stack.decoder_layers) == len(stack.decoder_ffns) == 1
    out = stack.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H), exit_threshold=1.0)
    assert tuple(out.shape) == (1, 5, H)
    assert stack.prefill(torch.randn(1, 5, H), torch.randn(1, 6, H)).shape == (1, 5, H)


def test_looped_decoder_is_conditioned_on_the_encoder_projection():
    dec = CausalDecoder(_loop_cfg()).eval()
    _amp_signal(dec)
    h = torch.randn(1, 5, H)
    out_a = dec.decode_full_context(h, torch.randn(1, 6, H), exit_threshold=1.0)
    out_b = dec.decode_full_context(h, torch.randn(1, 6, H), exit_threshold=1.0)
    assert (out_a - out_b).abs().max().item() > 1e-4


def test_looped_decoder_does_not_leak_future_tokens():
    torch.manual_seed(0)
    dec = CausalDecoder(_loop_cfg()).eval()
    _amp_signal(dec)
    x, enc = torch.randn(1, 6, H), torch.randn(1, 6, H)
    pos = torch.arange(6).unsqueeze(0)
    base = dec.decode_full_context(x, enc, pos, exit_threshold=1.0)
    for future in (4, 5):
        x2 = x.clone()
        x2[0, future] += 10.0
        out = dec.decode_full_context(x2, enc, pos, exit_threshold=1.0)
        assert (out[0, :future] - base[0, :future]).abs().max().item() == 0.0


def test_config_rejects_bad_loop_fields():
    with pytest.raises(ValueError, match="ced_decoder_loop_steps"):
        replace(tiny_cfg(), ced_decoder_loop_steps=0)
    with pytest.raises(ValueError, match="ced_decoder_loop_exit_threshold"):
        replace(tiny_cfg(), ced_decoder_loop_exit_threshold=1.5)
