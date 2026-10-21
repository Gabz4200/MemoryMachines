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
    CEDGlobalKVProjection,
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
    assert dec.loop_steps == 16


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


def test_plain_looped_decoder_skips_trm_and_runs_the_shared_layer() -> None:
    dec = CausalDecoder(_loop_cfg(ced_decoder_trm=False, ced_decoder_loop_steps=3)).eval()
    # The [X, Y, Z] mix and loop-local latent are TRM-only.
    assert not hasattr(dec, "trm_mix")
    assert not hasattr(dec, "z_proj")
    with _record_loop_modes(dec) as modes:
        out = dec.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H), exit_threshold=1.0)
    assert modes == ["full", "reuse", "reuse"]
    assert tuple(out.shape) == (1, 5, H)


def test_plain_looped_decoder_matches_a_manual_single_stream_replay() -> None:
    torch.manual_seed(0)
    dec = CausalDecoder(_loop_cfg(ced_decoder_trm=False, ced_decoder_loop_steps=3)).eval()
    gate = dec.gate
    assert gate is not None
    with torch.no_grad():
        torch.nn.init.constant_(gate.proj.bias, -20.0)  # lambda -> 0, no early exit
    h, enc = torch.randn(1, 5, H), torch.randn(1, 6, H)
    with torch.no_grad(), _record_loop_modes(dec) as modes:
        out = dec.decode_full_context(h, enc)
    assert modes == ["full", "reuse", "reuse"]
    # The plain path threads one running state through the shared layer + FFN,
    # seeded once from the encoder projection.
    layer = cast("CompressedSparseAttention2", dec.layers[0])
    ffn = cast("DenseFFN", dec.ffns[0])
    comp, ends = dec.global_projs[0](enc)
    shared: CSA2SharedState | None = CSA2SharedState(compressed_kv=comp, comp_end_pos=ends, total_len=enc.shape[1])
    x = h
    with torch.no_grad():
        for step in range(3):
            x, shared = layer.forward_shared(x, shared, seeded_store=True, mode=CausalDecoder._loop_mode(step))
            x = ffn(x, step=0)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)


def test_looped_decoder_reindexes_on_the_configured_cycle():
    dec = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=9, ced_decoder_loop_exit_threshold=1.0)).eval()
    with _record_loop_modes(dec) as modes:
        dec.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H))
    # Grouped layout: full first, then groups of 3 reuse opened by a reindex.
    assert modes == ["full", "reuse", "reuse", "reuse", "reindex", "reuse", "reuse", "reuse", "reindex"]


def test_looped_decoder_uses_full_kv_for_masked_queries():
    torch.manual_seed(0)
    dec = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=6, ced_decoder_loop_exit_threshold=1.0)).eval()
    _amp_signal(dec)
    N, L = 6, 2
    enc = torch.randn(1, N, H)
    queries = torch.randn(1, L, H)
    kv_a = torch.randn(1, N, H)
    kv_b = kv_a.clone()
    kv_b[0, 0] += 1e3  # an earlier KV token the masked queries can see
    q_pos = torch.arange(N - L, N).unsqueeze(0)
    kv_pos = torch.arange(N).unsqueeze(0)
    out_a = dec.decode_full_context(queries, enc, q_pos, start_pos=N - L, exit_threshold=1.0, kv_states=kv_a, kv_position_ids=kv_pos)
    out_b = dec.decode_full_context(queries, enc, q_pos, start_pos=N - L, exit_threshold=1.0, kv_states=kv_b, kv_position_ids=kv_pos)
    assert (out_a - out_b).abs().max().item() > 1e-4


def test_looped_decoder_kv_stream_stays_causal():
    torch.manual_seed(0)
    dec = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=5, ced_decoder_loop_exit_threshold=1.0)).eval()
    _amp_signal(dec)
    N = 6
    enc = torch.randn(1, N, H)
    query = torch.randn(1, 1, H)
    kv_a = torch.randn(1, N, H)
    kv_b = kv_a.clone()
    kv_b[0, 4] += 10.0  # future relative to the query at position 2
    q_pos = torch.tensor([[2]])
    kv_pos = torch.arange(N).unsqueeze(0)
    out_a = dec.decode_full_context(query, enc, q_pos, start_pos=2, exit_threshold=1.0, kv_states=kv_a, kv_position_ids=kv_pos)
    out_b = dec.decode_full_context(query, enc, q_pos, start_pos=2, exit_threshold=1.0, kv_states=kv_b, kv_position_ids=kv_pos)
    torch.testing.assert_close(out_a, out_b, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(6,), (2, 6, H), (1, 6, H + 1)], ids=["rank2", "batch", "width"])
def test_decoder_rejects_a_malformed_kv_stream(shape: tuple[int, ...]) -> None:
    dec = CausalDecoder(_loop_cfg()).eval()
    with pytest.raises(ValueError, match="kv_states must be"):
        dec.decode_full_context(torch.randn(1, 2, H), torch.randn(1, 6, H), torch.arange(2).unsqueeze(0), kv_states=torch.randn(*shape))


def test_stack_forwards_the_kv_stream_to_the_decoder() -> None:
    torch.manual_seed(0)
    stack = CausalEncoderDecoderStack(_loop_cfg(ced_decoder_loop_steps=5, ced_decoder_loop_exit_threshold=1.0)).eval()
    h, enc = torch.randn(1, 2, H), torch.randn(1, 6, H)
    kv = torch.randn(1, 6, H)
    kv_pos = torch.arange(6).unsqueeze(0)
    via_stack = stack.decode_full_context(h, enc, exit_threshold=1.0, kv_states=kv, kv_position_ids=kv_pos)
    via_decoder = stack.decoder.decode_full_context(h, enc, exit_threshold=1.0, kv_states=kv, kv_position_ids=kv_pos)
    torch.testing.assert_close(via_stack, via_decoder)


def test_looped_decoder_passes_the_kv_stream_to_every_iteration():
    dec = CausalDecoder(_loop_cfg(ced_decoder_loop_steps=5, ced_decoder_loop_exit_threshold=1.0)).eval()
    layer = dec.layers[0]
    orig = cast("Callable[..., tuple[torch.Tensor, CSA2SharedState]]", layer.forward_shared)
    seen: list[tuple[int, int]] = []

    def spy(*args, **kwargs):
        seen.append((args[0].shape[1], kwargs["kv_states"].shape[1]))
        return orig(*args, **kwargs)

    object.__setattr__(layer, "forward_shared", spy)
    try:
        dec.decode_full_context(
            torch.randn(1, 2, H),
            torch.randn(1, 6, H),
            torch.arange(4, 6).unsqueeze(0),
            start_pos=4,
            exit_threshold=1.0,
            kv_states=torch.randn(1, 6, H),
            kv_position_ids=torch.arange(6).unsqueeze(0),
        )
    finally:
        object.__delattr__(layer, "forward_shared")
    assert seen == [(2, 6)] * 5


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
        gate = dec.gate
        assert gate is not None
        torch.nn.init.constant_(gate.proj.bias, 20.0)  # lambda -> 1
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
        gate = dec.gate
        assert gate is not None
        torch.nn.init.constant_(gate.proj.bias, -20.0)  # lambda -> 0
    with _record_loop_modes(dec) as modes:
        dec.decode_full_context(torch.randn(1, 5, H), torch.randn(1, 6, H))
    assert modes == ["full", "reuse", "reuse"]


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


def test_exit_steps_stress_never_exceeds_recurrent_depth():
    # R4 regression: exit_steps is (cdf < threshold).sum(dim=1).clamp(max=T-1) + 1,
    # so the result is pinned to [1, T] even under numerical drift in the
    # cumprod/cumsum chain. The invariant this relies on is that the final
    # step always exits (exit_distribution assigns the survival mass to it,
    # so cdf[:, -1] == 1 exactly and no threshold in [0, 1] can push the count
    # past T). Stress-test with random lambdas across the whole range.
    torch.manual_seed(0)
    for _ in range(200):
        T = int(torch.randint(1, 9, (1,)).item())
        lam = torch.rand(2, T, 3)  # in (0, 1): never exactly 0 or 1
        steps = AdaptiveExitGate.exit_steps(lam, threshold=0.5)
        assert tuple(steps.shape) == (2, 3)
        assert bool((steps >= 1).all()) and bool((steps <= T).all())
        # The final step's CDF is exactly 1, so a threshold of 1.0 always exits
        # at the last step (the clamp pins the count to T-1, +1 -> T) when
        # some earlier step is still below threshold; when every step exits
        # immediately (lambda == 1) the count is 0 and the result is 1.
        q1 = AdaptiveExitGate.exit_steps(lam, 1.0).tolist()
        assert all(1 <= v <= T for row in q1 for v in row)


def test_exit_steps_threshold_one_forces_the_final_step_even_when_all_exit_early():
    # Every step exits with probability 1 (lambda == 1), so the CDF reaches 1
    # at step 1 and the position exits immediately. The clamp must still pin
    # the result to [1, T] — it never reaches T+1 even though the count of
    # steps below threshold is 0 (clamped to 0, +1 -> 1).
    lam = torch.ones(1, 5, 2)
    assert AdaptiveExitGate.exit_steps(lam, 1.0).tolist() == [[1, 1]]


# 6. decoder construction and decode-entry guards -----------------------------
def test_decoder_rejects_an_ffn_count_mismatch() -> None:
    cfg = replace(tiny_cfg(), ced_num_decoder_layers=1)
    base = CausalDecoder(cfg)
    with pytest.raises(ValueError, match=r"ffns \(2\) must match"):
        CausalDecoder(cfg, layers=[base.layers[0]], ffns=[base.ffns[0], base.ffns[0]])


def test_global_kv_projection_rejects_indivisible_decoder_width() -> None:
    with pytest.raises(ValueError, match="divisible by num_attention_heads"):
        CEDGlobalKVProjection(tiny_cfg(), decoder_hidden_size=tiny_cfg().hidden_size - 2)


def test_decode_full_context_rejects_a_2d_encoder_stream() -> None:
    dec = CausalDecoder(replace(tiny_cfg(), ced_num_decoder_layers=1)).eval()
    with pytest.raises(ValueError, match="encoder_final must be 3D"):
        dec.decode_full_context(torch.randn(1, 2, H), torch.randn(6, H))


def test_zero_layer_decoder_returns_its_input_unchanged() -> None:
    cfg = replace(tiny_cfg(), ced_num_decoder_layers=1)
    dec = CausalDecoder(cfg, layers=[], global_projs=[], ffns=[]).eval()
    h = torch.randn(1, 4, H)
    with torch.no_grad():
        out = dec.decode_full_context(h, torch.randn(1, 6, H))
    torch.testing.assert_close(out, h, atol=0, rtol=0)
