"""Regression tests for cross-layer position/SWA-state bugs in CSA2 stacks.

Each test pins one invariant of the CED/CSA2 design (DeepSeek-V4.1-Flash,
arXiv:2609.19969): layers share global KV and Top-K indices but each layer
keeps its own Q and SWA KV (Sec. 2.3.1, Fig. 4), and implicit positions
count up from the *sequence* past, not one past per layer.
"""

from dataclasses import replace

import torch

import memory_machines.models.blocks.sparse as sparse_mod
from memory_machines.models.blocks import CausalEncoderDecoderStack, CompressedSparseAttention2
from memory_machines.models.blocks.memory import EngramMemory
from memory_machines.models.blocks.shared import CSA2SharedState
from memory_machines.utils.config_schema import MemoryMachineConfig
from tests.test_blocks_shapes import tiny_cfg


def _stack_cfg() -> MemoryMachineConfig:
    return replace(
        tiny_cfg(),
        swa_window_size=4,
        ced_num_encoder_layers=2,
        ced_num_decoder_layers=3,
        csa_encoder_compression_ratio=2,
        csa_decoder_compression_ratio=1,
    )


def test_csa2_positions_do_not_advance_between_layers():
    """Implicit positions must be identical for every layer of a stack.

    With position_ids=None, every layer of an encode/decode stack must
    resolve query positions from the sequence start (plus the stack's
    own past), not from a past inflated by earlier layers.
    """
    torch.manual_seed(0)
    cfg = _stack_cfg()
    layer0 = CompressedSparseAttention2(cfg, mode="full")
    layer1 = CompressedSparseAttention2(cfg, mode="reuse")

    seen: list[tuple[int, list[int]]] = []

    orig = sparse_mod._resolve_positions

    def spy(hidden_states, q_len, past_len, position_ids):
        q_pos, k_pos = orig(hidden_states, q_len, past_len, position_ids)
        seen.append((past_len, q_pos[:, -1].tolist()))
        return q_pos, k_pos

    sparse_mod._resolve_positions = spy
    try:
        x = torch.randn(1, 8, cfg.hidden_size)
        shared = CSA2SharedState()
        layer0.mode = "full"
        layer0.forward_shared(x, shared, use_cache=True)
        layer1.mode = "reuse"
        layer1.forward_shared(x, shared, use_cache=True)
    finally:
        sparse_mod._resolve_positions = orig

    assert len(seen) == 2
    for past_len, last_q in seen:
        assert past_len == 0, f"layer saw past_len={past_len}; expected 0 (shared sequence past, not per-layer)"
        assert last_q == [7], f"last q_pos={last_q}; expected [7] (positions 0..7 for both layers)"


def test_csa2_swa_state_is_layer_local():
    """Each layer's SWA KV must not leak into the next layer's window.

    CSA2 shares global KV across layers but every layer keeps its own
    SWA KV (Fig. 4). Threading swa_k/swa_v/swa_pos through the shared
    state makes layer 1 attend to layer 0's window keys.
    """
    torch.manual_seed(0)
    cfg = _stack_cfg()
    layer0 = CompressedSparseAttention2(cfg, mode="full")
    layer1 = CompressedSparseAttention2(cfg, mode="reuse")

    x = torch.randn(1, 8, cfg.hidden_size)
    shared = CSA2SharedState()
    layer0.mode = "full"
    _, shared = layer0.forward_shared(x, shared, use_cache=True)

    # After layer 0, the shared state carries global KV +
    # indices but no SWA cache: SWA is per-layer.
    assert shared.compressed_kv is not None
    assert shared.topk_indices is not None
    assert not hasattr(shared, "swa_k"), "SWA state must not live in the shared state"

    layer1.mode = "reuse"
    out1, shared = layer1.forward_shared(x, shared, use_cache=True)
    assert not hasattr(shared, "swa_k")
    assert tuple(out1.shape) == (1, 8, cfg.hidden_size)


def test_ced_stack_positions_stay_correct_across_layers():
    """End-to-end: every CED layer resolves the same implicit positions."""
    torch.manual_seed(0)
    cfg = _stack_cfg()
    stack = CausalEncoderDecoderStack(cfg).eval()

    seen: list[list[int]] = []
    orig = sparse_mod._resolve_positions

    def spy(hidden_states, q_len, past_len, position_ids):
        q_pos, k_pos = orig(hidden_states, q_len, past_len, position_ids)
        seen.append(q_pos[:, -1].tolist())
        return q_pos, k_pos

    sparse_mod._resolve_positions = spy
    try:
        enc = torch.randn(1, 8, cfg.hidden_size)
        stack.encode(enc)
        dec = torch.randn(1, 6, cfg.hidden_size)
        enc_final = torch.randn(1, 8, cfg.hidden_size)
        stack.decode_full_context(dec, enc_final)
    finally:
        sparse_mod._resolve_positions = orig

    # 2 encoder layers: all see positions ending at 7.
    assert seen[:2] == [[7], [7]], f"encoder positions {seen[:2]}; expected both [7]"
    # 3 decoder layers: all see positions ending at 13 (8 + 6 - 1).
    assert seen[2:] == [[13], [13], [13]], f"decoder positions {seen[2:]}; expected all [13]"


def test_engram_hash_is_dtype_invariant():
    """The multiplicative-XOR hash must not depend on the input dtype.

    int32 input_ids must produce the same table ids (and forward
    output) as int64 for the same tokens.
    """
    torch.manual_seed(0)
    cfg = replace(
        tiny_cfg(),
        engram_num_entries=1024,
        engram_orders=(3, 4),
        engram_hash_heads=2,
    )
    mem = EngramMemory(cfg).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    h64 = mem._ngram_hashes(ids.to(torch.int64), layer_id=0)
    h32 = mem._ngram_hashes(ids.to(torch.int32), layer_id=0)
    torch.testing.assert_close(h64, h32)
    # Forward path agrees for both dtypes.
    hidden = torch.randn(1, 5, cfg.hidden_size)
    out64 = mem(ids, hidden)
    out32 = mem(ids.to(torch.int32), hidden)
    torch.testing.assert_close(out64, out32)
