"""``MemoryMachineConfig`` boot validation: one case per ``__post_init__`` guard.

Every case flips a single invariant on top of the default config and matches
the message fragment naming the offending field, so a case cannot pass by
tripping an unrelated earlier guard. Boundary cases prove the guards are not
over-broad.
"""

from __future__ import annotations

from typing import Any

import pytest

from memory_machines.utils.config_schema import MemoryMachineConfig


def make_cfg(**overrides: Any) -> MemoryMachineConfig:
    """Build a config from the validated defaults plus ``overrides``."""
    return MemoryMachineConfig(**overrides)


# (overrides, expected message fragment) — fragment names the governing field.
INVALID_CASES: list[tuple[dict[str, Any], str]] = [
    # --- core width / head geometry -------------------------------------
    ({"hidden_size": 0}, "hidden_size"),
    ({"num_layers": 0}, "num_layers"),
    ({"num_attention_heads": 0}, "num_attention_heads"),
    ({"num_key_value_heads": 0}, "num_key_value_heads"),
    ({"hidden_size": 1000, "num_attention_heads": 16}, "divisible by num_attention_heads"),
    ({"num_attention_heads": 16, "num_key_value_heads": 5}, "divisible by num_key_value_heads"),
    ({"intermediate_size": 0}, "intermediate_size"),
    ({"rms_norm_eps": 0.0}, "rms_norm_eps"),
    ({"attention_dropout": 1.5}, "attention_dropout"),
    ({"attention_dropout": -0.1}, "attention_dropout"),
    ({"hidden_dropout": 1.5}, "hidden_dropout"),
    ({"hidden_dropout": -0.1}, "hidden_dropout"),
    ({"initializer_range": 0.0}, "initializer_range"),
    # --- CVPE / FRVM -----------------------------------------------------
    ({"cvpe_max_dims": 0}, "cvpe_max_dims"),
    ({"cvpe_mlp_hidden": 0}, "cvpe_mlp_hidden"),
    ({"frvm_chunk_size": 0}, "frvm_chunk_size"),
    ({"frvm_ov_mlp_hidden": 0}, "frvm_ov_mlp_hidden"),
    ({"cvpe_max_dims": 2, "cvpe_axis_max": (10,)}, "one per axis"),
    ({"cvpe_max_dims": 2, "cvpe_axis_max": (10, 0)}, "entries must be positive"),
    ({"cvpe_max_dims": 2, "cvpe_axis_max": (10, -1)}, "entries must be positive"),
    # --- conv / perceiver ------------------------------------------------
    ({"conv_kernel_size": 0}, "conv_kernel_size"),
    ({"conv_kernel_size": 4}, "must be odd"),
    ({"num_perceiver_query": 0}, "num_perceiver_query"),
    ({"perceiver_num_iterations": 0}, "perceiver_num_iterations"),
    ({"perceiver_attention_mode": "dense"}, "perceiver_attention_mode"),
    ({"perceiver_query_mode": "random"}, "perceiver_query_mode"),
    ({"perceiver_sparse_compression_ratio": 0}, "perceiver_sparse_compression_ratio"),
    ({"max_position_embeddings": 0}, "max_position_embeddings"),
    # --- CSA2 ------------------------------------------------------------
    ({"csa_compression_ratio": 0}, "csa_compression_ratio"),
    ({"csa_encoder_compression_ratio": 0}, "csa_encoder_compression_ratio"),
    ({"csa_decoder_compression_ratio": 0}, "csa_decoder_compression_ratio"),
    ({"hca_compression_ratio": 0}, "hca_compression_ratio"),
    ({"csa_indexer_heads": 0}, "csa_indexer_heads"),
    ({"csa_indexer_dim": 0}, "csa_indexer_dim"),
    ({"csa_top_k": 0}, "csa_top_k"),
    ({"csa_q_latent_dim": -1}, "csa_q_latent_dim"),
    ({"csa_num_output_groups": 0}, "csa_num_output_groups"),
    ({"csa_num_output_groups": 7}, "divisible by csa_num_output_groups"),
    ({"csa_output_group_dim": 0}, "csa_output_group_dim"),
    ({"csa_hier_pool_blocks": 0}, "csa_hier_pool_blocks"),
    ({"csa_hier_pool_positions": 0}, "csa_hier_pool_positions"),
    ({"csa_hier_block_size": 0}, "csa_hier_block_size"),
    ({"csa_hier_pool_positions": 256}, ">= csa_top_k"),
    ({"csa_hier_pool_blocks": 2}, "smaller candidate pool"),
    # --- CED layout ------------------------------------------------------
    ({"ced_masked_token_count": 0}, "ced_masked_token_count"),
    ({"ced_num_encoder_layers": -1}, "ced_num_encoder_layers"),
    ({"ced_num_decoder_layers": -1}, "ced_num_decoder_layers"),
    ({"ced_encoder_ffn": "gelu"}, "ced_encoder_ffn"),
    ({"ced_decoder_ffn": "gelu"}, "ced_decoder_ffn"),
    ({"ced_encoder_first_window_size": -1}, "ced_encoder_first_window_size"),
    ({"ced_encoder_norm": "none"}, "ced_encoder_norm"),
    ({"ced_encoder_ffn_norm": "none"}, "ced_encoder_ffn_norm"),
    ({"ced_decoder_ffn_norm": "none"}, "ced_decoder_ffn_norm"),
    ({"ced_encoder_ffn_intermediate_size": -1}, "ced_encoder_ffn_intermediate_size"),
    ({"ced_decoder_ffn_intermediate_size": -1}, "ced_decoder_ffn_intermediate_size"),
    ({"ced_encoder_ffn_update_intervals": (1, 2)}, "must hold 20 entries"),
    ({"ced_encoder_ffn_update_intervals": (0,) * 20}, "positive ints"),
    ({"ced_decoder_ffn_update_intervals": (1, 2)}, "must hold 20 entries"),
    ({"ced_decoder_ffn_update_intervals": (0,) * 20}, "positive ints"),
    ({"ced_decoder_looping": True, "ced_decoder_ffn_update_intervals": (1, 2)}, "must hold 1 entries"),
    ({"ced_decoder_loop_steps": 0}, "ced_decoder_loop_steps"),
    ({"ced_decoder_loop_exit_threshold": 1.5}, "ced_decoder_loop_exit_threshold"),
    ({"ced_decoder_loop_exit_threshold": -0.1}, "ced_decoder_loop_exit_threshold"),
    ({"ced_decoder_hidden_size": -1}, "ced_decoder_hidden_size"),
    ({"ced_decoder_hidden_size": 100}, "divisible by num_attention_heads"),
    # --- GDN-2 -----------------------------------------------------------
    ({"ced_gdn2_backend": "flash"}, "ced_gdn2_backend"),
    ({"ced_gdn2_chunk_size": 0}, "ced_gdn2_chunk_size"),
    ({"ced_gdn2_expand_v": 0.0}, "ced_gdn2_expand_v"),
    ({"ced_gdn2_num_v_heads": -1}, "ced_gdn2_num_v_heads"),
    ({"ced_gdn2_num_v_heads": 5}, "multiple of num_attention_heads"),
    # --- Engram / mHC ----------------------------------------------------
    ({"engram_conv_kernel_size": 0}, "engram_conv_kernel_size"),
    ({"mhc_num_streams": 0}, "mhc_num_streams"),
    ({"mhc_sinkhorn_iters": 0}, "mhc_sinkhorn_iters"),
    ({"engram_orders": ()}, "engram_orders must be non-empty"),
    ({"engram_orders": (1, 2)}, "must all be >= 2"),
    ({"engram_hash_heads": 0}, "engram_hash_heads"),
    ({"engram_dim_per_order": 0}, "engram_dim_per_order"),
    ({"engram_num_entries": 1}, "engram_num_entries"),
    ({"engram_layer_positions": (-1,)}, "engram_layer_positions"),
    ({"engram_offload": "disk"}, "engram_offload"),
    ({"engram_offload_dir": ""}, "engram_offload_dir"),
    ({"engram_offload_ram_threshold": 0}, "engram_offload_ram_threshold"),
    ({"engram_hash_heads": 8, "engram_num_entries": 1 << 21}, "Engram tables total"),
    # --- Titans / CMS ----------------------------------------------------
    ({"titan_memory_depth": 0}, "titan_memory_depth"),
    ({"titan_fast_expansion": 0.0}, "titan_fast_expansion"),
    ({"titan_objective": "cosine"}, "titan_objective"),
    ({"titan_conv_kernel_size": 0}, "titan_conv_kernel_size"),
    ({"titan_chunk_size": -1}, "titan_chunk_size"),
    ({"titan_backend": "flash"}, "titan_backend"),
    ({"cms_ffn": "gelu"}, "cms_ffn"),
    ({"cms_norm": "none"}, "cms_norm"),
    # --- dSpark ----------------------------------------------------------
    ({"dspark_num_drafts": 0}, "dspark_num_drafts"),
    ({"dspark_window": 0}, "dspark_window"),
    # --- BLT local byte model -------------------------------------------
    ({"blt_vocab_size": -1}, "blt_vocab_size"),
    ({"blt_patch_size": 0}, "blt_patch_size"),
    ({"blt_local_hidden_size": -1}, "blt_local_hidden_size"),
    ({"blt_local_hidden_size": 100}, "divisible by num_attention_heads"),
    ({"blt_local_layers": 0}, "blt_local_layers"),
    ({"blt_local_ffn": "gelu"}, "blt_local_ffn"),
    ({"blt_local_ffn_intermediate_size": -1}, "blt_local_ffn_intermediate_size"),
    ({"blt_local_norm": "none"}, "blt_local_norm"),
    ({"blt_ngram_sizes": (1, 3)}, "must all be >= 2"),
    ({"blt_ngram_hash_size": 1}, "blt_ngram_hash_size"),
    ({"blt_cross_attn_heads": -1}, "blt_cross_attn_heads"),
    ({"blt_cross_attn_heads": 7}, "must divide the local width"),
    ({"blt_max_patch_bytes": 0}, "blt_max_patch_bytes"),
    ({"blt_local_max_position_embeddings": -1}, "blt_local_max_position_embeddings"),
    # --- BLT latent sampler (blt_latent_*) --------------------------------
    ({"blt_latent_target_std": 0.0}, "blt_latent_target_std"),
    ({"blt_latent_temperature": -0.1}, "blt_latent_temperature"),
    ({"blt_latent_min_std": 0.0}, "blt_latent_min_std"),
    ({"blt_latent_min_std": 0.5}, "must not exceed"),
    ({"blt_latent_std_reg_weight": -0.1}, "blt_latent_std_reg_weight"),
]


def _case_id(case: tuple[dict[str, Any], str]) -> str:
    overrides, _ = case
    return "-".join(f"{key}={value!r}" for key, value in overrides.items())


@pytest.mark.parametrize("overrides,message", INVALID_CASES, ids=[_case_id(c) for c in INVALID_CASES])
def test_invalid_config_raises_value_error(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_cfg(**overrides)


def test_default_config_is_valid() -> None:
    cfg = make_cfg()
    assert cfg.hidden_size == 1024
    assert cfg.csa_hier_pool_blocks * cfg.csa_hier_block_size >= cfg.csa_top_k


def test_zero_means_unset_boundaries_are_accepted() -> None:
    # 0 is the documented "derive from the shared width" sentinel for each of
    # these; the guards must reject negatives but accept the sentinel.
    cfg = make_cfg(
        csa_q_latent_dim=0,
        ced_encoder_first_window_size=0,
        ced_encoder_ffn_intermediate_size=0,
        ced_decoder_ffn_intermediate_size=0,
        ced_decoder_hidden_size=0,
        blt_vocab_size=0,
        blt_local_hidden_size=0,
        blt_cross_attn_heads=0,
        blt_local_ffn_intermediate_size=0,
        blt_local_max_position_embeddings=0,
        titan_chunk_size=0,
        ced_gdn2_num_v_heads=0,
        titan_memory_depth=1,
        blt_patch_size=1,
        blt_local_layers=1,
        blt_ngram_sizes=(2,),
        blt_ngram_hash_size=2,
        blt_max_patch_bytes=1,
        engram_num_entries=2,
        engram_orders=(2,),
        ced_decoder_loop_steps=1,
        ced_masked_token_count=1,
        csa_top_k=1,
        csa_hier_pool_positions=1,
        csa_hier_pool_blocks=1,
        csa_hier_block_size=1,
        perceiver_num_iterations=1,
        num_perceiver_query=1,
    )
    assert cfg.blt_patch_size == 1


def test_closed_interval_boundaries_are_accepted() -> None:
    cfg = make_cfg(
        attention_dropout=0.0,
        hidden_dropout=1.0,
        ced_decoder_loop_exit_threshold=0.0,
        cvpe_max_dims=2,
        cvpe_axis_max=(1, 1),
        ced_encoder_ffn_update_intervals=(1,) * 20,
        ced_decoder_ffn_update_intervals=(1,) * 20,
    )
    assert cfg.attention_dropout == 0.0


def test_looped_decoder_accepts_single_update_interval() -> None:
    # The looped decoder shares one CSA2 -> DenseFFN layer, so dec_n == 1.
    cfg = make_cfg(ced_decoder_looping=True, ced_decoder_ffn_update_intervals=(3,))
    assert cfg.ced_decoder_ffn_update_intervals == (3,)


@pytest.mark.parametrize(
    "size,hidden_size,heads,kv_heads",
    [("350M", 768, 12, 3), ("700M", 1024, 16, 4), ("1.2B", 1536, 24, 6)],
    ids=["350M", "700M", "1.2B"],
)
def test_from_pretrained_size_returns_valid_config(size: str, hidden_size: int, heads: int, kv_heads: int) -> None:
    cfg = MemoryMachineConfig.from_pretrained_size(size)
    assert isinstance(cfg, MemoryMachineConfig)
    assert cfg.hidden_size == hidden_size
    assert cfg.num_attention_heads == heads
    assert cfg.num_key_value_heads == kv_heads
    assert cfg.hidden_size % cfg.num_attention_heads == 0
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0


def test_from_pretrained_size_350m_keeps_output_groups_dividing_heads() -> None:
    cfg = MemoryMachineConfig.from_pretrained_size("350M")
    assert cfg.csa_num_output_groups == 6
    assert cfg.num_attention_heads % cfg.csa_num_output_groups == 0


def test_from_pretrained_size_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model size"):
        MemoryMachineConfig.from_pretrained_size("42B")
