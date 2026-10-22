"""Guard that the rank budget covers every projection in the shipped models.

Each projection is either factored or dense for a stated reason: the role
budget asks for a rank, and the builder keeps it dense when factoring would
not actually save parameters. These tests fail when a new ``nn.Linear`` lands
in a backbone without a role, and when the budget stops saving parameters.
"""

from typing import Any

import pytest
import torch
from torch import nn

from memory_machines.models.backbone import BLTBackbone, CheapLLM
from memory_machines.models.blocks.linear_init import effective_weight, init_linear
from memory_machines.models.blocks.linears import FactoredLinear
from memory_machines.models.blocks.low_rank import (
    RANK_ROLES,
    RankPolicy,
    apply_low_rank,
    linear_role,
)
from memory_machines.utils.config_schema import MemoryMachineConfig

VOCAB = 64


def _ced_cfg(**kw: Any) -> MemoryMachineConfig:
    base: dict[str, Any] = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": VOCAB,
        "ced_num_encoder_layers": 2,
        "ced_num_decoder_layers": 2,
        "csa_top_k": 4,
        "csa_hier_pool_blocks": 4,
        "csa_hier_pool_positions": 8,
        "csa_indexer_heads": 2,
        "csa_indexer_dim": 8,
        "csa_num_output_groups": 2,
        "csa_output_group_dim": 8,
        "ced_decoder_hidden_size": 16,
    }
    base.update(kw)
    return MemoryMachineConfig(**base)


def _blt_cfg(**kw: Any) -> MemoryMachineConfig:
    base: dict[str, Any] = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": VOCAB,
        "blt_vocab_size": 32,
        "blt_local_hidden_size": 16,
        "blt_local_layers": 1,
        "blt_local_max_position_embeddings": 64,
        "blt_cross_attn_heads": 2,
        "blt_ngram_sizes": (3, 4),
        "blt_ngram_hash_size": 8,
        "blt_latent_sampling": True,
        "ced_num_encoder_layers": 1,
        "ced_num_decoder_layers": 1,
        "csa_top_k": 4,
        "csa_hier_pool_blocks": 4,
        "csa_hier_pool_positions": 8,
        "csa_indexer_heads": 2,
        "csa_indexer_dim": 8,
        "csa_num_output_groups": 2,
        "csa_output_group_dim": 8,
    }
    base.update(kw)
    return MemoryMachineConfig(**base)


def _unbudgeted_linears(model: nn.Module, policy: RankPolicy) -> list[str]:
    """Projections still dense although the policy would factor them."""
    factor_children = {f"{name}.{leaf}" for name, child in model.named_modules() if isinstance(child, FactoredLinear) for leaf in ("in_linear", "out_linear")}
    leaks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name in factor_children:
            continue
        role = linear_role(name)
        if policy.rank_for(module.in_features, module.out_features, role) is not None:
            leaks.append(f"{name} ({role}): {module.in_features}x{module.out_features}")
    return leaks


def _factored(model: nn.Module) -> dict[str, FactoredLinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, FactoredLinear)}


@pytest.mark.parametrize("factory", [CheapLLM, BLTBackbone], ids=["ced", "blt"])
def test_budget_leaves_no_projection_uncovered(factory: Any) -> None:
    cfg = _ced_cfg() if factory is CheapLLM else _blt_cfg()
    model = factory(cfg)
    assert _unbudgeted_linears(model, RankPolicy.from_config(cfg)) == []
    assert len(_factored(model)) > 5


def test_default_budget_saves_parameters() -> None:
    cfg = _ced_cfg()
    factored = sum(p.numel() for p in CheapLLM(cfg).parameters())
    dense = sum(p.numel() for p in CheapLLM(_ced_cfg(low_rank_mode="dense")).parameters())
    assert factored < dense


def test_factored_shapes_and_rank_stay_below_cap() -> None:
    model = CheapLLM(_ced_cfg())
    for name, module in _factored(model).items():
        assert module.rank < min(module.input_dim, module.output_dim), name
        assert module.rank * (module.input_dim + module.output_dim) < module.input_dim * module.output_dim, name


def test_apply_low_rank_is_idempotent_and_rejects_unknown_roles() -> None:
    model = CheapLLM(_ced_cfg())
    before = len(_factored(model))
    assert apply_low_rank(model, RankPolicy.from_config(model.config)) == 0
    assert len(_factored(model)) == before
    with pytest.raises(ValueError, match="unknown rank role"):
        RankPolicy(ratios={"no_such_role": 0.5})


def test_every_config_ratio_names_a_known_role() -> None:
    cfg = MemoryMachineConfig()
    assert {role for role, _ in cfg.rank_ratios} <= RANK_ROLES


def test_factored_init_matches_dense_scale() -> None:
    """A factored projection preserves the activation scale of its dense peer."""
    torch.manual_seed(0)
    dense = nn.Linear(64, 64, bias=False)
    init_linear(dense, 0.1)
    holder = nn.ModuleDict({"q_proj": dense})
    assert apply_low_rank(holder, RankPolicy(ratios={"attn_q": 0.25})) == 1
    factored = holder["q_proj"]
    assert isinstance(factored, FactoredLinear)
    std = effective_weight(factored).std().item()
    assert std == pytest.approx(0.1, rel=0.35)
