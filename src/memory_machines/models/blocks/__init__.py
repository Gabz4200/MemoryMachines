r"""Composable blocks for MemoryMachine architectures (back-compat re-export).

Aggregates dense attention, sparse attention, compression primitives, LIV
mixers, auxiliary memory, positioning, and shared cache helpers so existing
``memory_machines.models.blocks`` imports keep working.
"""

from .attention import GroupedQueryAttention, SwiGLU
from .causal_conv1d import ShortConvolution
from .ced import CausalDecoder, CausalEncoderDecoderStack, CEDGlobalKVProjection
from .compression import (
    AttentionSink,
    BlockCompressor,
    BranchGate,
    GroupedOutputProjection,
    HeadRMSNorm,
    SparseIndexer,
)
from .fast_weights import (
    ContinuumMemorySystem,
    HopeAttentionBlock,
    HopeBlock,
    SelfReferentialTitanMemory,
    SelfReferentialTitanMixer,
    TitansFastWeightsMemory,
    TitansMAGBlock,
)
from .gdn2 import GatedDeltaNet2, add_resting_projection, apply_beta_guard, gated_delta_rule_2, resting_corrected_write
from .linears import ChunkedBottleneckLinear, FactoredLinear
from .liv import LivConvBlock, LivDoubleGatedBase, LivPerceiver
from .memory import DSparkDrafter, EngramMemory, SinglePassMHC
from .positioning import CVPEContextBias, FRVMValueGate
from .shared import (
    CompressedCache,
    CSA2SharedState,
    _causal_comp_mask,
    _check_position_bounds,
    _past_total_len,
    _resolve_positions,
    _sinkhorn,
    _update_swa_pos,
    _window_mask,
)
from .sparse import (
    CompressedSparseAttention,
    CompressedSparseAttention2,
    HeavilyCompressedAttention,
    NativeSparseAttention,
)

__all__ = [
    "AttentionSink",
    "BlockCompressor",
    "BranchGate",
    "CEDGlobalKVProjection",
    "CSA2SharedState",
    "CVPEContextBias",
    "CausalDecoder",
    "CausalEncoderDecoderStack",
    "ChunkedBottleneckLinear",
    "CompressedCache",
    "CompressedSparseAttention",
    "CompressedSparseAttention2",
    "ContinuumMemorySystem",
    "DSparkDrafter",
    "EngramMemory",
    "FRVMValueGate",
    "FactoredLinear",
    "GatedDeltaNet2",
    "GroupedOutputProjection",
    "GroupedQueryAttention",
    "HeadRMSNorm",
    "HeavilyCompressedAttention",
    "HopeAttentionBlock",
    "HopeBlock",
    "LivConvBlock",
    "LivDoubleGatedBase",
    "LivPerceiver",
    "NativeSparseAttention",
    "SelfReferentialTitanMemory",
    "SelfReferentialTitanMixer",
    "ShortConvolution",
    "SinglePassMHC",
    "SparseIndexer",
    "SwiGLU",
    "TitansFastWeightsMemory",
    "TitansMAGBlock",
    "_causal_comp_mask",
    "_check_position_bounds",
    "_past_total_len",
    "_resolve_positions",
    "_sinkhorn",
    "_update_swa_pos",
    "_window_mask",
    "add_resting_projection",
    "apply_beta_guard",
    "gated_delta_rule_2",
    "resting_corrected_write",
]
