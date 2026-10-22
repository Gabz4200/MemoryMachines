r"""Composable blocks for MemoryMachine architectures (back-compat re-export).

Aggregates dense attention, sparse attention, compression primitives, LIV
mixers, auxiliary memory, positioning, and shared cache helpers so existing
``memory_machines.models.blocks`` imports keep working.
"""

from .attention import GroupedQueryAttention
from .blt import VocabularyDecoder, VocabularyEncoder, compute_patch_ids
from .causal_conv1d import ShortConvolution
from .ced import AdaptiveExitGate, CausalDecoder, CausalEncoderDecoderStack, CEDGlobalKVProjection, VocabularyUpProjection
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
from .ffn import DenseFFN, SwiGLU, build_ffn
from .gdn2 import GatedDeltaNet2, GatedDeltaNet2Mixer, add_resting_projection, apply_beta_guard, gated_delta_rule_2, resting_corrected_write
from .latent_sampling import GaussianLatentSampler, LatentSample
from .linear_init import (
    copy_linear_weight_,
    effective_weight,
    factorize_linear,
    init_linear,
    init_linear_columns,
    init_linear_xavier,
    linear_dtype,
    linear_in_features,
    linear_out_features,
    zero_linear,
)
from .linears import ChunkedBottleneckLinear, FactoredLinear
from .liv import LivConvBlock, LivDoubleGatedBase, LivPerceiver
from .low_rank import RANK_ROLES, RankConfig, RankPolicy, apply_low_rank, linear_role
from .memory import DSparkDrafter, EngramMemory, SinglePassMHC, from_streams, to_streams
from .parallel_hybrid import ParallelAttentionGDN2, ParallelAttentionTitan
from .positioning import CVPEContextBias, FRVMSpatialOV, FRVMValueGate
from .rkan import JacobiRKAN, PadeRKAN
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
    "RANK_ROLES",
    "AdaptiveExitGate",
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
    "DenseFFN",
    "EngramMemory",
    "FRVMSpatialOV",
    "FRVMValueGate",
    "FactoredLinear",
    "GatedDeltaNet2",
    "GatedDeltaNet2Mixer",
    "GaussianLatentSampler",
    "GroupedOutputProjection",
    "GroupedQueryAttention",
    "HeadRMSNorm",
    "HeavilyCompressedAttention",
    "HopeAttentionBlock",
    "HopeBlock",
    "JacobiRKAN",
    "LatentSample",
    "LivConvBlock",
    "LivDoubleGatedBase",
    "LivPerceiver",
    "NativeSparseAttention",
    "PadeRKAN",
    "ParallelAttentionGDN2",
    "ParallelAttentionTitan",
    "RankConfig",
    "RankPolicy",
    "SelfReferentialTitanMemory",
    "SelfReferentialTitanMixer",
    "ShortConvolution",
    "SinglePassMHC",
    "SparseIndexer",
    "SwiGLU",
    "TitansFastWeightsMemory",
    "TitansMAGBlock",
    "VocabularyDecoder",
    "VocabularyEncoder",
    "VocabularyUpProjection",
    "_causal_comp_mask",
    "_check_position_bounds",
    "_past_total_len",
    "_resolve_positions",
    "_sinkhorn",
    "_update_swa_pos",
    "_window_mask",
    "add_resting_projection",
    "apply_beta_guard",
    "apply_low_rank",
    "build_ffn",
    "compute_patch_ids",
    "copy_linear_weight_",
    "effective_weight",
    "factorize_linear",
    "from_streams",
    "gated_delta_rule_2",
    "init_linear",
    "init_linear_columns",
    "init_linear_xavier",
    "linear_dtype",
    "linear_in_features",
    "linear_out_features",
    "linear_role",
    "resting_corrected_write",
    "to_streams",
    "zero_linear",
]
