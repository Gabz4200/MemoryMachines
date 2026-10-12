"""Structured Config schemas. Validate cross-group invariants at boot, before GPU init."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    embed_dim: int = 256
    num_layers: int = 6
    backend: str = "naive"


@dataclass
class MemoryMachineConfig:
    """Configuration class for MemoryMachine model.

    Attributes:
        vocab_size: Size of the vocabulary
        hidden_size: Hidden dimension size
        intermediate_size: Feedforward intermediate dimension
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads for GQA
        conv_kernel_size: Kernel size for short convolutions
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: RMS normalization epsilon
        tie_word_embeddings: Whether to tie input/output embeddings
        cvpe_mlp_hidden: Hidden width of the CVPE QK functional-bias MLP
        cvpe_use_log_transform: Log-warp the normalized offset before the MLP
        cvpe_enabled: Whether to add the CVPE dynamic QK bias
        frvm_enabled: Whether to apply F-RVM antisymmetric value gating
        frvm_chunk_size: Query chunk size for the F-RVM gated reduction
        attention_dropout: Dropout rate for attention
        hidden_dropout: Dropout rate for hidden layers
        initializer_range: Standard deviation for weight initialization
        use_cache: Whether to use key-value caching
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        use_return_dict: Whether to return outputs as a dictionary
        output_attentions: Whether to output attention weights by default
        output_hidden_states: Whether to output hidden states by default
        num_perceiver_query: Learned latent queries per head (LivPerceiver)
        swa_window_size: Sliding-window width (V4/V4.1: 128; NSA: 512)
        swa_bounded_replay: Enable SWA Bounded Replay (replay last window only)
        csa_compression_ratio: Default m:1 block compression (V4 CSA: 4)
        csa_encoder_compression_ratio: CED encoder m (V4.1: 2)
        csa_decoder_compression_ratio: CED decoder m (V4.1: 1 = uncompressed)
        hca_compression_ratio: Heavily Compressed Attention m' (V4: 128)
        csa_indexer_heads: Sparse indexer query heads (V4.1: 32; V4: 64)
        csa_indexer_dim: Indexer head dim (V4/V4.1: 128)
        csa_top_k: Top-K compressed entries per query (V4.1: 512)
        csa_q_latent_dim: Shared low-rank query latent dim (0 = hidden_size)
        csa_num_output_groups: Grouped output projection groups g (V4.1: 8)
        csa_output_group_dim: Per-group output dim d_g (V4.1: 1024)
        csa_use_sink: Learnable attention sink per head (mass < 1)
        csa_use_qkv_norm: Extra RMSNorm on Q heads + shared KV head
        csa_hier_pool_blocks: Hierarchical indexer pool blocks (V4.1: 2048)
        csa_hier_pool_positions: Hierarchical pool position budget (V4.1: 16384); caps pool_blocks*block_size
        csa_hier_block_size: Positions per pool block (V4.1: 8)
        ced_num_encoder_layers: Bottom-half encoder layers L/2 (V4.1: 20)
        ced_num_decoder_layers: Top-half decoder layers L/2 (V4.1: 20)
        mhc_num_streams: Residual stream expansion factor n (V4/V4.1: 4)
        mhc_sinkhorn_iters: Sinkhorn-Knopp iterations (V4: 20)
        engram_orders: N-gram orders (V4.1: 2,3,4)
        engram_hash_heads: Hash heads (V4.1: 8)
        engram_dim_per_order: Dim per order (V4.1: 2048; default scaled down)
        engram_num_entries: Minimum hash-table size per head; actual sizes are
            distinct primes >= engram_num_entries (V4.1: ~16M/head; default tiny)
        engram_use_conv: Short depthwise causal conv refinement (original paper:
            True; V4.1 omits the conv, so the flag is configurable)
        engram_layer_positions: Zero-indexed layers that host Engram blocks
            (V4.1: 1 and 14; empty = every layer)
        engram_offload: Storage mode for the hashed tables: "ram" (default,
            an in-memory nn.Embedding), "ssd" (always a numpy-memmap file
            on SSD, gathered on demand), or "auto" (ssd when the table
            would exceed engram_offload_ram_threshold bytes of RAM, else ram)
        engram_offload_dir: Directory for SSD-backed table files (engram_offload
            in {"ssd", "auto"}); created if missing. Default: "engram_tables"
        engram_offload_ram_threshold: Table bytes above which "auto" picks
            SSD offloading. Default: 1 GiB.
        dspark_num_drafts: Parallel draft positions (V4.1: 5)
        dspark_window: Drafter window (V4.1: 128)
    """

    vocab_size: int = 32768
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    conv_kernel_size: int = 3
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    cvpe_mlp_hidden: int = 32
    cvpe_use_log_transform: bool = True
    cvpe_enabled: bool = True
    frvm_enabled: bool = True
    frvm_chunk_size: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_return_dict: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    num_perceiver_query: int = 6
    swa_window_size: int = 128
    swa_bounded_replay: bool = True
    csa_compression_ratio: int = 4
    csa_encoder_compression_ratio: int = 2
    csa_decoder_compression_ratio: int = 1
    hca_compression_ratio: int = 128
    csa_indexer_heads: int = 32
    csa_indexer_dim: int = 128
    csa_top_k: int = 512
    csa_q_latent_dim: int = 0  # 0 = hidden_size (no bottleneck)
    csa_num_output_groups: int = 8
    csa_output_group_dim: int = 1024
    csa_use_sink: bool = True
    csa_use_qkv_norm: bool = True
    csa_hier_pool_blocks: int = 2048
    csa_hier_pool_positions: int = 16384
    csa_hier_block_size: int = 8
    ced_num_encoder_layers: int = 20
    ced_num_decoder_layers: int = 20
    mhc_num_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    engram_orders: tuple[int, ...] = (2, 3, 4)
    engram_hash_heads: int = 4
    engram_dim_per_order: int = 128
    engram_num_entries: int = 16384
    engram_use_conv: bool = True
    engram_layer_positions: tuple[int, ...] = ()
    engram_offload: str = "ram"
    engram_offload_dir: str = "engram_tables"
    engram_offload_ram_threshold: int = 1 << 30
    seed: int = 0
    dspark_num_drafts: int = 5
    dspark_window: int = 128

    def __post_init__(self):
        """Validate invariant bounds across configuration fields."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads ({self.num_attention_heads}) must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads ({self.num_key_value_heads}) must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size ({self.intermediate_size}) must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps ({self.rms_norm_eps}) must be positive")
        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError(f"attention_dropout ({self.attention_dropout}) must be in [0, 1]")
        if not 0.0 <= self.hidden_dropout <= 1.0:
            raise ValueError(f"hidden_dropout ({self.hidden_dropout}) must be in [0, 1]")
        if self.initializer_range <= 0:
            raise ValueError(f"initializer_range ({self.initializer_range}) must be positive")
        if self.cvpe_mlp_hidden <= 0:
            raise ValueError(f"cvpe_mlp_hidden ({self.cvpe_mlp_hidden}) must be positive")
        if self.frvm_chunk_size <= 0:
            raise ValueError(f"frvm_chunk_size ({self.frvm_chunk_size}) must be positive")
        if self.conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size ({self.conv_kernel_size}) must be positive")
        if self.conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size ({self.conv_kernel_size}) must be odd to preserve length with padding kernel//2")
        if self.num_perceiver_query <= 0:
            raise ValueError(f"num_perceiver_query ({self.num_perceiver_query}) must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings ({self.max_position_embeddings}) must be positive")
        if self.swa_window_size <= 0:
            raise ValueError(f"swa_window_size ({self.swa_window_size}) must be positive")
        if self.csa_compression_ratio <= 0:
            raise ValueError(f"csa_compression_ratio ({self.csa_compression_ratio}) must be positive")
        if self.csa_encoder_compression_ratio <= 0:
            raise ValueError(f"csa_encoder_compression_ratio ({self.csa_encoder_compression_ratio}) must be positive")
        if self.csa_decoder_compression_ratio <= 0:
            raise ValueError(f"csa_decoder_compression_ratio ({self.csa_decoder_compression_ratio}) must be positive")
        if self.hca_compression_ratio <= 0:
            raise ValueError(f"hca_compression_ratio ({self.hca_compression_ratio}) must be positive")
        if self.csa_indexer_heads <= 0:
            raise ValueError(f"csa_indexer_heads ({self.csa_indexer_heads}) must be positive")
        if self.csa_indexer_dim <= 0:
            raise ValueError(f"csa_indexer_dim ({self.csa_indexer_dim}) must be positive")
        if self.csa_top_k <= 0:
            raise ValueError(f"csa_top_k ({self.csa_top_k}) must be positive")
        if self.csa_q_latent_dim < 0:
            raise ValueError(f"csa_q_latent_dim ({self.csa_q_latent_dim}) must be non-negative (0 = hidden_size)")
        if self.csa_num_output_groups <= 0:
            raise ValueError(f"csa_num_output_groups ({self.csa_num_output_groups}) must be positive")
        if self.csa_output_group_dim <= 0:
            raise ValueError(f"csa_output_group_dim ({self.csa_output_group_dim}) must be positive")
        if self.num_attention_heads % self.csa_num_output_groups != 0:
            raise ValueError(f"num_attention_heads ({self.num_attention_heads}) must be divisible by csa_num_output_groups ({self.csa_num_output_groups})")
        if self.csa_hier_pool_blocks <= 0:
            raise ValueError(f"csa_hier_pool_blocks ({self.csa_hier_pool_blocks}) must be positive")
        if self.csa_hier_pool_positions <= 0:
            raise ValueError(f"csa_hier_pool_positions ({self.csa_hier_pool_positions}) must be positive")
        if self.csa_hier_block_size <= 0:
            raise ValueError(f"csa_hier_block_size ({self.csa_hier_block_size}) must be positive")
        if self.csa_hier_pool_positions < self.csa_top_k:
            raise ValueError(f"csa_hier_pool_positions ({self.csa_hier_pool_positions}) must be >= csa_top_k ({self.csa_top_k})")
        if self.csa_hier_pool_blocks * self.csa_hier_block_size < self.csa_top_k:
            raise ValueError(
                f"csa_hier_pool_blocks ({self.csa_hier_pool_blocks}) * csa_hier_block_size ({self.csa_hier_block_size}) "
                f"= {self.csa_hier_pool_blocks * self.csa_hier_block_size} must be >= csa_top_k ({self.csa_top_k}); "
                "otherwise Reindex silently degrades to a smaller candidate pool"
            )
        if self.ced_num_encoder_layers < 0:
            raise ValueError(f"ced_num_encoder_layers ({self.ced_num_encoder_layers}) must be non-negative")
        if self.ced_num_decoder_layers < 0:
            raise ValueError(f"ced_num_decoder_layers ({self.ced_num_decoder_layers}) must be non-negative")
        if self.mhc_num_streams <= 0:
            raise ValueError(f"mhc_num_streams ({self.mhc_num_streams}) must be positive")
        if self.mhc_sinkhorn_iters <= 0:
            raise ValueError(f"mhc_sinkhorn_iters ({self.mhc_sinkhorn_iters}) must be positive")
        if len(self.engram_orders) == 0:
            raise ValueError("engram_orders must be non-empty")
        if any(o < 2 for o in self.engram_orders):
            raise ValueError(f"engram_orders ({self.engram_orders}) must all be >= 2")
        if self.engram_hash_heads <= 0:
            raise ValueError(f"engram_hash_heads ({self.engram_hash_heads}) must be positive")
        if self.engram_dim_per_order <= 0:
            raise ValueError(f"engram_dim_per_order ({self.engram_dim_per_order}) must be positive")
        if self.engram_num_entries < 2:
            raise ValueError(f"engram_num_entries ({self.engram_num_entries}) must be >= 2")
        if any(p < 0 for p in self.engram_layer_positions):
            raise ValueError(f"engram_layer_positions ({self.engram_layer_positions}) must be non-negative")
        if self.engram_offload not in ("ram", "ssd", "auto"):
            raise ValueError(f'engram_offload ({self.engram_offload!r}) must be "ram", "ssd", or "auto"')
        if not self.engram_offload_dir:
            raise ValueError("engram_offload_dir must be non-empty")
        if self.engram_offload_ram_threshold <= 0:
            raise ValueError(f"engram_offload_ram_threshold ({self.engram_offload_ram_threshold}) must be positive")
        engram_elements = len(self.engram_orders) * self.engram_hash_heads * self.engram_num_entries * self.engram_dim_per_order
        if engram_elements > 268_435_456:
            raise ValueError(f"Engram tables total {engram_elements:,} elements (> 256M, ~1GB fp32); reduce engram_num_entries/engram_dim_per_order or shard explicitly")
        if self.dspark_num_drafts <= 0:
            raise ValueError(f"dspark_num_drafts ({self.dspark_num_drafts}) must be positive")
        if self.dspark_window <= 0:
            raise ValueError(f"dspark_window ({self.dspark_window}) must be positive")

    @classmethod
    def from_pretrained_size(cls, size: str) -> "MemoryMachineConfig":
        configs = {
            "350M": cls(
                hidden_size=768,
                intermediate_size=2048,
                num_attention_heads=12,
                num_key_value_heads=3,
                # 12 heads is not divisible by the default 8 output groups.
                csa_num_output_groups=6,
            ),
            "700M": cls(
                hidden_size=1024,
                intermediate_size=2816,
                num_attention_heads=16,
                num_key_value_heads=4,
            ),
            "1.2B": cls(
                hidden_size=1536,
                intermediate_size=4096,
                num_attention_heads=24,
                num_key_value_heads=6,
            ),
        }

        if size not in configs:
            raise ValueError(f"Unknown model size: {size}. Available: {list(configs.keys())}")

        return configs[size]


@dataclass
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "bf16-mixed"
    strategy: str = "auto"
    fast_dev_run: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
