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
        num_layers: Total transformer layers ``N`` (encoder + decoder
            or single-stack). Scales residual output-projection init by
            ``1 / sqrt(2N)`` when ``residual_init_scale`` is set.
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads for GQA
        conv_kernel_size: Kernel size for short convolutions
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: RMS normalization epsilon
        tie_word_embeddings: Whether to tie input/output embeddings
        cvpe_mlp_hidden: Hidden width of the CVPE QK functional-bias MLP
        cvpe_use_log_transform: Log-warp the normalized offset before the MLP
        cvpe_enabled: Whether to add the CVPE dynamic QK bias
        cvpe_max_dims: Fixed input width of the CVPE/FRVM MLPs (N_max).
            Pure text uses 1, images 2, video 3, embodied/robotics 6.
            Unused axes are zero-padded by the dataloader, so a single
            set of model weights handles every modality.
        cvpe_axis_max: Per-axis normalization caps C_max (tuple of
            length cvpe_max_dims). Empty -> max_position_embeddings for
            every axis.
        frvm_enabled: Whether to apply F-RVM antisymmetric value gating
        frvm_chunk_size: Query chunk size for the F-RVM gated reduction
        frvm_ov_mlp_hidden: Hidden width of the F-RVM spatial OV MLP
            (MLP_OV: R^N -> R^Dh)
        frvm_use_log_transform: Log-warp the normalized offset before the
            F-RVM spatial OV MLP (mirrors cvpe_use_log_transform)
        frvm_use_spatial: Whether to add the F-RVM spatial OV term
        frvm_cross_heads: When True (cross-attention), the F-RVM value
            path uses separate modality-specific projection heads
            (Phi_q for the query stream, Phi_k for the key-value stream)
            so V_q and V_k are aligned before the antisymmetric
            difference. In self-attention the two heads collapse to a
            single shared projection.
        attention_dropout: Dropout rate for attention
        hidden_dropout: Dropout rate for hidden layers
        initializer_range: Standard deviation for weight
            initialization (DeepSeek uses 0.006; standard
            Transformer default 0.02)
        residual_init_scale: When True, scale residual output
            projections (attention ``o_proj``, FFN ``down_proj``,
            grouped output) by ``initializer_range / sqrt(2N)``
            (Megatron-LM/Qwen depth scaling). When False, residual
            outputs are zero-initialized (identity residual start).
        use_cache: Whether to use key-value caching
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        use_return_dict: Whether to return outputs as a dictionary
        output_attentions: Whether to output attention weights by default
        output_hidden_states: Whether to output hidden states by default
        num_perceiver_query: Learned latent queries per head (LivPerceiver)
        window_size: Sliding-window width. ``<= 0`` disables the window
            (full attention); ``> 0`` bounds causal attention to the last
            ``window_size`` keys (plain SWA: 256). Compressed/sparse/shared
            variants scale the window by their mechanism (see the
            ``window_size_*`` fields); every value stays a tunable
            hyperparameter, not an architectural guarantee.
        swa_bounded_replay: Enable SWA Bounded Replay (replay last window only)
        csa_compression_ratio: Default m:1 block compression (V4 CSA: 4)
        window_size_compressed: Window for m:1 compressed attention, in
            compressed tokens. Scales with ``csa_compression_ratio``:
            ``256 * 4 = 1024`` covers the same token span as plain SWA.
        window_size_sparse: Window for compressed-sparse attention, in
            compressed tokens. The sparse indexer already filters most
            of the store, so the window can be wider (``256 * 8``).
        window_size_shared: Window for layers sharing the compressed KV
            store with other layers; the per-layer budget scales further
            (``256 * 16``).
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
        ced_attention_gdn2_parallel: Build each CED *encoder* layer as
            parallel attention ∥ MultiHead GDN-2 branches routed by
            Single-Pass mHC (mHC streams persist between adjacent encoder
            layers). The decoder stays plain CSA2: GDN-2 reaches it only
            indirectly, through the encoder's global KV (Eq. 1). Default:
            True.
        ced_encoder_alternating: In the CED encoder, alternate between
            parallel Attention ∥ GDN-2 (even layers) and parallel
            Attention ∥ SelfReferentialTitans (odd layers) mixed by mHC.
            Default: True.
        ced_attention_variant: MultiHead SWA variant for the encoder
            attention branch: ``dense`` (GroupedQueryAttention),
            ``compressed``, ``compressed_sparse`` (default), or
            ``heavily_compressed``. Each maps to its own sliding window
            (``window_size`` / ``window_size_compressed`` /
            ``window_size_sparse`` / ``window_size_shared``).
        ced_encoder_ffn: Dense FFN kind for every CED *encoder* layer
            (currently ``"swiglu"``). Configured independently of the decoder
            so the two halves may use different FFNs. Default: ``"swiglu"``.
        ced_decoder_ffn: Dense FFN kind for every CED *decoder* layer
            (currently ``"swiglu"``). Default: ``"swiglu"``.
        ced_encoder_ffn_intermediate_size: Encoder FFN hidden width; ``0``
            means the shared ``intermediate_size``. Default: ``0``.
        ced_decoder_ffn_intermediate_size: Decoder FFN hidden width; ``0``
            means the shared ``intermediate_size``. Default: ``0``.
        ced_decoder_looping: Build the CED decoder in the Looped Language
            Model style (arXiv:2510.25741): the repeated
            ``CSA2 -> DenseFFN`` pattern collapses into one
            parameter-shared layer plus one shared encoder-KV projection,
            applied ``ced_decoder_loop_steps`` times. Default: ``False``.
        ced_decoder_loop_steps: Recurrent depth ``T_max`` of the looped
            decoder (number of times the shared layer is applied). Default:
            ``4``.
        ced_decoder_loop_exit_threshold: Q-exit threshold ``q`` for the
            decoder's exit gate: the loop stops at the first step whose
            cumulative exit probability reaches ``q``. ``0.0`` exits
            immediately, ``1.0`` forces a full loop. Default: ``0.5``.
        ced_decoder_hidden_size: Asymmetric decoder width. ``0`` (default)
            keeps the decoder at ``hidden_size`` (symmetric encoder/decoder).
            When set, the encoder stays at ``hidden_size`` (wide prefill) and
            the decoder half is built at this narrower width; the decoder
            global-KV projection doubles as the cross-attention downsample
            (``W_K``/``W_V``: ``hidden_size`` -> decoder head space).
            Must be divisible by ``num_attention_heads``.
        ced_gdn2_backend: GDN-2 reference path: ``naive``/``recurrent``
            (recurrent form, used at inference) or ``chunk``/``wy_fast``
            (parallel chunkwise form, training speed-up only). Both forms
            mix the *full* encoder sequence — GDN-2 is never windowed.
            Default: ``naive``.
        ced_gdn2_chunk_size: WY chunk width for the parallel chunkwise
            GDN-2 form. Default: ``64``.
        ced_gdn2_expand_v: GDN-2 value-dim expansion relative to the key
            head dim (``head_dim_v = head_dim_k * expand_v``). Default:
            ``1.0``.
        ced_gdn2_num_v_heads: GDN-2 value heads for GVA; ``0`` means
            ``num_attention_heads`` (no expansion). Must be a multiple of
            ``num_attention_heads`` when set. Default: ``0``.
        ced_gdn2_use_short_conv: Depthwise short convs on GDN-2 q/k/v.
            Default: ``True``.
        ced_gdn2_conv_bias: Bias in the GDN-2 short convs. Default: ``False``.
        ced_gdn2_use_qk_l2norm: L2-normalize GDN-2 q/k. Default: ``True``.
        ced_gdn2_allow_neg_eigval: Scale the GDN-2 erase gate to ``[0, 2]``.
            Default: ``False``.
        ced_gdn2_beta_guard: Erase-contrast projection (requires
            ``ced_gdn2_use_qk_l2norm``). Default: ``False``.
        ced_gdn2_use_resting_state: Learn a GDN-2 resting prior the state
            relaxes toward instead of zero. Default: ``False``.
        engram_conv_kernel_size: Kernel width of the Engram short
            depthwise causal conv (used when ``engram_use_conv``).
            Default: ``4``.
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
        titan_memory_depth: Self-referential Titans fast-MLP depth per box.
            Default: ``2``.
        titan_fast_expansion: Titans fast-weight expansion over the head dim.
            Default: ``1.0``.
        titan_objective: Titans inner objective (``"l2"`` or ``"dot"``).
            Default: ``"l2"``.
        titan_use_short_conv: Local short conv on the Titans q/k/v.
            Default: ``True``.
        titan_conv_kernel_size: Kernel width of that Titans short conv.
            Default: ``4``.
        titan_chunk_size: Titans state-carry chunk width; ``0`` means the
            whole sequence. Default: ``0``.
        titan_backend: Backend for Titans fast weights: "naive"/"recurrent"
            (token-by-token recurrence) or "chunk"/"tnt"/"wy_fast" (TNT
            chunkwise parallel training, arXiv:2511.07343). Default: "naive".
        titan_use_qk_projection: In TNT chunkwise mode, project retrieval queries
            onto the running key subspace (arXiv:2511.07343 Sec. 4.1.2).
            Default: False.
        cms_ffn: Dense FFN kind for every Continuum Memory System level
            (``"swiglu"`` or ``"classic_mlp"``). Default: ``"swiglu"``.
        cms_norm: Pre-norm kind for every CMS level
            (``"rmsnorm"``, ``"layernorm"``, or ``"dyT"``). Default: ``"dyT"``.
        dspark_num_drafts: Parallel draft positions (V4.1: 5)
        dspark_window: Drafter window (V4.1: 128)
    """

    vocab_size: int = 32768
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    conv_kernel_size: int = 3
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    cvpe_mlp_hidden: int = 32
    cvpe_use_log_transform: bool = True
    cvpe_enabled: bool = True
    cvpe_max_dims: int = 1
    cvpe_axis_max: tuple[int, ...] = ()
    frvm_enabled: bool = True
    frvm_chunk_size: int = 128
    frvm_ov_mlp_hidden: int = 32
    frvm_use_log_transform: bool = True
    frvm_use_spatial: bool = True
    frvm_cross_heads: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.006
    residual_init_scale: bool = True
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_return_dict: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    num_perceiver_query: int = 6
    window_size: int = 256
    swa_bounded_replay: bool = True
    csa_compression_ratio: int = 4
    window_size_compressed: int = 256 * 4
    window_size_sparse: int = 256 * 8
    window_size_shared: int = 256 * 16
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
    ced_attention_gdn2_parallel: bool = True
    ced_encoder_alternating: bool = True
    ced_attention_variant: str = "compressed_sparse"
    ced_encoder_ffn: str = "swiglu"
    ced_decoder_ffn: str = "swiglu"
    ced_encoder_ffn_intermediate_size: int = 0
    ced_decoder_ffn_intermediate_size: int = 0
    ced_encoder_ffn_update_intervals: tuple[int, ...] = ()
    ced_decoder_ffn_update_intervals: tuple[int, ...] = ()
    ced_decoder_looping: bool = False
    ced_decoder_loop_steps: int = 4
    ced_decoder_loop_exit_threshold: float = 0.5
    ced_decoder_hidden_size: int = 0
    ced_gdn2_backend: str = "naive"
    ced_gdn2_chunk_size: int = 64
    ced_gdn2_expand_v: float = 1.0
    ced_gdn2_num_v_heads: int = 0
    ced_gdn2_use_short_conv: bool = True
    ced_gdn2_conv_bias: bool = False
    ced_gdn2_use_qk_l2norm: bool = True
    ced_gdn2_allow_neg_eigval: bool = False
    ced_gdn2_beta_guard: bool = False
    ced_gdn2_use_resting_state: bool = False
    engram_conv_kernel_size: int = 4
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
    titan_memory_depth: int = 2
    titan_fast_expansion: float = 1.0
    titan_objective: str = "l2"
    titan_use_short_conv: bool = True
    titan_conv_kernel_size: int = 4
    titan_chunk_size: int = 0
    titan_backend: str = "naive"
    titan_use_qk_projection: bool = False
    cms_ffn: str = "swiglu"
    cms_norm: str = "dyT"
    seed: int = 0
    dspark_num_drafts: int = 5
    dspark_window: int = 128

    def __post_init__(self):
        """Validate invariant bounds across configuration fields."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be positive")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers ({self.num_layers}) must be positive")
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
        if self.cvpe_max_dims <= 0:
            raise ValueError(f"cvpe_max_dims ({self.cvpe_max_dims}) must be positive")
        if self.cvpe_mlp_hidden <= 0:
            raise ValueError(f"cvpe_mlp_hidden ({self.cvpe_mlp_hidden}) must be positive")
        if self.frvm_chunk_size <= 0:
            raise ValueError(f"frvm_chunk_size ({self.frvm_chunk_size}) must be positive")
        if self.frvm_ov_mlp_hidden <= 0:
            raise ValueError(f"frvm_ov_mlp_hidden ({self.frvm_ov_mlp_hidden}) must be positive")
        if self.cvpe_axis_max:
            if len(self.cvpe_axis_max) != self.cvpe_max_dims:
                raise ValueError(f"cvpe_axis_max must hold {self.cvpe_max_dims} entries (one per axis), got {tuple(self.cvpe_axis_max)}")
            if any(c <= 0 for c in self.cvpe_axis_max):
                raise ValueError(f"cvpe_axis_max entries must be positive, got {tuple(self.cvpe_axis_max)}")
        if self.conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size ({self.conv_kernel_size}) must be positive")
        if self.conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size ({self.conv_kernel_size}) must be odd to preserve length with padding kernel//2")
        if self.num_perceiver_query <= 0:
            raise ValueError(f"num_perceiver_query ({self.num_perceiver_query}) must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings ({self.max_position_embeddings}) must be positive")
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
        if self.ced_encoder_ffn not in ("swiglu",):
            raise ValueError(f"ced_encoder_ffn ({self.ced_encoder_ffn!r}) must be 'swiglu'")
        if self.ced_decoder_ffn not in ("swiglu",):
            raise ValueError(f"ced_decoder_ffn ({self.ced_decoder_ffn!r}) must be 'swiglu'")
        if self.ced_encoder_ffn_intermediate_size < 0:
            raise ValueError(f"ced_encoder_ffn_intermediate_size ({self.ced_encoder_ffn_intermediate_size}) must be non-negative (0 = intermediate_size)")
        if self.ced_decoder_ffn_intermediate_size < 0:
            raise ValueError(f"ced_decoder_ffn_intermediate_size ({self.ced_decoder_ffn_intermediate_size}) must be non-negative (0 = intermediate_size)")
        if self.ced_encoder_ffn_update_intervals and len(self.ced_encoder_ffn_update_intervals) != self.ced_num_encoder_layers:
            raise ValueError(f"ced_encoder_ffn_update_intervals must hold {self.ced_num_encoder_layers} entries (one per encoder layer), got {tuple(self.ced_encoder_ffn_update_intervals)}")
        if any(c < 1 for c in self.ced_encoder_ffn_update_intervals):
            raise ValueError(f"ced_encoder_ffn_update_intervals must hold positive ints, got {tuple(self.ced_encoder_ffn_update_intervals)}")
        dec_n = 1 if self.ced_decoder_looping else self.ced_num_decoder_layers
        if self.ced_decoder_ffn_update_intervals and len(self.ced_decoder_ffn_update_intervals) != dec_n:
            raise ValueError(
                f"ced_decoder_ffn_update_intervals must hold {dec_n} entries (one per decoder layer; the looped decoder "
                "shares one CSA2 -> DenseFFN layer), got {tuple(self.ced_decoder_ffn_update_intervals)}"
            )
        if any(c < 1 for c in self.ced_decoder_ffn_update_intervals):
            raise ValueError(f"ced_decoder_ffn_update_intervals must hold positive ints, got {tuple(self.ced_decoder_ffn_update_intervals)}")
        if self.ced_decoder_loop_steps < 1:
            raise ValueError(f"ced_decoder_loop_steps ({self.ced_decoder_loop_steps}) must be >= 1")
        if not 0.0 <= self.ced_decoder_loop_exit_threshold <= 1.0:
            raise ValueError(f"ced_decoder_loop_exit_threshold ({self.ced_decoder_loop_exit_threshold}) must be in [0, 1]")
        if self.ced_decoder_hidden_size < 0:
            raise ValueError(f"ced_decoder_hidden_size ({self.ced_decoder_hidden_size}) must be non-negative (0 = hidden_size)")
        if self.ced_decoder_hidden_size and self.ced_decoder_hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"ced_decoder_hidden_size ({self.ced_decoder_hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        if self.ced_gdn2_backend not in ("naive", "recurrent", "chunk", "wy_fast"):
            raise ValueError(f"ced_gdn2_backend ({self.ced_gdn2_backend!r}) must be one of 'naive', 'recurrent', 'chunk', 'wy_fast'")
        if self.ced_gdn2_chunk_size <= 0:
            raise ValueError(f"ced_gdn2_chunk_size ({self.ced_gdn2_chunk_size}) must be positive")
        if self.ced_gdn2_expand_v <= 0:
            raise ValueError(f"ced_gdn2_expand_v ({self.ced_gdn2_expand_v}) must be positive")
        if self.ced_gdn2_num_v_heads < 0:
            raise ValueError(f"ced_gdn2_num_v_heads ({self.ced_gdn2_num_v_heads}) must be non-negative (0 = num_attention_heads)")
        if self.ced_gdn2_num_v_heads and self.ced_gdn2_num_v_heads % self.num_attention_heads != 0:
            raise ValueError(f"ced_gdn2_num_v_heads ({self.ced_gdn2_num_v_heads}) must be a multiple of num_attention_heads ({self.num_attention_heads})")
        if self.engram_conv_kernel_size <= 0:
            raise ValueError(f"engram_conv_kernel_size ({self.engram_conv_kernel_size}) must be positive")
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
        if self.titan_memory_depth < 1:
            raise ValueError(f"titan_memory_depth ({self.titan_memory_depth}) must be >= 1")
        if self.titan_fast_expansion <= 0:
            raise ValueError(f"titan_fast_expansion ({self.titan_fast_expansion}) must be positive")
        if self.titan_objective not in ("l2", "dot"):
            raise ValueError(f"titan_objective ({self.titan_objective!r}) must be 'l2' or 'dot'")
        if self.titan_conv_kernel_size <= 0:
            raise ValueError(f"titan_conv_kernel_size ({self.titan_conv_kernel_size}) must be positive")
        if self.titan_chunk_size < 0:
            raise ValueError(f"titan_chunk_size ({self.titan_chunk_size}) must be non-negative (0 = whole sequence)")
        if self.titan_backend not in ("naive", "recurrent", "chunk", "tnt", "wy_fast"):
            raise ValueError(f"titan_backend ({self.titan_backend!r}) must be one of 'naive', 'recurrent', 'chunk', 'tnt', 'wy_fast'")
        if self.cms_ffn not in ("swiglu", "classic_mlp"):
            raise ValueError(f"cms_ffn ({self.cms_ffn!r}) must be 'swiglu' or 'classic_mlp'")
        if self.cms_norm not in ("rmsnorm", "layernorm", "dyT"):
            raise ValueError(f"cms_norm ({self.cms_norm!r}) must be one of 'rmsnorm', 'layernorm', 'dyT'")
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
