"""Structured Config schemas. Validate cross-group invariants at boot, before GPU init."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    embed_dim: int = 256
    num_layers: int = 6
    backend: str = "naive"


@dataclass
class MemoryMachineConfig:  # todo: Use actual necessary values (this is copying LFM2 for now)
    """Configuration class for MemoryMachine model.

    Attributes:
        vocab_size: Size of the vocabulary
        hidden_size: Hidden dimension size
        intermediate_size: Feedforward intermediate dimension
        num_conv_blocks: Number of LIV convolution blocks (default: 10)
        num_attention_blocks: Number of GQA blocks (default: 6)
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads for GQA
        conv_kernel_size: Kernel size for short convolutions
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: RMS normalization epsilon
        tie_word_embeddings: Whether to tie input/output embeddings
        rope_theta: RoPE base frequency
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
    """

    vocab_size: int = 32768
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_conv_blocks: int = 10
    num_attention_blocks: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    conv_kernel_size: int = 3
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
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

    @classmethod
    def from_pretrained_size(cls, size: str) -> "MemoryMachineConfig":
        """Create configuration for standard MemoryMachine model sizes.

        Args:
            size: Model size ('350M', '700M', or '1.2B')

        Returns:
            MemoryMachineConfig instance for the specified size
        """
        configs = {
            "350M": cls(
                hidden_size=768,
                intermediate_size=2048,
                num_attention_heads=12,
                num_key_value_heads=3,
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
