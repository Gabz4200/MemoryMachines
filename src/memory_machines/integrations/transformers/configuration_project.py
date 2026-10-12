"""HF config adapter: bridges Hub JSON to the native model's constructor kwargs."""

from transformers import PretrainedConfig


class CustomResearchConfig(PretrainedConfig):
    model_type = "project"

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 6,
        backend: str = "naive",
        vocab_size: int = 50257,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.backend = backend
        self.vocab_size = vocab_size

    # PretrainedConfig inherits nn.Module, so plain attribute access
    # is typed Module | Tensor; these accessors give the declared
    # types back without a cast at each call site.
    @property
    def typed_embed_dim(self) -> int:
        return self.embed_dim

    @property
    def typed_num_layers(self) -> int:
        return self.num_layers

    @property
    def typed_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def typed_backend(self) -> str:
        return self.backend
