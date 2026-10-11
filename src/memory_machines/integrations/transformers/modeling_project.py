"""HF model adapter: translates HF kwargs <-> typed ModelInput/ModelOutput.

The core backbone stays a pure nn.Module; all HF coupling lives here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from memory_machines.integrations.transformers.configuration_project import CustomResearchConfig
from memory_machines.models.backbone import ExampleBackbone
from memory_machines.models.io import ModelInput, ModelOutput


class CustomModelForCausalLM(PreTrainedModel):
    config_class = CustomResearchConfig
    base_model_prefix = "project"
    _supports_flash_attn_2 = True

    def __init__(self, config: CustomResearchConfig):
        super().__init__(config)
        # Instantiate the pure PyTorch model using config kwargs. The typed_*
        # accessors exist because PretrainedConfig inherits nn.Module, which
        # widens plain attribute access to Module | Tensor.
        self.backbone = ExampleBackbone(
            embed_dim=config.typed_embed_dim,
            num_layers=config.typed_num_layers,
            backend=config.typed_backend,
        )
        self.vocab_size = config.typed_vocab_size
        self.lm_head = nn.Linear(config.typed_embed_dim, self.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.input_proj

    def set_input_embeddings(self, value: nn.Module):
        self.backbone.embed = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Translate HF input -> typed ModelInput.
        clean_input = ModelInput(
            input_ids=input_ids, attention_mask=attention_mask, state=past_key_values
        )

        # 2. Execute pure PyTorch math.
        clean_output: ModelOutput = self.backbone(clean_input)
        logits = self.lm_head(clean_output.logits)

        # 3. Loss if labels provided (HF standard).
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            if clean_output.aux_loss is not None:
                loss = loss + clean_output.aux_loss

        # 4. Translate typed ModelOutput -> HF output.
        if not return_dict:
            output = (logits, clean_output.new_state)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=clean_output.new_state,
        )
