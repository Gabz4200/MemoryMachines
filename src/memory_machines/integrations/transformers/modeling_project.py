"""HF model adapter: translates HF kwargs <-> typed ModelInput/ModelOutput.

The core backbone stays a pure nn.Module; all HF coupling lives here.
``ExampleBackbone`` returns vocabulary logits from its own ``head``; this
adapter reuses that head as the model's ``lm_head`` (one head, no double
projection) and exposes the backbone embedding so HF resizing/tying works.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
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
        # The backbone owns the single embedding + LM head. ``lm_head`` is a
        # property aliasing ``backbone.head`` so the head tensor is registered
        # once (an alias attribute would register it twice and break safetensors
        # save_pretrained with a shared-tensor error).
        # The typed_* accessors exist because PretrainedConfig inherits
        # nn.Module, which widens plain attribute access to Module | Tensor.
        self.backbone = ExampleBackbone(
            embed_dim=config.typed_embed_dim,
            num_layers=config.typed_num_layers,
            vocab_size=config.typed_vocab_size,
            backend=config.typed_backend,
        )
        self.vocab_size = config.typed_vocab_size
        self.post_init()

    def __getattr__(self, name: str) -> Any:
        if name == "lm_head":
            return self.backbone.head
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # nn.Module.__setattr__ registers Module values as submodules, so the
        # alias must be routed explicitly to keep one registered head tensor.
        modules = self.__dict__.get("_modules")
        if name == "lm_head" and modules is not None and "backbone" in modules:
            self.backbone.head = cast(nn.Linear, value)
            return
        super().__setattr__(name, value)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return cast(nn.Linear, self.lm_head)

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.backbone.head = cast(nn.Linear, new_embeddings)

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
        clean_input = ModelInput(input_ids=input_ids, attention_mask=attention_mask, state=past_key_values)

        # 2. Execute pure PyTorch math (backbone returns vocab logits).
        clean_output: ModelOutput = self.backbone(clean_input)
        logits = clean_output.logits

        # 3. Loss if labels provided (HF standard).
        loss = None
        if labels is not None:
            # Local int: reading self.vocab_size widens to Module | Tensor because
            # PretrainedConfig inherits nn.Module.
            vocab_size = int(self.vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            if clean_output.aux_loss is not None:
                loss = loss + clean_output.aux_loss

        # 4. Translate typed ModelOutput -> HF output.
        if not return_dict:
            output = (logits, clean_output.new_state)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=cast("torch.FloatTensor", logits),
            past_key_values=clean_output.new_state,
        )
