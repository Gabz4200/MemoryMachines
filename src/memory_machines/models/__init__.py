"""Model package: importing it triggers the ``@register_model`` side effects."""

from memory_machines.models.backbone import ExampleBackbone, CheapLLM
from memory_machines.models.registry import get_model, register_model

__all__ = ["ExampleBackbone", "CheapLLM", "get_model", "register_model"]
