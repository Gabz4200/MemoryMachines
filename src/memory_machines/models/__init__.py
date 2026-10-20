"""Model package: importing it triggers the ``@register_model`` side effects."""

from memory_machines.models.backbone import BLTBackbone, CheapLLM, ExampleBackbone
from memory_machines.models.registry import get_model, register_model

__all__ = ["BLTBackbone", "CheapLLM", "ExampleBackbone", "get_model", "register_model"]
