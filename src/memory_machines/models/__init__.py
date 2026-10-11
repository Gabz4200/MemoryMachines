# Importing the modules triggers @register_model decorators.
from memory_machines.models.backbone import ExampleBackbone  # noqa: F401
from memory_machines.models.registry import get_model, register_model  # noqa: F401
