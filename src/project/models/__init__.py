# Importing the modules triggers @register_model decorators.
from project.models.backbone import ExampleBackbone  # noqa: F401
from project.models.registry import get_model, register_model  # noqa: F401
