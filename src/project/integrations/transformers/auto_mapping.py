"""Auto-class registration so AutoModelForCausalLM.from_pretrained("project") works."""

from transformers import AutoConfig, AutoModelForCausalLM

from project.integrations.transformers.configuration_project import CustomResearchConfig
from project.integrations.transformers.modeling_project import CustomModelForCausalLM


def register_auto_classes():
    AutoConfig.register("project", CustomResearchConfig)
    AutoModelForCausalLM.register(CustomResearchConfig, CustomModelForCausalLM)
