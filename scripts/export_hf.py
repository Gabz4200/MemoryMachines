"""Export trained weights to Hugging Face format."""

import argparse
from typing import Any, cast

from project.integrations.transformers.configuration_project import CustomResearchConfig
from project.integrations.transformers.modeling_project import CustomModelForCausalLM
from project.modules.lightning_module import LitResearchSystem


def export(checkpoint_path: str, output_dir: str):
    # Load trained weights from the Lightning checkpoint.
    system = LitResearchSystem.load_from_checkpoint(checkpoint_path)
    # nn.Module.__getattr__ is typed Module | Tensor, so the casts go through
    # Any: int() on Module | Tensor is itself a type error.
    backbone = cast(Any, system.model)
    config = CustomResearchConfig(
        embed_dim=int(cast(Any, backbone.embed_dim)),
        num_layers=int(cast(Any, backbone.num_layers)),
    )
    hf_model = CustomModelForCausalLM(config)
    hf_model.backbone.load_state_dict(system.model.state_dict())
    hf_model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    print(f"Exported Hugging Face model to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)
