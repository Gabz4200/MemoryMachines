"""Export trained weights to Hugging Face format.

The checkpoint stores the injected model under a ``model.`` prefix; the
ExampleBackbone architecture is inferred from the checkpoint tensors (the
embedding shape gives ``vocab_size``/``embed_dim``; the block count gives
``num_layers``), so no fragile attribute reads on the model object are needed.
"""

import argparse

import torch

from memory_machines.integrations.transformers.configuration_project import CustomResearchConfig
from memory_machines.integrations.transformers.modeling_project import CustomModelForCausalLM
from memory_machines.models.backbone import ExampleBackbone

_PREFIX = "model."


def _strip_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix(_PREFIX): v for k, v in state_dict.items()}


def _infer_arch(backbone_state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Infer (vocab_size, embed_dim, num_layers) from ExampleBackbone weights."""
    vocab_size, embed_dim = backbone_state["embed.weight"].shape
    layer_ids = {k.split(".")[1] for k in backbone_state if k.startswith("blocks.") and k.split(".")[1].isdigit()}
    return int(vocab_size), int(embed_dim), len(layer_ids)


def export(checkpoint_path: str, output_dir: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone_state = _strip_prefix(ckpt.get("state_dict", ckpt))

    vocab_size, embed_dim, num_layers = _infer_arch(backbone_state)
    backbone = ExampleBackbone(embed_dim=embed_dim, num_layers=num_layers, vocab_size=vocab_size)
    backbone.load_state_dict(backbone_state)

    config = CustomResearchConfig(embed_dim=embed_dim, num_layers=num_layers, vocab_size=vocab_size)
    hf_model = CustomModelForCausalLM(config)
    hf_model.backbone.load_state_dict(backbone.state_dict())
    hf_model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    print(f"Exported Hugging Face model to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)
