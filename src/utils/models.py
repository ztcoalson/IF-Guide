"""
Utilities for loading models.
"""
import os
import yaml
import logging
import torch
import torch.nn as nn

from transformers.pytorch_utils import Conv1D
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

REGISTRY_PATH = "utils/registry.yaml"

if not os.path.exists(REGISTRY_PATH):
    raise ValueError(f"Registry file {REGISTRY_PATH} not found.")

with open(REGISTRY_PATH, "r") as f:
    REGISTRY = yaml.safe_load(f)


def get_registry_config(model_name: str):
    """
    Get the configuration for a specific model (and its tokenizer) from the registry.

    Args:
        model_name (str): The name of the model to retrieve.
    """
    if model_name not in REGISTRY['models']:
        raise ValueError(f"Model {model_name} not found in registry.")

    model_config = REGISTRY['models'][model_name]
    model_config['name'] = model_name

    if model_config['tokenizer'] not in REGISTRY['tokenizers']:
        raise ValueError(f"Tokenizer {model_config['tokenizer']} not found in registry.")
    
    tokenizer_config = REGISTRY['tokenizers'][model_config['tokenizer']]
    tokenizer_config['name'] = model_config['tokenizer']
    
    return {
        "model": model_config,
        "tokenizer": tokenizer_config
    }


@torch.no_grad()
def replace_conv1d_modules(model: nn.Module) -> int:
    changed_modules = 0
    for name, module in model.named_children():
        # Recurse into child modules
        if len(list(module.children())) > 0:
            changed_modules += replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            changed_modules += 1
            new_module = nn.Linear(in_features=module.weight.shape[0], out_features=module.weight.shape[1])
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)

    return changed_modules


def construct_hf_model(
    registry_config: dict, 
    checkpoint_dir: str = None,
) -> PreTrainedModel:
    """
    Constructs a HuggingFace model with Kronfluence compatability.

    Args:
        config (dict): Configuration dictionary containing model and tokenizer information.
        checkpoint_dir (str): Path to a checkpoint directory to load.
    """
    model_kwargs = registry_config['model']['kwargs']

    if checkpoint_dir:
        if not os.path.isdir(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist.")

        logging.info(f"Loading model from checkpoint: {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
            trust_remote_code=True,
            **model_kwargs,
        )
    elif not registry_config['model']['pretrained']:
        logging.info(f"Loading {registry_config['model']['name']} from scratch")
        model_config = AutoConfig.from_pretrained(
            registry_config['model']['path'],
            trust_remote_code=True,
        )
        model_config.vocab_size = registry_config['tokenizer']['vocab_size']
        model = AutoModelForCausalLM.from_config(
            config=model_config,
            trust_remote_code=True,
            **model_kwargs,
        )
    else:
        logging.info(f"Loading pretrained {registry_config['model']['name']} from HuggingFace hub")
        model = AutoModelForCausalLM.from_pretrained(
            registry_config['model']['path'],
            trust_remote_code=True,
            **model_kwargs,
        )

    # check if vocab size is the same
    if model.config.vocab_size != registry_config['tokenizer']['vocab_size']:
        raise ValueError(f"Vocab size mismatch: actual = {model.config.vocab_size} vs expected = {registry_config['tokenizer']['vocab_size']}")
    
    # Replace Conv1D modules with Linear modules for Kronfluence compatability
    num_changed_modules = replace_conv1d_modules(model)
    if num_changed_modules > 0:
        logging.info(f"Replaced {num_changed_modules} Conv1D modules with Linear modules for Kronfluence compatability")

    return model