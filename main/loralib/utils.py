"""
Utility functions for SALora
"""

import torch
import torch.nn as nn
from typing import Dict


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """
    Extract LoRA parameters from model state dict

    Args:
        model: The model
        bias: Which biases to include ('none', 'all', 'lora_only')

    Returns:
        Dictionary of LoRA parameters
    """
    state_dict = model.state_dict()
    if bias == 'none':
        return {k: v for k, v in state_dict.items() if 'lora_' in k}
    elif bias == 'all':
        return {
            k: v for k, v in state_dict.items()
            if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        for k, v in state_dict.items():
            if 'lora_' in k:
                to_return[k] = v
            elif 'bias' in k:
                # Check if this bias belongs to a LoRA layer
                prefix = k.rsplit('.bias', 1)[0]
                if any(f'{prefix}.lora_' in key for key in state_dict.keys()):
                    to_return[k] = v
        return to_return
    else:
        raise NotImplementedError(f"Bias mode {bias} not implemented")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model

    Args:
        model: The model
        trainable_only: Count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
