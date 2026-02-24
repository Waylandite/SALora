"""
Utility functions for SALora experiments
"""

import torch
import numpy as np
import random
from typing import Dict, Any
import json
import os


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_config(config: Any, path: str):
    """Save configuration to JSON file"""
    config_dict = vars(config) if hasattr(config, '__dict__') else config
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def count_parameters(model, trainable_only=True):
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model,
    architecture,
    optimizer,
    arch_optimizer,
    epoch,
    path,
    **kwargs
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'architecture_state_dict': architecture.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'arch_optimizer_state_dict': arch_optimizer.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, architecture, optimizer=None, arch_optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    architecture.load_state_dict(checkpoint['architecture_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if arch_optimizer is not None:
        arch_optimizer.load_state_dict(checkpoint['arch_optimizer_state_dict'])

    return checkpoint['epoch']


def visualize_ranks(rank_summary: Dict[str, list], save_path: str = None):
    """
    Visualize rank allocation across layers

    Args:
        rank_summary: Dictionary mapping module names to list of ranks
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        module_names = list(rank_summary.keys())
        n_layers = len(rank_summary[module_names[0]])

        for module_name in module_names:
            ranks = rank_summary[module_name]
            ax.plot(range(n_layers), ranks, marker='o', label=module_name)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Rank Allocation across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rank visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualization.")


def compute_parameter_efficiency(
    model,
    architecture,
    total_params: int = None
):
    """
    Compute parameter efficiency metrics

    Args:
        model: Model with LoRA
        architecture: Architecture module
        total_params: Total parameters in original model (optional)

    Returns:
        Dictionary with efficiency metrics
    """
    trainable_params = count_parameters(model, trainable_only=True)
    total_params_with_lora = count_parameters(model, trainable_only=False)

    # Get effective ranks
    effective_ranks = architecture.get_effective_ranks()
    total_rank = sum(effective_ranks.values())
    max_possible_rank = len(effective_ranks) * architecture.r_max

    metrics = {
        'trainable_params': trainable_params,
        'total_params': total_params_with_lora,
        'trainable_ratio': trainable_params / total_params_with_lora,
        'total_effective_rank': total_rank,
        'max_possible_rank': max_possible_rank,
        'rank_utilization': total_rank / max_possible_rank,
    }

    if total_params is not None:
        metrics['params_vs_full_ft'] = trainable_params / total_params

    return metrics


def print_rank_statistics(architecture):
    """Print statistics about rank allocation"""
    rank_summary = architecture.get_rank_summary()

    print("\n" + "=" * 80)
    print("RANK ALLOCATION STATISTICS")
    print("=" * 80)

    for module_name, ranks in rank_summary.items():
        ranks_array = np.array(ranks)
        print(f"\n{module_name.upper()}:")
        print(f"  Mean: {ranks_array.mean():.2f}")
        print(f"  Std:  {ranks_array.std():.2f}")
        print(f"  Min:  {ranks_array.min()}")
        print(f"  Max:  {ranks_array.max()}")
        print(f"  Distribution: {ranks}")

    print("\n" + "=" * 80)
