"""
SALora-enabled Model Wrapper

This module provides a wrapper to inject SALora into Qwen2.5-Coder models.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Dict, Optional, List
import sys
import os

# Add parent directory to path to import loralib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loralib


def inject_salora_to_qwen(
    model,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    modules_to_replace: list = None,
):
    """
    Inject SALora layers into a Qwen2.5 model

    Args:
        model: Pretrained Qwen2.5 model
        r: Rank of LoRA
        lora_alpha: Alpha parameter for scaling
        lora_dropout: Dropout rate
        modules_to_replace: List of module names to replace
                           Default: all attention and MLP modules

    Returns:
        Modified model with SALora layers, and a dict of LoRA module names
    """
    if modules_to_replace is None:
        # Default: replace all linear layers in attention and MLP
        # Qwen2 architecture:
        # - Attention: q_proj, k_proj, v_proj, o_proj
        # - MLP: gate_proj, up_proj, down_proj
        modules_to_replace = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj']

    lora_module_names = {}

    # Get the base model
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model

    # Iterate through decoder layers
    if hasattr(base_model, 'layers'):
        layers = base_model.layers
        for layer_idx, layer in enumerate(layers):
            # Replace attention layers
            attention = layer.self_attn

            # Q, K, V, O projections
            for module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if module_name in modules_to_replace:
                    old_module = getattr(attention, module_name)
                    new_module = loralib.Linear(
                        in_features=old_module.in_features,
                        out_features=old_module.out_features,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=old_module.bias is not None,
                    )
                    # Copy pre-trained weights
                    new_module.weight.data = old_module.weight.data.clone()
                    if old_module.bias is not None:
                        new_module.bias.data = old_module.bias.data.clone()

                    setattr(attention, module_name, new_module)

                    # Map to our naming convention
                    name_map = {
                        'q_proj': 'query',
                        'k_proj': 'key',
                        'v_proj': 'value',
                        'o_proj': 'output'
                    }
                    standard_name = name_map.get(module_name, module_name)
                    lora_module_names[f"layer.{layer_idx}.{standard_name}"] = new_module

            # Replace MLP layers
            mlp = layer.mlp
            for module_name in ['gate_proj', 'up_proj', 'down_proj']:
                if module_name in modules_to_replace:
                    old_module = getattr(mlp, module_name)
                    new_module = loralib.Linear(
                        in_features=old_module.in_features,
                        out_features=old_module.out_features,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=old_module.bias is not None,
                    )
                    # Copy pre-trained weights
                    new_module.weight.data = old_module.weight.data.clone()
                    if old_module.bias is not None:
                        new_module.bias.data = old_module.bias.data.clone()

                    setattr(mlp, module_name, new_module)

                    # Map to our naming convention
                    name_map = {
                        'gate_proj': 'gate',
                        'up_proj': 'up',
                        'down_proj': 'down'
                    }
                    standard_name = name_map[module_name]
                    lora_module_names[f"layer.{layer_idx}.{standard_name}"] = new_module

    # Mark only LoRA parameters as trainable
    loralib.mark_only_lora_as_trainable(model, bias='none')

    return model, lora_module_names


def create_salora_model(
    model_name: str,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    task_type: str = 'causal_lm',
    modules_to_replace: list = None,
):
    """
    Create a Qwen2.5-Coder model with SALora injected

    Args:
        model_name: Name of the pretrained model (should be a Qwen2.5-Coder model)
        r: Rank of LoRA
        lora_alpha: Alpha scaling parameter
        lora_dropout: Dropout rate
        task_type: Type of task (only 'causal_lm' supported)
        modules_to_replace: List of modules to replace with LoRA

    Returns:
        model, lora_module_names, n_layers
    """
    # Load pretrained model
    if task_type != 'causal_lm':
        raise ValueError(f"Only 'causal_lm' task type is supported for code tasks, got: {task_type}")

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Inject SALora to Qwen model
    model, lora_module_names = inject_salora_to_qwen(
        model, r=r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, modules_to_replace=modules_to_replace
    )

    # Get number of layers
    if hasattr(model, 'model'):
        n_layers = len(model.model.layers)
    else:
        n_layers = len(model.layers)

    return model, lora_module_names, n_layers


def save_lora_config_for_peft(
    architecture,
    lora_module_names: Dict[str, nn.Module],
    save_path: str,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: List[str] = None,
    model_type: str = "qwen2",
):
    """
    Save the searched LoRA configuration for PEFT with rank_pattern support

    PEFT 0.7.0+ supports rank_pattern for per-layer per-module rank specification!
    We generate both the full rank_pattern and compressed fallback strategies.

    Args:
        architecture: SALoraArchitecture instance with optimized alphas
        lora_module_names: Dictionary of LoRA modules
        save_path: Path to save the configuration
        r: Maximum rank used in search
        lora_alpha: Alpha parameter
        target_modules: List of target module names
        model_type: Model type ("qwen2" or "llama") for correct naming
    """
    import json
    import numpy as np

    # Get effective ranks for each module
    effective_ranks = architecture.get_effective_ranks()
    rank_summary = architecture.get_rank_summary()

    # Get number of layers
    n_layers = len(rank_summary[list(rank_summary.keys())[0]])

    # ============= Generate rank_pattern for PEFT =============
    # Map SALora module names to PEFT module paths
    module_name_mapping = {
        "qwen2": {
            "query": "self_attn.q_proj",
            "key": "self_attn.k_proj",
            "value": "self_attn.v_proj",
            "output": "self_attn.o_proj",
            "gate": "mlp.gate_proj",
            "up": "mlp.up_proj",
            "down": "mlp.down_proj",
        },
        "llama": {
            "query": "self_attn.q_proj",
            "key": "self_attn.k_proj",
            "value": "self_attn.v_proj",
            "output": "self_attn.o_proj",
            "gate": "mlp.gate_proj",
            "up": "mlp.up_proj",
            "down": "mlp.down_proj",
        },
    }

    mapping = module_name_mapping.get(model_type, module_name_mapping["qwen2"])

    # Generate rank_pattern
    rank_pattern = {}
    for key, rank in effective_ranks.items():
        # Parse: "layer.0.query" -> layer=0, module=query
        parts = key.split('.')
        layer_idx = int(parts[1])
        module_type = parts[2]

        # Skip if module not in mapping (e.g., gate for RoBERTa)
        if module_type not in mapping:
            continue

        # Convert to PEFT format: "model.layers.0.self_attn.q_proj"
        peft_key = f"model.layers.{layer_idx}.{mapping[module_type]}"
        rank_pattern[peft_key] = rank

    # ============= Compression Strategies (fallback) =============

    # Strategy 1: Global median (across all layers and modules)
    all_ranks_list = [r for ranks in rank_summary.values() for r in ranks]
    global_median_rank = int(np.median(all_ranks_list))
    global_mean_rank = int(np.mean(all_ranks_list))

    # Strategy 2: Per-module-type median
    module_type_medians = {}
    module_type_means = {}
    for module_type, ranks in rank_summary.items():
        module_type_medians[module_type] = int(np.median(ranks))
        module_type_means[module_type] = int(np.mean(ranks))

    # Strategy 3: Layer-group-wise
    early_layers = list(range(0, n_layers // 3))
    middle_layers = list(range(n_layers // 3, 2 * n_layers // 3))
    late_layers = list(range(2 * n_layers // 3, n_layers))

    def get_layer_group_ranks(layer_indices):
        group_ranks = []
        for module_type, ranks in rank_summary.items():
            for idx in layer_indices:
                if idx < len(ranks):
                    group_ranks.append(ranks[idx])
        return int(np.median(group_ranks)) if group_ranks else global_median_rank

    early_median = get_layer_group_ranks(early_layers)
    middle_median = get_layer_group_ranks(middle_layers)
    late_median = get_layer_group_ranks(late_layers)

    # ============= PEFT Config with rank_pattern =============

    peft_config_with_pattern = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "rank_pattern": rank_pattern,  # Full per-layer per-module ranks!
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules if target_modules else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "description": "Full per-layer per-module rank configuration using rank_pattern (PEFT 0.7.0+)"
    }

    # Fallback config (single global rank)
    peft_config_fallback = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": global_median_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules if target_modules else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "description": "Fallback: single global rank (for older PEFT versions)"
    }

    # ============= Full Configuration =============

    detailed_config = {
        "notice": "✅ This configuration uses rank_pattern (PEFT 0.7.0+) for full per-layer per-module ranks!",
        "version": "2.0",  # Version with rank_pattern support

        # Primary config: Full rank_pattern
        "peft_config": peft_config_with_pattern,

        # Fallback config for older PEFT versions
        "peft_config_fallback": peft_config_fallback,

        # rank_pattern in standard format for direct use
        "rank_pattern": rank_pattern,

        # Compression strategies (for analysis/fallback)
        "compression_strategies": {
            "global_median": global_median_rank,
            "global_mean": global_mean_rank,
            "module_type_medians": module_type_medians,
            "module_type_means": module_type_means,
            "layer_groups": {
                "early_layers": {"indices": early_layers, "median_rank": early_median},
                "middle_layers": {"indices": middle_layers, "median_rank": middle_median},
                "late_layers": {"indices": late_layers, "median_rank": late_median},
            }
        },

        # Full SALora search results (per-layer per-module)
        "salora_full_config": {
            "layer_module_ranks": effective_ranks,
            "rank_summary_by_module_type": rank_summary,
            "n_layers": n_layers,
            "model_type": model_type,
            "search_params": {
                "max_rank": r,
                "lora_alpha": lora_alpha,
            }
        },
    }

    # Save to file
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(detailed_config, f, indent=2)

    # Also save rank_pattern separately for easy loading
    rank_pattern_path = save_path.replace('.json', '_rank_pattern.json')
    with open(rank_pattern_path, 'w') as f:
        json.dump(rank_pattern, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ SALora Configuration Saved with rank_pattern Support!")
    print(f"{'='*70}")
    print(f"\nFiles saved:")
    print(f"  1. {save_path}")
    print(f"  2. {rank_pattern_path} (rank_pattern only)")

    print(f"\nFull configuration: {len(rank_pattern)} layer-module combinations")
    print(f"  Layers: {n_layers}")
    print(f"  Modules per layer: {len(rank_summary)}")

    print(f"\nrank_pattern preview (first 5 entries):")
    for i, (key, rank) in enumerate(list(rank_pattern.items())[:5]):
        print(f"  {key}: {rank}")
    print(f"  ...")

    print(f"\nPer-module-type statistics:")
    for module_type, median_rank in module_type_medians.items():
        mean_rank = module_type_means[module_type]
        ranks = rank_summary[module_type]
        print(f"  {module_type:8s}: median={median_rank}, mean={mean_rank:.1f}, range=[{min(ranks)}, {max(ranks)}]")

    print(f"\nLayer groups:")
    print(f"  Early  (layers {early_layers[0]:2d}-{early_layers[-1]:2d}): median={early_median}")
    print(f"  Middle (layers {middle_layers[0]:2d}-{middle_layers[-1]:2d}): median={middle_median}")
    print(f"  Late   (layers {late_layers[0]:2d}-{late_layers[-1]:2d}): median={late_median}")

    print(f"\n{'='*70}")
    print(f"Usage:")
    print(f"  1. Use with PEFT 0.7.0+:")
    print(f"     config = LoraConfig(**peft_config)")
    print(f"     # Will use full rank_pattern automatically!")
    print(f"\n  2. Load rank_pattern directly:")
    print(f"     with open('{rank_pattern_path}') as f:")
    print(f"         rank_pattern = json.load(f)")
    print(f"     config = LoraConfig(rank_pattern=rank_pattern, ...)")
    print(f"{'='*70}\n")

    return peft_config_with_pattern, rank_pattern

