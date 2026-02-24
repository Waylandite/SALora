"""
Architecture Search Module for SALora

This module defines the architecture parameters (alphas) for rank selection
across all modules (Q, K, V, O, FFN gate/up/down)
"""

import torch
import torch.nn as nn
from typing import Dict, List


class SALoraArchitecture(nn.Module):
    """
    Architecture parameters for SALora

    Manages alpha parameters for all modules across all layers.
    For a Transformer with n_layers:
    - Each layer has 7 modules: Q, K, V, O, gate, up, down
    - Each module has r_max possible ranks
    """

    # Module names
    MODULE_NAMES = ['query', 'key', 'value', 'output', 'gate', 'up', 'down']

    def __init__(
        self,
        n_layers: int,
        r_max: int,
        init_strategy: str = 'uniform'
    ):
        """
        Args:
            n_layers: Number of transformer layers
            r_max: Maximum rank for each module
            init_strategy: How to initialize alphas ('uniform', 'high_rank', 'low_rank')
        """
        super().__init__()
        self.n_layers = n_layers
        self.r_max = r_max
        self.n_modules = len(self.MODULE_NAMES)

        # Alpha parameters: [n_layers, n_modules, r_max]
        # Each alpha[i, j, :] controls the rank distribution for module j in layer i
        if init_strategy == 'uniform':
            init_val = torch.zeros(n_layers, self.n_modules, r_max)
        elif init_strategy == 'high_rank':
            # Initialize to favor higher ranks
            init_val = torch.linspace(-1, 1, r_max).unsqueeze(0).unsqueeze(0)
            init_val = init_val.expand(n_layers, self.n_modules, r_max)
        elif init_strategy == 'low_rank':
            # Initialize to favor lower ranks
            init_val = torch.linspace(1, -1, r_max).unsqueeze(0).unsqueeze(0)
            init_val = init_val.expand(n_layers, self.n_modules, r_max)
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        self.alphas = nn.Parameter(init_val.clone())

    def forward(self) -> torch.Tensor:
        """
        Returns:
            Alpha parameters [n_layers, n_modules, r_max]
        """
        return self.alphas

    def get_alpha_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get alphas organized by layer and module name

        Returns:
            Dictionary: {f"layer.{i}.{module_name}": alpha_tensor}
        """
        alpha_dict = {}
        for layer_idx in range(self.n_layers):
            for module_idx, module_name in enumerate(self.MODULE_NAMES):
                key = f"layer.{layer_idx}.{module_name}"
                alpha_dict[key] = self.alphas[layer_idx, module_idx, :]
        return alpha_dict

    def get_effective_ranks(self, threshold: float = None) -> Dict[str, int]:
        """
        Compute effective rank for each module based on current alphas

        Args:
            threshold: If None, uses 1/r_max as threshold.
                      Ranks with softmax(alpha) >= threshold are counted.

        Returns:
            Dictionary: {f"layer.{i}.{module_name}": effective_rank}
        """
        if threshold is None:
            threshold = 1.0 / self.r_max

        effective_ranks = {}
        alphas_softmax = torch.softmax(self.alphas, dim=-1)  # [n_layers, n_modules, r_max]

        for layer_idx in range(self.n_layers):
            for module_idx, module_name in enumerate(self.MODULE_NAMES):
                key = f"layer.{layer_idx}.{module_name}"
                alpha_sm = alphas_softmax[layer_idx, module_idx, :]
                effective_rank = int(torch.sum(alpha_sm >= threshold).item())
                effective_ranks[key] = effective_rank

        return effective_ranks

    def get_rank_summary(self) -> Dict[str, List[int]]:
        """
        Get a summary of effective ranks by module type

        Returns:
            Dictionary: {module_name: [rank_layer_0, rank_layer_1, ...]}
        """
        effective_ranks = self.get_effective_ranks()
        summary = {name: [] for name in self.MODULE_NAMES}

        for layer_idx in range(self.n_layers):
            for module_name in self.MODULE_NAMES:
                key = f"layer.{layer_idx}.{module_name}"
                summary[module_name].append(effective_ranks[key])

        return summary

    def prune_low_ranks(self, min_rank: int = 1):
        """
        Prune modules with very low effective ranks

        This is a post-processing step to enforce minimum ranks
        """
        with torch.no_grad():
            effective_ranks = self.get_effective_ranks()
            for key, rank in effective_ranks.items():
                if rank < min_rank:
                    # Parse the key to get indices
                    parts = key.split('.')
                    layer_idx = int(parts[1])
                    module_name = parts[2]
                    module_idx = self.MODULE_NAMES.index(module_name)

                    # Set the first min_rank alphas to high values
                    self.alphas[layer_idx, module_idx, :min_rank] = 10.0
                    self.alphas[layer_idx, module_idx, min_rank:] = -10.0


class ArchitectureSearchEngine:
    """
    Engine for managing architecture search with spectral constraints
    """

    def __init__(
        self,
        architecture: SALoraArchitecture,
        lambda_spectral: float = 1e-4,
        lambda_l1: float = 1e-3,
    ):
        """
        Args:
            architecture: SALoraArchitecture instance
            lambda_spectral: Weight for spectral intrusion penalty
            lambda_l1: Weight for L1 regularization
        """
        self.architecture = architecture
        self.lambda_spectral = lambda_spectral
        self.lambda_l1 = lambda_l1

    def compute_regularization(
        self,
        spectral_score: torch.Tensor,
        l1_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total regularization loss

        Args:
            spectral_score: Spectral intrusion score
            l1_score: L1 regularization score

        Returns:
            Total regularization loss
        """
        return self.lambda_spectral * spectral_score + self.lambda_l1 * l1_score
