"""
Spectral Intrusion Metric for SALora

This module implements the spectral health constraint that measures
how much LoRA updates deviate from the pre-trained model's knowledge manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SpectralIntrusionMetric(nn.Module):
    """
    Compute spectral intrusion score for LoRA updates

    The intrusion score measures how much the LoRA update deviates from
    the pre-trained weight's principal singular subspace.

    R_spec = Σ_{l,j} α_{l,j} · ||P_l^⊥ u_{l,j}||_2^2

    where P_l^⊥ projects onto the space orthogonal to the top-k singular
    vectors of the pre-trained weight W_0.
    """

    def __init__(
        self,
        keep_ratio: float = 0.1,
        device: str = 'cuda',
    ):
        """
        Args:
            keep_ratio: Ratio of singular vectors to keep (default: 0.1 = top 10%)
            device: Device to store projection matrices
        """
        super().__init__()
        self.keep_ratio = keep_ratio
        self.device = device
        self.projection_matrices = {}  # Store P^⊥ for each layer

    def register_pretrained_weights(
        self,
        model: nn.Module,
        module_names: List[str]
    ):
        """
        Pre-compute projection matrices for specified modules

        Args:
            model: The model with pre-trained weights
            module_names: List of module names to register (e.g., ['layer.0.query', ...])
        """
        self.projection_matrices = {}

        for name in module_names:
            # Get the module
            module = self._get_module_by_name(model, name)
            if module is None or not hasattr(module, 'weight'):
                continue

            W0 = module.weight.data  # [out_features, in_features]

            # Compute SVD
            try:
                U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
                # U: [out_features, min(out, in)]
                # S: [min(out, in)]
                # Vh: [min(out, in), in_features]

                # Keep top-k singular vectors
                k = max(1, int(self.keep_ratio * len(S)))
                U_k = U[:, :k]  # [out_features, k]

                # Compute projection matrix P^⊥ = I - U_k @ U_k^T
                # We don't store the full matrix, just U_k
                self.projection_matrices[name] = U_k.to(self.device)

            except Exception as e:
                print(f"Warning: Failed to compute SVD for {name}: {e}")
                continue

    def _get_module_by_name(self, model: nn.Module, name: str):
        """Get module by its name"""
        parts = name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def compute_intrusion_score(
        self,
        lora_modules: Dict[str, nn.Module],
        alphas: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the spectral intrusion score

        Args:
            lora_modules: Dictionary mapping module names to LoRA modules
            alphas: Dictionary mapping module names to alpha tensors [r]

        Returns:
            Intrusion score (scalar tensor)
        """
        total_score = 0.0

        for name, module in lora_modules.items():
            if name not in self.projection_matrices:
                continue

            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            U_k = self.projection_matrices[name]  # [out_features, k]
            lora_A = module.lora_A  # [r, in_features]
            lora_B = module.lora_B  # [out_features, r]

            alpha = alphas.get(name, None)
            if alpha is None:
                continue

            # Apply softmax to alpha
            alpha_weights = F.softmax(alpha, dim=0)  # [r]

            # For each rank component j, compute ||P^⊥ u_j||^2
            # where u_j is the j-th column of lora_B (the left singular vector)
            for j in range(module.r):
                u_j = lora_B[:, j]  # [out_features]

                # Project onto orthogonal space: P^⊥ u_j = u_j - U_k @ (U_k^T @ u_j)
                projection = U_k @ (U_k.T @ u_j)  # [out_features]
                orthogonal_component = u_j - projection

                # Compute squared norm
                norm_sq = torch.sum(orthogonal_component ** 2)

                # Weight by alpha
                total_score += alpha_weights[j] * norm_sq

        return total_score

    def compute_intrusion_score_simplified(
        self,
        lora_modules: Dict[str, nn.Module],
        alphas: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Simplified version: compute ||P^⊥ (B @ A)||_F^2

        This is computationally more efficient and gives similar results.

        Args:
            lora_modules: Dictionary mapping module names to LoRA modules
            alphas: Dictionary mapping module names to alpha tensors [r]

        Returns:
            Intrusion score (scalar tensor)
        """
        total_score = 0.0

        for name, module in lora_modules.items():
            if name not in self.projection_matrices:
                continue

            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            U_k = self.projection_matrices[name]  # [out_features, k]

            # Get the weighted LoRA matrix
            delta_W = module.get_lora_weight_matrix(alphas.get(name))
            if delta_W is None:
                continue

            # delta_W: [out_features, in_features]

            # Project each row of delta_W
            # P^⊥ delta_W = delta_W - U_k @ (U_k^T @ delta_W)
            projection = U_k @ (U_k.T @ delta_W)  # [out_features, in_features]
            orthogonal_component = delta_W - projection

            # Compute Frobenius norm squared
            norm_sq = torch.sum(orthogonal_component ** 2)

            total_score += norm_sq

        return total_score


def compute_l1_regularization(alphas: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute L1 regularization on alpha parameters to induce sparsity

    Args:
        alphas: Dictionary mapping module names to alpha tensors

    Returns:
        L1 norm of all alphas
    """
    total_l1 = 0.0
    for alpha in alphas.values():
        # Apply softmax first to get the actual weights
        alpha_weights = F.softmax(alpha, dim=0)
        total_l1 += torch.sum(torch.abs(alpha_weights))
    return total_l1
