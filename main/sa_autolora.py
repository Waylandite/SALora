"""
SA-AutoLoRA: Spectral-Aware Meta-Learning for Automated Multi-Module Low-Rank Adaptation
Complete Implementation with Betty Bilevel Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm

# Betty for bilevel optimization
try:
    from betty.engine import Engine
    from betty.problems import ImplicitProblem
    from betty.configs import Config, EngineConfig
except ImportError:
    print("Please install betty: pip install betty-ml")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SAAutoLoRAConfig:
    """Configuration for SA-AutoLoRA"""
    # Model architecture
    r_max: int = 8                          # Maximum rank
    target_modules: List[str] = None        # Modules to apply LoRA
    lora_alpha: float = 16.0                # LoRA scaling factor
    lora_dropout: float = 0.0
    
    # Spectral constraint
    k_ratio: float = 0.1                    # SVD truncation ratio
    lambda_spectral: float = 1e-4           # Spectral penalty coefficient
    gamma_l1: float = 1e-3                  # L1 sparsity penalty
    
    # Optimization
    lr_theta: float = 1e-4                  # Learning rate for LoRA parameters
    lr_alpha: float = 1e-2                  # Learning rate for module weights
    weight_decay: float = 0.01
    
    # Training strategy
    warmup_epochs: int = 5                  # Stage 1: train theta
    search_epochs: int = 10                 # Stage 2: train alpha
    val_split: float = 0.1
    
    # Pruning
    alpha_threshold: float = 0.1            # Prune if alpha < threshold
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                   'gate_proj', 'up_proj', 'down_proj']


# ============================================================================
# Spectral-Aware LoRA Layer
# ============================================================================

class SALoRALayer(nn.Module):
    """
    Spectral-Aware Low-Rank Adaptation Layer
    Supports adaptive rank allocation with spectral intrusion penalty
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_max: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        
        # LoRA parameters - shape: (r_max, in/out_features)
        # We create r_max separate rank-1 updates
        self.lora_A = nn.Parameter(torch.zeros(r_max, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r_max))
        
        # Module selection weights - learnable via meta-learning
        self.alpha = nn.Parameter(torch.ones(r_max) * 0.5)
        
        # Scaling factor
        self.scaling = self.lora_alpha / self.r_max
        
        # Dropout
        if lora_dropout > 0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()
        
        # Original weight (will be set during replacement)
        self.original_weight = None
        self.projection_operator = None  # P_perp for spectral constraint
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters"""
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def compute_projection_operator(self, k_ratio: float = 0.1):
        """
        Compute projection operator P_perp for spectral intrusion measurement
        P_perp = I - U_k @ U_k^T where U_k are top-k singular vectors
        """
        if self.original_weight is None:
            return None
        
        with torch.no_grad():
            W0 = self.original_weight.data
            # Use randomized SVD for efficiency
            k = max(1, int(k_ratio * min(W0.shape)))
            try:
                U, S, V = torch.svd_lowrank(W0, q=k)
                # P_perp = I - U @ U^T
                I = torch.eye(W0.shape[0], device=W0.device, dtype=W0.dtype)
                self.projection_operator = I - U @ U.T
            except:
                logger.warning("SVD failed, using identity as projection")
                self.projection_operator = torch.eye(
                    W0.shape[0], device=W0.device, dtype=W0.dtype
                )
        
        return self.projection_operator
    
    def compute_spectral_intrusion(self) -> torch.Tensor:
        """
        Compute spectral intrusion score: sum_j alpha_j * ||P_perp @ u_j||^2
        where u_j is the j-th column of lora_A
        """
        if self.projection_operator is None:
            return torch.tensor(0.0, device=self.alpha.device)
        
        intrusion = 0.0
        for j in range(self.r_max):
            u_j = self.lora_A[j, :]  # j-th rank-1 component
            # Project to orthogonal space
            projected = self.projection_operator @ u_j
            # Weighted by alpha
            intrusion += self.alpha[j] * (projected.norm() ** 2)
        
        return intrusion
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive rank selection
        x: (batch_size, seq_len, in_features)
        """
        # Compute weighted LoRA update: sum_j alpha_j * (B_j @ A_j)
        # Shape: (r_max,) -> (r_max, 1, 1) for broadcasting
        alpha_weights = self.alpha.view(-1, 1, 1)
        
        # Apply dropout to input
        x_dropout = self.lora_dropout_layer(x)
        
        # Compute: x @ A^T for all ranks -> (batch, seq, r_max)
        lora_out = F.linear(x_dropout, self.lora_A)  # (batch, seq, r_max)
        
        # Apply alpha weights
        lora_out = lora_out * alpha_weights.squeeze(-1)  # (batch, seq, r_max)
        
        # Compute: result @ B^T -> (batch, seq, out_features)
        result = F.linear(lora_out, self.lora_B.T)
        
        return result * self.scaling
    
    def get_effective_rank(self, threshold: float = 0.1) -> int:
        """Return number of active ranks (alpha > threshold)"""
        return (self.alpha > threshold).sum().item()
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'r_max={self.r_max}, effective_rank={self.get_effective_rank()}'


# ============================================================================
# Model Surgery: Replace Linear Layers with SA-LoRA
# ============================================================================

def replace_linear_with_lora(
    model: nn.Module,
    config: SAAutoLoRAConfig,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, SALoRALayer]]:
    """
    Replace target linear layers in the model with SA-LoRA layers
    
    Returns:
        model: Modified model
        lora_layers: Dictionary of added LoRA layers for easy access
    """
    lora_layers = {}
    
    def recursive_replace(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this is a target module
            is_target = any(target in name for target in config.target_modules)
            
            if isinstance(child, nn.Linear) and is_target:
                # Create LoRA layer
                lora_layer = SALoRALayer(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    r_max=config.r_max,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                )
                
                # Store original weight for spectral analysis
                lora_layer.original_weight = child.weight.clone().detach()
                
                # Compute projection operator
                lora_layer.compute_projection_operator(config.k_ratio)
                
                # Create wrapper that combines original + LoRA
                class LoRALinear(nn.Module):
                    def __init__(self, original, lora):
                        super().__init__()
                        self.original = original
                        self.lora = lora
                        # Freeze original weights
                        for param in self.original.parameters():
                            param.requires_grad = False
                    
                    def forward(self, x):
                        return self.original(x) + self.lora(x)
                
                wrapped = LoRALinear(child, lora_layer)
                setattr(module, name, wrapped)
                lora_layers[full_name] = lora_layer
                
                if verbose:
                    logger.info(f"Replaced {full_name}: {child.in_features}x{child.out_features}")
            else:
                # Recurse into child modules
                recursive_replace(child, full_name)
    
    recursive_replace(model)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(p.numel() for layer in lora_layers.values() 
                         for p in layer.parameters())
        logger.info(f"Added {len(lora_layers)} LoRA layers")
        logger.info(f"LoRA parameters: {lora_params:,} / {total_params:,} "
                   f"({100*lora_params/total_params:.2f}%)")
    
    return model, lora_layers


# ============================================================================
# Betty Bilevel Optimization Problems
# ============================================================================

class InnerProblem(ImplicitProblem):
    """Inner loop: optimize LoRA parameters (theta)"""
    
    def __init__(self, name, config, model, lora_layers, train_loader):
        super().__init__(name, config)
        self.model = model
        self.lora_layers = lora_layers
        self.train_loader = train_loader
        self.train_iter = iter(train_loader)
    
    def forward(self):
        """Compute training loss"""
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        
        # Move to device
        input_ids = batch['input_ids'].to(self.model.device)
        labels = batch['labels'].to(self.model.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        return loss
    
    def training_step(self, batch=None):
        """One optimization step"""
        loss = self.forward()
        return loss


class OuterProblem(ImplicitProblem):
    """Outer loop: optimize module selection weights (alpha)"""
    
    def __init__(self, name, config, model, lora_layers, val_loader, sa_config):
        super().__init__(name, config)
        self.model = model
        self.lora_layers = lora_layers
        self.val_loader = val_loader
        self.val_iter = iter(val_loader)
        self.sa_config = sa_config
    
    def forward(self):
        """Compute validation loss + spectral regularization"""
        try:
            batch = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            batch = next(self.val_iter)
        
        # Move to device
        input_ids = batch['input_ids'].to(self.model.device)
        labels = batch['labels'].to(self.model.device)
        
        # Validation loss
        outputs = self.model(input_ids=input_ids, labels=labels)
        val_loss = outputs.loss
        
        # Spectral intrusion penalty
        spectral_penalty = 0.0
        for layer in self.lora_layers.values():
            spectral_penalty += layer.compute_spectral_intrusion()
        
        # L1 sparsity penalty on alpha
        l1_penalty = 0.0
        for layer in self.lora_layers.values():
            l1_penalty += layer.alpha.abs().sum()
        
        # Total loss
        total_loss = (val_loss + 
                     self.sa_config.lambda_spectral * spectral_penalty +
                     self.sa_config.gamma_l1 * l1_penalty)
        
        return total_loss
    
    def training_step(self, batch=None):
        """One optimization step"""
        loss = self.forward()
        return loss


# ============================================================================
# Two-Stage Training (Simplified Alternative)
# ============================================================================

class TwoStageTrainer:
    """
    Two-stage training as a simpler alternative to bilevel optimization
    Stage 1: Train theta with fixed alpha
    Stage 2: Train alpha with fixed theta
    """
    
    def __init__(
        self,
        model: nn.Module,
        lora_layers: Dict[str, SALoRALayer],
        train_loader,
        val_loader,
        config: SAAutoLoRAConfig,
    ):
        self.model = model
        self.lora_layers = lora_layers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = config.device
        self.model.to(self.device)
    
    def stage1_train_theta(self):
        """Stage 1: Train LoRA parameters with fixed alpha=0.5"""
        logger.info("=" * 80)
        logger.info("Stage 1: Training LoRA parameters (theta)")
        logger.info("=" * 80)
        
        # Fix alpha at 0.5 (equal weight for all ranks)
        for layer in self.lora_layers.values():
            layer.alpha.requires_grad = False
            layer.alpha.data.fill_(0.5)
        
        # Collect trainable parameters (only lora_A and lora_B)
        theta_params = []
        for layer in self.lora_layers.values():
            theta_params.extend([layer.lora_A, layer.lora_B])
        
        optimizer = Adam(theta_params, lr=self.config.lr_theta, 
                        weight_decay=self.config.weight_decay)
        
        self.model.train()
        
        for epoch in range(self.config.warmup_epochs):
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Stage1 Epoch {epoch+1}")
            
            for batch in pbar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(theta_params, 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.warmup_epochs}, "
                       f"Avg Loss: {avg_loss:.4f}")
    
    def stage2_train_alpha(self):
        """Stage 2: Train module selection weights with fixed theta"""
        logger.info("=" * 80)
        logger.info("Stage 2: Training module selection weights (alpha)")
        logger.info("=" * 80)
        
        # Freeze theta
        for layer in self.lora_layers.values():
            layer.lora_A.requires_grad = False
            layer.lora_B.requires_grad = False
            layer.alpha.requires_grad = True
        
        # Collect alpha parameters
        alpha_params = [layer.alpha for layer in self.lora_layers.values()]
        
        optimizer = Adam(alpha_params, lr=self.config.lr_alpha)
        
        self.model.eval()  # No dropout during architecture search
        
        for epoch in range(self.config.search_epochs):
            total_loss = 0.0
            total_val_loss = 0.0
            total_spectral = 0.0
            total_l1 = 0.0
            
            pbar = tqdm(self.val_loader, desc=f"Stage2 Epoch {epoch+1}")
            
            for batch in pbar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward
                with torch.set_grad_enabled(True):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    val_loss = outputs.loss
                    
                    # Compute spectral intrusion
                    spectral_penalty = 0.0
                    for layer in self.lora_layers.values():
                        spectral_penalty += layer.compute_spectral_intrusion()
                    
                    # L1 penalty
                    l1_penalty = 0.0
                    for layer in self.lora_layers.values():
                        l1_penalty += layer.alpha.abs().sum()
                    
                    # Total loss
                    loss = (val_loss + 
                           self.config.lambda_spectral * spectral_penalty +
                           self.config.gamma_l1 * l1_penalty)
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(alpha_params, 1.0)
                    optimizer.step()
                    
                    # Clamp alpha to [0, 1]
                    for layer in self.lora_layers.values():
                        layer.alpha.data.clamp_(0, 1)
                    
                    total_loss += loss.item()
                    total_val_loss += val_loss.item()
                    total_spectral += spectral_penalty.item()
                    total_l1 += l1_penalty.item()
                    
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'val': val_loss.item(),
                        'spec': spectral_penalty.item(),
                        'l1': l1_penalty.item()
                    })
            
            # Log statistics
            n_batches = len(self.val_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.search_epochs}")
            logger.info(f"  Val Loss: {total_val_loss/n_batches:.4f}")
            logger.info(f"  Spectral: {total_spectral/n_batches:.4f}")
            logger.info(f"  L1: {total_l1/n_batches:.4f}")
            
            # Print alpha statistics
            self.print_alpha_statistics()
    
    def print_alpha_statistics(self):
        """Print statistics about alpha values"""
        all_alphas = []
        module_stats = {}
        
        for name, layer in self.lora_layers.items():
            alpha_vals = layer.alpha.detach().cpu().numpy()
            all_alphas.extend(alpha_vals)
            
            module_stats[name] = {
                'mean': alpha_vals.mean(),
                'std': alpha_vals.std(),
                'active': (alpha_vals > self.config.alpha_threshold).sum(),
                'total': len(alpha_vals)
            }
        
        all_alphas = np.array(all_alphas)
        logger.info(f"  Alpha Statistics:")
        logger.info(f"    Mean: {all_alphas.mean():.3f}")
        logger.info(f"    Std: {all_alphas.std():.3f}")
        logger.info(f"    Active: {(all_alphas > self.config.alpha_threshold).sum()}"
                   f"/{len(all_alphas)}")
        logger.info(f"    Sparsity: "
                   f"{100*(all_alphas < self.config.alpha_threshold).mean():.1f}%")
    
    def train(self):
        """Run complete two-stage training"""
        self.stage1_train_theta()
        self.stage2_train_alpha()
        
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        
        # Final statistics
        self.print_alpha_statistics()
        
        # Print per-module effective ranks
        logger.info("\nFinal Effective Ranks:")
        for name, layer in self.lora_layers.items():
            eff_rank = layer.get_effective_rank(self.config.alpha_threshold)
            logger.info(f"  {name}: {eff_rank}/{self.config.r_max}")


# ============================================================================
# Betty Bilevel Trainer
# ============================================================================

class BettyBilevelTrainer:
    """
    Bilevel optimization trainer using Betty
    More theoretically sound but computationally expensive
    """
    
    def __init__(
        self,
        model: nn.Module,
        lora_layers: Dict[str, SALoRALayer],
        train_loader,
        val_loader,
        config: SAAutoLoRAConfig,
    ):
        self.model = model
        self.lora_layers = lora_layers
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = config.device
        self.model.to(self.device)
        
        self.setup_betty()
    
    def setup_betty(self):
        """Setup Betty engine and problems"""
        
        # Inner problem config
        inner_config = Config(
            type='darts',
            unroll_steps=1,  # Use 1-step unrolling for efficiency
            retain_graph=True
        )
        
        # Outer problem config  
        outer_config = Config(
            type='darts',
            unroll_steps=1
        )
        
        # Collect parameters
        theta_params = []
        for layer in self.lora_layers.values():
            theta_params.extend([layer.lora_A, layer.lora_B])
        
        alpha_params = [layer.alpha for layer in self.lora_layers.values()]
        
        # Create optimizers
        inner_optimizer = Adam(theta_params, lr=self.config.lr_theta)
        outer_optimizer = Adam(alpha_params, lr=self.config.lr_alpha)
        
        # Create problems
        self.inner = InnerProblem(
            name='inner',
            config=inner_config,
            model=self.model,
            lora_layers=self.lora_layers,
            train_loader=self.train_loader
        )
        
        self.outer = OuterProblem(
            name='outer',
            config=outer_config,
            model=self.model,
            lora_layers=self.lora_layers,
            val_loader=self.val_loader,
            sa_config=self.config
        )
        
        # Setup dependencies
        self.inner.configure(optimizer=inner_optimizer)
        self.outer.configure(optimizer=outer_optimizer)
        
        # Create engine
        problems = [self.inner, self.outer]
        u2l = {self.inner: [self.outer]}  # inner influences outer
        l2u = {}  # no reverse dependency in this case
        dependencies = {'u2l': u2l, 'l2u': l2u}
        
        engine_config = EngineConfig(
            train_iters=len(self.train_loader),
            valid_step=100,
        )
        
        self.engine = Engine(
            config=engine_config,
            problems=problems,
            dependencies=dependencies
        )
    
    def train(self, num_epochs: int = 10):
        """Run bilevel optimization"""
        logger.info("=" * 80)
        logger.info("Starting Betty Bilevel Optimization")
        logger.info("=" * 80)
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Run one epoch
            self.engine.run()
            
            # Print statistics
            if (epoch + 1) % 5 == 0:
                self.print_statistics()
        
        logger.info("=" * 80)
        logger.info("Bilevel Optimization Complete!")
        logger.info("=" * 80)
    
    def print_statistics(self):
        """Print current statistics"""
        all_alphas = []
        for layer in self.lora_layers.values():
            all_alphas.extend(layer.alpha.detach().cpu().numpy())
        
        all_alphas = np.array(all_alphas)
        logger.info(f"Alpha Statistics:")
        logger.info(f"  Mean: {all_alphas.mean():.3f}")
        logger.info(f"  Active: {(all_alphas > self.config.alpha_threshold).sum()}"
                   f"/{len(all_alphas)}")


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Example of how to use SA-AutoLoRA with a pretrained model
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset
    
    # 1. Load pretrained model
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Create SA-AutoLoRA config
    config = SAAutoLoRAConfig(
        r_max=8,
        target_modules=['query', 'key', 'value', 'output', 
                       'intermediate', 'output'],  # BERT module names
        lambda_spectral=1e-4,
        gamma_l1=1e-3,
        warmup_epochs=5,
        search_epochs=10,
    )
    
    # 3. Replace linear layers with SA-LoRA
    model, lora_layers = replace_linear_with_lora(model, config)
    
    # 4. Create dummy data loaders (replace with your actual data)
    # This is just for demonstration
    dummy_input = torch.randint(0, 1000, (100, 32))  # 100 samples, seq_len=32
    dummy_labels = torch.randint(0, 2, (100,))
    
    dataset = TensorDataset(dummy_input, dummy_labels)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {'input_ids': input_ids, 'labels': labels}
    
    train_loader = DataLoader(train_dataset, batch_size=8, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, 
                           collate_fn=collate_fn)
    
    # 5. Choose training strategy
    print("\nChoose training strategy:")
    print("1. Two-Stage Training (recommended for first try)")
    print("2. Betty Bilevel Optimization (more advanced)")
    
    strategy = "two_stage"  # Change to "betty" for bilevel optimization
    
    if strategy == "two_stage":
        trainer = TwoStageTrainer(
            model=model,
            lora_layers=lora_layers,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        trainer.train()
    else:
        trainer = BettyBilevelTrainer(
            model=model,
            lora_layers=lora_layers,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        trainer.train(num_epochs=10)
    
    # 6. Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'lora_layers': {k: v.state_dict() for k, v in lora_layers.items()},
        'config': config
    }, 'sa_autolora_checkpoint.pt')
    
    logger.info("Model saved to sa_autolora_checkpoint.pt")


if __name__ == "__main__":
    example_usage()