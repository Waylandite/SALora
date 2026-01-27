"""
Complete Experiment Script for SA-AutoLoRA
Includes: data loading, training, evaluation, and analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    default_data_collator
)
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import os
from tqdm import tqdm
import argparse
from datetime import datetime

# Import from the main implementation file
# Assuming the previous code is saved as sa_autolora.py
from sa_autolora import (
    SAAutoLoRAConfig,
    replace_linear_with_lora,
    TwoStageTrainer,
    BettyBilevelTrainer,
)


# ============================================================================
# Data Loading
# ============================================================================

class GLUEDataLoader:
    """Load and prepare GLUE benchmark tasks"""
    
    TASK_CONFIG = {
        'sst2': {
            'dataset': 'glue',
            'subset': 'sst2',
            'num_labels': 2,
            'text_fields': ['sentence'],
            'metric': 'accuracy'
        },
        'mrpc': {
            'dataset': 'glue',
            'subset': 'mrpc',
            'num_labels': 2,
            'text_fields': ['sentence1', 'sentence2'],
            'metric': 'accuracy'
        },
        'qnli': {
            'dataset': 'glue',
            'subset': 'qnli',
            'num_labels': 2,
            'text_fields': ['question', 'sentence'],
            'metric': 'accuracy'
        },
        'qqp': {
            'dataset': 'glue',
            'subset': 'qqp',
            'num_labels': 2,
            'text_fields': ['question1', 'question2'],
            'metric': 'accuracy'
        },
    }
    
    def __init__(
        self,
        task_name: str,
        model_name: str,
        max_length: int = 128,
        train_size: int = None,
        val_size: int = None,
    ):
        self.task_name = task_name.lower()
        self.model_name = model_name
        self.max_length = max_length
        self.train_size = train_size
        self.val_size = val_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load dataset
        task_info = self.TASK_CONFIG[self.task_name]
        self.dataset = load_dataset(task_info['dataset'], task_info['subset'])
        self.num_labels = task_info['num_labels']
        self.text_fields = task_info['text_fields']
        
        # Preprocess
        self.train_dataset = self._preprocess(self.dataset['train'])
        self.val_dataset = self._preprocess(self.dataset['validation'])
        
        # Subsample if specified
        if train_size:
            self.train_dataset = self.train_dataset.select(range(min(train_size, len(self.train_dataset))))
        if val_size:
            self.val_dataset = self.val_dataset.select(range(min(val_size, len(self.val_dataset))))
    
    def _preprocess(self, dataset):
        """Tokenize dataset"""
        def tokenize_function(examples):
            if len(self.text_fields) == 1:
                return self.tokenizer(
                    examples[self.text_fields[0]],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length
                )
            else:
                return self.tokenizer(
                    examples[self.text_fields[0]],
                    examples[self.text_fields[1]],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length
                )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized
    
    def get_dataloaders(self, batch_size: int = 16):
        """Create train and validation dataloaders"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator
        )
        
        return train_loader, val_loader


# ============================================================================
# Evaluation Metrics
# ============================================================================

class MetricsTracker:
    """Track and compute evaluation metrics"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'spectral_intrusion': [],
            'l1_sparsity': [],
            'active_ranks': [],
            'alpha_mean': [],
            'alpha_std': [],
        }
        
        self.per_module_metrics = {}
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def compute_accuracy(self, model, dataloader, device):
        """Compute accuracy on a dataset"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids)
                predictions = outputs.logits.argmax(dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def compute_perplexity(self, model, tokenizer, text_samples, device):
        """
        Compute perplexity on text samples to measure knowledge retention
        Lower perplexity = better knowledge retention
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in text_samples:
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                ).to(device)
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def analyze_alpha_distribution(self, lora_layers, threshold=0.1):
        """Analyze alpha distribution across modules"""
        all_alphas = []
        module_stats = {}
        
        for name, layer in lora_layers.items():
            alpha_vals = layer.alpha.detach().cpu().numpy()
            all_alphas.extend(alpha_vals)
            
            module_type = 'attention' if any(x in name for x in ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']) else 'ffn'
            
            module_stats[name] = {
                'type': module_type,
                'mean': float(alpha_vals.mean()),
                'std': float(alpha_vals.std()),
                'min': float(alpha_vals.min()),
                'max': float(alpha_vals.max()),
                'active': int((alpha_vals > threshold).sum()),
                'total': int(len(alpha_vals)),
                'effective_rank': int(layer.get_effective_rank(threshold))
            }
        
        all_alphas = np.array(all_alphas)
        
        return {
            'overall': {
                'mean': float(all_alphas.mean()),
                'std': float(all_alphas.std()),
                'active_ratio': float((all_alphas > threshold).mean()),
                'sparsity': float((all_alphas < threshold).mean())
            },
            'per_module': module_stats
        }
    
    def save_results(self, output_dir: str, filename: str = 'metrics.json'):
        """Save metrics to file"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def plot_training_curves(self, output_dir: str):
        """Plot training curves"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        if self.metrics['train_loss']:
            axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        if self.metrics['val_loss']:
            axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        if self.metrics['val_accuracy']:
            axes[0, 1].plot(self.metrics['val_accuracy'])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].grid(True)
        
        # Spectral intrusion
        if self.metrics['spectral_intrusion']:
            axes[1, 0].plot(self.metrics['spectral_intrusion'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Spectral Intrusion')
            axes[1, 0].set_title('Spectral Intrusion Score')
            axes[1, 0].grid(True)
        
        # Active ranks
        if self.metrics['active_ranks']:
            axes[1, 1].plot(self.metrics['active_ranks'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Number of Active Ranks')
            axes[1, 1].set_title('Active Ranks Over Time')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
        print(f"Training curves saved to {output_dir}/training_curves.png")
        plt.close()
    
    def plot_alpha_heatmap(self, lora_layers, output_dir: str):
        """Plot heatmap of alpha values across modules"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect alpha values
        module_names = []
        alpha_matrix = []
        
        for name, layer in lora_layers.items():
            # Shorten name for visualization
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
            module_names.append(short_name)
            alpha_vals = layer.alpha.detach().cpu().numpy()
            alpha_matrix.append(alpha_vals)
        
        alpha_matrix = np.array(alpha_matrix)
        
        # Create heatmap
        plt.figure(figsize=(12, max(6, len(module_names) * 0.3)))
        sns.heatmap(
            alpha_matrix,
            yticklabels=module_names,
            xticklabels=[f'r{i+1}' for i in range(alpha_matrix.shape[1])],
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Alpha Value'},
            annot=True,
            fmt='.2f'
        )
        plt.title('Alpha Values Across Modules and Ranks')
        plt.xlabel('Rank')
        plt.ylabel('Module')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'alpha_heatmap.png'), dpi=300)
        print(f"Alpha heatmap saved to {output_dir}/alpha_heatmap.png")
        plt.close()


# ============================================================================
# Baseline Comparisons
# ============================================================================

def train_baseline_lora(
    model,
    train_loader,
    val_loader,
    config,
    num_epochs=10,
    device='cuda'
):
    """Train standard LoRA baseline for comparison"""
    from peft import LoraConfig, get_peft_model
    
    # Standard LoRA config
    lora_config = LoraConfig(
        r=config.r_max,
        lora_alpha=config.lora_alpha,
        target_modules=['query', 'key', 'value'],  # Only q, v like standard LoRA
        lora_dropout=config.lora_dropout,
        bias='none',
        task_type='SEQ_CLS'
    )
    
    model = get_peft_model(model, lora_config)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_theta)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids)
                predictions = outputs.logits.argmax(dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.4f}")
        
        best_acc = max(best_acc, accuracy)
    
    return model, best_acc


# ============================================================================
# Main Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Orchestrate complete experiments"""
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            args.output_dir,
            f"{args.task}_{args.model_name.split('/')[-1]}_{self.timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    def run_sa_autolora_experiment(self):
        """Run SA-AutoLoRA experiment"""
        print("=" * 80)
        print("Running SA-AutoLoRA Experiment")
        print("=" * 80)
        
        # Load data
        print("\n1. Loading data...")
        data_loader = GLUEDataLoader(
            task_name=self.args.task,
            model_name=self.args.model_name,
            max_length=self.args.max_length,
            train_size=self.args.train_size,
            val_size=self.args.val_size,
        )
        train_loader, val_loader = data_loader.get_dataloaders(
            batch_size=self.args.batch_size
        )
        
        # Load model
        print("\n2. Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name,
            num_labels=data_loader.num_labels
        )
        
        # Create SA-AutoLoRA config
        print("\n3. Setting up SA-AutoLoRA...")
        config = SAAutoLoRAConfig(
            r_max=self.args.r_max,
            target_modules=self.args.target_modules,
            lambda_spectral=self.args.lambda_spectral,
            gamma_l1=self.args.gamma_l1,
            lr_theta=self.args.lr_theta,
            lr_alpha=self.args.lr_alpha,
            warmup_epochs=self.args.warmup_epochs,
            search_epochs=self.args.search_epochs,
            device=self.args.device,
        )
        
        # Replace layers
        model, lora_layers = replace_linear_with_lora(model, config)
        
        # Initialize metrics tracker
        metrics = MetricsTracker()
        
        # Train
        print("\n4. Training...")
        if self.args.training_mode == 'two_stage':
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
            trainer.train(num_epochs=self.args.warmup_epochs + self.args.search_epochs)
        
        # Evaluation
        print("\n5. Evaluating...")
        final_accuracy = metrics.compute_accuracy(
            model, val_loader, self.args.device
        )
        print(f"Final Validation Accuracy: {final_accuracy:.4f}")
        
        # Analyze results
        print("\n6. Analyzing results...")
        alpha_analysis = metrics.analyze_alpha_distribution(lora_layers)
        
        print("\nAlpha Distribution:")
        print(f"  Mean: {alpha_analysis['overall']['mean']:.3f}")
        print(f"  Std: {alpha_analysis['overall']['std']:.3f}")
        print(f"  Sparsity: {alpha_analysis['overall']['sparsity']*100:.1f}%")
        
        print("\nPer-Module Effective Ranks:")
        for name, stats in alpha_analysis['per_module'].items():
            print(f"  {name}: {stats['effective_rank']}/{config.r_max}")
        
        # Save results
        print("\n7. Saving results...")
        results = {
            'final_accuracy': final_accuracy,
            'alpha_analysis': alpha_analysis,
            'config': vars(self.args),
        }
        
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'lora_layers': {k: v.state_dict() for k, v in lora_layers.items()},
            'config': config,
        }, os.path.join(self.output_dir, 'model.pt'))
        
        # Plot visualizations
        metrics.plot_alpha_heatmap(lora_layers, self.output_dir)
        
        print(f"\nResults saved to {self.output_dir}")
        
        return final_accuracy, alpha_analysis
    
    def run_comparison_experiments(self):
        """Run SA-AutoLoRA vs baselines"""
        results = {}
        
        # 1. SA-AutoLoRA
        print("\n" + "=" * 80)
        print("Experiment 1: SA-AutoLoRA (Full)")
        print("=" * 80)
        acc, analysis = self.run_sa_autolora_experiment()
        results['sa_autolora'] = {
            'accuracy': acc,
            'params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # 2. Baseline LoRA (optional, if peft is installed)
        # Uncomment if you want to run baseline comparison
        # print("\n" + "=" * 80)
        # print("Experiment 2: Baseline LoRA")
        # print("=" * 80)
        # ...
        
        return results


# ============================================================================
# CLI Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='SA-AutoLoRA Experiments')
    
    # Task and model
    parser.add_argument('--task', type=str, default='sst2',
                       choices=['sst2', 'mrpc', 'qnli', 'qqp'],
                       help='GLUE task name')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Pretrained model name')
    
    # Data
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--train_size', type=int, default=None,
                       help='Number of training samples (None = use all)')
    parser.add_argument('--val_size', type=int, default=None,
                       help='Number of validation samples')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    
    # SA-AutoLoRA config
    parser.add_argument('--r_max', type=int, default=8,
                       help='Maximum rank')
    parser.add_argument('--target_modules', type=str, nargs='+',
                       default=['query', 'key', 'value', 'output', 'intermediate'],
                       help='Target modules for LoRA')
    parser.add_argument('--lambda_spectral', type=float, default=1e-4,
                       help='Spectral penalty coefficient')
    parser.add_argument('--gamma_l1', type=float, default=1e-3,
                       help='L1 sparsity penalty')
    
    # Optimization
    parser.add_argument('--lr_theta', type=float, default=1e-4,
                       help='Learning rate for theta')
    parser.add_argument('--lr_alpha', type=float, default=1e-2,
                       help='Learning rate for alpha')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Warmup epochs for stage 1')
    parser.add_argument('--search_epochs', type=int, default=10,
                       help='Search epochs for stage 2')
    
    # Training mode
    parser.add_argument('--training_mode', type=str, default='two_stage',
                       choices=['two_stage', 'betty'],
                       help='Training strategy')
    
    # Other
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run experiment
    runner = ExperimentRunner(args)
    results = runner.run_sa_autolora_experiment()
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()