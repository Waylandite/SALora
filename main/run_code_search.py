"""
Main training script for SALora on code-related tasks with Qwen2.5-Coder

This script implements SALora training for:
- code2nl: Code summarization (JCSD)
- code2code: Assertion generation (ATLAS)
- nl2code: Code generation (conCode)

Uses bilevel optimization to search for optimal rank configuration.
"""

import os
import sys
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import argparse
from tqdm import tqdm
import json

# Import Betty framework
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AutoLoRA'))
from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

# Import SALora modules
from config import SALoraConfig, Code2NLConfig, Code2CodeConfig, NL2CodeConfig
from models import create_salora_model, save_lora_config_for_peft
from search import (
    SALoraArchitecture,
    ArchitectureSearchEngine,
    SpectralIntrusionMetric,
    compute_l1_regularization,
)
from loralib import mark_only_lora_as_trainable
from data_loaders import get_data_loader, DataCollatorForSeq2Seq
from metrics import compute_metrics_by_task


class ArchitectureProblem(ImplicitProblem):
    """Outer loop: Update architecture parameters (alphas)"""

    def __init__(self, *args, spectral_metric=None, arch_engine=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectral_metric = spectral_metric
        self.arch_engine = arch_engine

    def training_step(self, batch):
        """
        Compute validation loss + spectral intrusion + L1 regularization
        """
        # Get architecture parameters
        alphas = self.module()
        alpha_dict = self._get_alpha_dict(alphas)

        # Forward pass on validation set
        model = self.model_problem.module
        outputs = model(**batch)
        loss = outputs.loss

        # Compute spectral intrusion score
        if self.spectral_metric is not None:
            lora_modules = self.model_problem.module_dict
            spectral_score = self.spectral_metric.compute_intrusion_score_simplified(
                lora_modules, alpha_dict
            )
        else:
            spectral_score = torch.tensor(0.0, device=loss.device)

        # Compute L1 regularization
        l1_score = compute_l1_regularization(alpha_dict)

        # Total loss
        if self.arch_engine is not None:
            reg_loss = self.arch_engine.compute_regularization(spectral_score, l1_score)
            total_loss = loss + reg_loss
        else:
            total_loss = loss

        return total_loss

    def _get_alpha_dict(self, alphas):
        """Convert alpha tensor to dictionary"""
        n_layers, n_modules, r_max = alphas.shape
        module_names = ['query', 'key', 'value', 'output', 'gate', 'up', 'down']

        alpha_dict = {}
        for layer_idx in range(n_layers):
            for module_idx, module_name in enumerate(module_names):
                key = f"layer.{layer_idx}.{module_name}"
                alpha_dict[key] = alphas[layer_idx, module_idx, :]

        return alpha_dict


class ModelProblem(ImplicitProblem):
    """Inner loop: Update LoRA parameters"""

    def __init__(self, *args, module_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_dict = module_dict

    def training_step(self, batch):
        """Compute training loss"""
        # Get architecture parameters from outer loop
        alphas = self.arch_problem()

        # Forward pass
        outputs = self.module(**batch)
        loss = outputs.loss

        return loss


class SALoraEngine(Engine):
    """Custom engine for SALora with validation metrics"""

    @torch.no_grad()
    def validation(self):
        """Compute validation metrics and effective ranks"""
        arch_module = self.arch_problem.module
        effective_ranks = arch_module.get_effective_ranks()
        rank_summary = arch_module.get_rank_summary()

        return {
            "rank_summary": rank_summary,
            "effective_ranks": effective_ranks,
        }


def evaluate_generation(
    model,
    dataloader,
    tokenizer,
    device,
    task_type,
    max_length=128,
    num_beams=5,
):
    """
    Evaluate generation model on dataset

    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: List of generated texts
        references: List of reference texts
    """
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Generate
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

            # Decode predictions
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Decode labels
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)

    # Compute metrics
    metrics = compute_metrics_by_task(task_type, all_predictions, all_references)

    return metrics, all_predictions, all_references


def main():
    parser = argparse.ArgumentParser(description="Train SALora on code tasks")
    parser.add_argument("--task", type=str, default="code2nl",
                       choices=["code2nl", "code2code", "nl2code"])
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Load config
    if args.task == "code2nl":
        config = Code2NLConfig(
            data_dir=args.data_dir,
            model_name=args.model_name,
            output_dir=args.output_dir,
            seed=args.seed,
            lora_r=args.lora_r,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )
        task_type = 'code2nl'
    elif args.task == "code2code":
        config = Code2CodeConfig(
            data_dir=args.data_dir,
            model_name=args.model_name,
            output_dir=args.output_dir,
            seed=args.seed,
            lora_r=args.lora_r,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )
        task_type = 'code2code'
    elif args.task == "nl2code":
        config = NL2CodeConfig(
            data_dir=args.data_dir,
            model_name=args.model_name,
            output_dir=args.output_dir,
            seed=args.seed,
            lora_r=args.lora_r,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )
        task_type = 'nl2code'
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    data_loader = get_data_loader(
        task_type=task_type,
        data_dir=config.data_dir,
        tokenizer=tokenizer,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )
    tokenized_dataset = data_loader.load_dataset()

    # Split train into train and search
    train_dataset = tokenized_dataset["train"]
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(config.train_split_ratio * num_train)

    train_indices = indices[:split]
    search_indices = indices[split:]

    train_subset = Subset(train_dataset, train_indices)
    search_subset = Subset(train_dataset, search_indices)

    # Create dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    train_dataloader = DataLoader(
        train_subset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.batch_size,
        drop_last=True,
    )
    search_dataloader = DataLoader(
        search_subset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.batch_size,
        drop_last=True,
    )

    # Create validation dataloader
    val_dataset = tokenized_dataset["validation"]
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=config.eval_batch_size,
    )

    # Create model with SALora
    print("Creating model...")
    model, lora_module_names, n_layers = create_salora_model(
        model_name=config.model_name,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=config.task_type,
    )
    model = model.to(config.device)

    # Create architecture
    architecture = SALoraArchitecture(
        n_layers=n_layers,
        r_max=config.lora_r,
        init_strategy=config.arch_init_strategy,
    )
    architecture = architecture.to(config.device)

    # Create spectral metric
    spectral_metric = SpectralIntrusionMetric(
        keep_ratio=config.spectral_keep_ratio,
        device=config.device,
    )

    # Register pretrained weights
    print("Registering pretrained weights for spectral metric...")
    spectral_metric.register_pretrained_weights(model, list(lora_module_names.keys()))

    # Create architecture search engine
    arch_engine = ArchitectureSearchEngine(
        architecture=architecture,
        lambda_spectral=config.lambda_spectral,
        lambda_l1=config.lambda_l1,
    )

    # Create optimizers
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = AdamW(lora_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    arch_optimizer = AdamW(
        architecture.parameters(),
        lr=config.arch_learning_rate,
        weight_decay=config.arch_weight_decay,
    )

    # Create scheduler
    num_training_steps = len(train_dataloader) * config.num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.06 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Setup Betty problems
    inner_config = Config(type="darts", unroll_steps=config.unroll_steps)
    outer_config = Config(retain_graph=True)

    model_problem = ModelProblem(
        name="model",
        module=model,
        optimizer=optimizer,
        train_data_loader=train_dataloader,
        config=inner_config,
        module_dict=lora_module_names,
    )

    arch_problem = ArchitectureProblem(
        name="arch",
        module=architecture,
        optimizer=arch_optimizer,
        train_data_loader=search_dataloader,
        config=outer_config,
        spectral_metric=spectral_metric,
        arch_engine=arch_engine,
    )

    # Link problems
    model_problem.arch_problem = arch_problem
    arch_problem.model_problem = model_problem

    problems = [arch_problem, model_problem]
    dependencies = {
        "l2u": {model_problem: [arch_problem]},
        "u2l": {arch_problem: [model_problem]},
    }

    # Calculate training iterations
    train_iters = int(
        config.num_epochs
        * (len(train_indices) // config.batch_size + 1)
        * config.unroll_steps
    )

    # Setup engine config
    engine_config = EngineConfig(
        valid_step=config.valid_step * config.unroll_steps,
        train_iters=train_iters,
        roll_back=config.roll_back,
    )

    # Create and run engine
    print("Starting SALora architecture search...")
    engine = SALoraEngine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies,
    )

    engine.run()

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics, val_predictions, val_references = evaluate_generation(
        model, val_dataloader, tokenizer, config.device,
        task_type, config.max_target_length, config.num_beams
    )

    print("\nValidation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save final results
    print("\nSaving final results...")
    final_ranks = architecture.get_rank_summary()
    print("\nFinal rank allocation:")
    for module_name, ranks in final_ranks.items():
        print(f"{module_name}: {ranks}")

    # Save model and architecture
    torch.save(model.state_dict(), os.path.join(config.output_dir, "model.pt"))
    torch.save(architecture.state_dict(), os.path.join(config.output_dir, "architecture.pt"))

    # Save LoRA config for PEFT
    peft_config_path = os.path.join(config.output_dir, "peft_config.json")

    # Detect model type for correct naming convention
    model_type = model.config.model_type if hasattr(model, 'config') else 'qwen2'
    # Map model types to our naming convention
    if 'qwen' in model_type.lower():
        model_type = 'qwen2'
    elif 'llama' in model_type.lower():
        model_type = 'llama'
    elif 'roberta' in model_type.lower():
        model_type = 'roberta'

    peft_config, rank_pattern = save_lora_config_for_peft(
        architecture,
        lora_module_names,
        peft_config_path,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.modules_to_replace,
        model_type=model_type,
    )

    # Save validation results
    results = {
        "task": task_type,
        "metrics": val_metrics,
        "rank_summary": final_ranks,
        "rank_pattern": rank_pattern,  # Full per-layer per-module ranks
        "peft_config": peft_config,
    }

    with open(os.path.join(config.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save some predictions for inspection
    sample_results = []
    for i in range(min(10, len(val_predictions))):
        sample_results.append({
            "prediction": val_predictions[i],
            "reference": val_references[i],
        })

    with open(os.path.join(config.output_dir, "sample_predictions.json"), 'w') as f:
        json.dump(sample_results, f, indent=2, ensure_ascii=False)

    print(f"\nTraining complete! Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
