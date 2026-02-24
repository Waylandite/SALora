"""
Verification script to validate SALora search results using standard PEFT

This script:
1. Loads the searched LoRA configuration from SALora
2. Trains a standard PEFT model with the discovered ranks
3. Evaluates performance to verify the search results
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

from data_loaders import get_data_loader, DataCollatorForSeq2Seq
from metrics import compute_metrics_by_task


def load_salora_config(config_path):
    """Load SALora search results"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_peft_config_from_salora(salora_config, use_rank_pattern=True):
    """
    Create PEFT LoraConfig from SALora search results

    Args:
        salora_config: Loaded SALora configuration
        use_rank_pattern: If True and available, use full rank_pattern (PEFT 0.7.0+)
                         If False, use fallback global rank

    Returns:
        LoraConfig instance, rank_used (rank_pattern or single value)
    """
    import numpy as np

    # Check format version
    version = salora_config.get('version', '1.0')

    if version >= '2.0' and use_rank_pattern and 'rank_pattern' in salora_config:
        # New format with rank_pattern support (PEFT 0.7.0+)
        print(f"\n{'='*70}")
        print(f"✅ Using rank_pattern (PEFT 0.7.0+ feature)")
        print(f"{'='*70}")

        rank_pattern = salora_config['rank_pattern']
        peft_config_dict = salora_config['peft_config']

        print(f"\nrank_pattern loaded: {len(rank_pattern)} layer-module combinations")
        print(f"Preview (first 5):")
        for i, (key, rank) in enumerate(list(rank_pattern.items())[:5]):
            print(f"  {key}: {rank}")
        print(f"  ...")

        # Statistics
        ranks = list(rank_pattern.values())
        print(f"\nRank distribution:")
        print(f"  Min: {min(ranks)}, Max: {max(ranks)}")
        print(f"  Mean: {np.mean(ranks):.1f}, Median: {int(np.median(ranks))}")

        # Create LoraConfig with rank_pattern
        lora_config = LoraConfig(
            rank_pattern=rank_pattern,
            lora_alpha=peft_config_dict['lora_alpha'],
            lora_dropout=peft_config_dict['lora_dropout'],
            bias=peft_config_dict['bias'],
            task_type=TaskType.CAUSAL_LM,
            target_modules=peft_config_dict['target_modules'],
        )

        return lora_config, rank_pattern

    elif 'peft_config_fallback' in salora_config:
        # Use fallback config (single global rank)
        print(f"\n{'='*70}")
        print(f"⚠️  Using fallback: single global rank")
        print(f"   (rank_pattern not available or disabled)")
        print(f"{'='*70}")

        peft_config_dict = salora_config['peft_config_fallback']
        rank = peft_config_dict['r']

        print(f"\nUsing global rank: {rank}")

        if 'compression_strategies' in salora_config:
            strategies = salora_config['compression_strategies']
            print(f"\nAvailable strategies from search:")
            print(f"  global_median: {strategies['global_median']}")
            print(f"  global_mean: {strategies['global_mean']}")

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=peft_config_dict['lora_alpha'],
            lora_dropout=peft_config_dict['lora_dropout'],
            bias=peft_config_dict['bias'],
            task_type=TaskType.CAUSAL_LM,
            target_modules=peft_config_dict['target_modules'],
        )

        return lora_config, rank

    else:
        # Old format - backward compatibility
        print(f"\n⚠️  Old format detected, using compression strategy")

        if 'compression_strategies' in salora_config:
            rank = salora_config['compression_strategies']['global_median']
        elif 'peft_config' in salora_config:
            rank = salora_config['peft_config']['r']
        else:
            raise ValueError("Cannot determine rank from config")

        print(f"Using rank: {rank}")

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=salora_config['peft_config']['lora_alpha'],
            lora_dropout=salora_config['peft_config']['lora_dropout'],
            bias=salora_config['peft_config']['bias'],
            task_type=TaskType.CAUSAL_LM,
            target_modules=salora_config['peft_config']['target_modules'],
        )

        return lora_config, rank


def evaluate_model(model, dataloader, tokenizer, device, task_type, max_length=128, num_beams=5):
    """Evaluate generation model"""
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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

            # Decode
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)

    # Compute metrics
    metrics = compute_metrics_by_task(task_type, all_predictions, all_references)

    return metrics, all_predictions, all_references


def main():
    parser = argparse.ArgumentParser(description="Verify SALora results with standard PEFT")
    parser.add_argument("--salora_config", type=str, required=True,
                       help="Path to SALora peft_config.json")
    parser.add_argument("--task", type=str, default="code2nl",
                       choices=["code2nl", "code2code", "nl2code"])
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--output_dir", type=str, default="./output_peft_verify")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--use_rank_pattern", action="store_true", default=True,
                       help="Use full rank_pattern if available (PEFT 0.7.0+)")
    parser.add_argument("--no_rank_pattern", action="store_true",
                       help="Force use of fallback global rank instead of rank_pattern")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load SALora config
    print("Loading SALora configuration...")
    salora_config = load_salora_config(args.salora_config)

    # Print configuration info
    if 'salora_full_config' in salora_config:
        full_config = salora_config['salora_full_config']
        print("\nSALora Search Results:")
        print(f"  Layers: {full_config['n_layers']}")
        print(f"  Total layer-module combinations: {len(full_config['layer_module_ranks'])}")

    # Check rank_pattern availability
    use_rank_pattern = args.use_rank_pattern and not args.no_rank_pattern
    if 'rank_pattern' not in salora_config:
        print("\n⚠️  rank_pattern not found in config, will use fallback")
        use_rank_pattern = False

    # Create PEFT config
    lora_config, rank_used = create_peft_config_from_salora(salora_config, use_rank_pattern)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    if args.task == "code2nl":
        max_source_length = 512
        max_target_length = 128
    elif args.task == "code2code":
        max_source_length = 512
        max_target_length = 128
    elif args.task == "nl2code":
        max_source_length = 128
        max_target_length = 512
    else:
        raise ValueError(f"Unknown task: {args.task}")

    data_loader = get_data_loader(
        task_type=args.task,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    tokenized_dataset = data_loader.load_dataset()

    # Create dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    val_dataloader = DataLoader(
        tokenized_dataset["validation"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    test_dataloader = DataLoader(
        tokenized_dataset["test"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Apply PEFT
    print("Applying PEFT with SALora-discovered configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Create scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.06 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Training loop
    print("\nStarting training with PEFT...")
    best_val_bleu = 0.0
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} - Average training loss: {avg_loss:.4f}")

        # Evaluate
        print("Evaluating...")
        val_metrics, _, _ = evaluate_model(
            model, val_dataloader, tokenizer, device,
            args.task, max_target_length, num_beams=5
        )

        print("Validation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save best model
        if 'bleu' in val_metrics and val_metrics['bleu'] > best_val_bleu:
            best_val_bleu = val_metrics['bleu']
            print(f"New best BLEU: {best_val_bleu:.4f}, saving model...")
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))

    # Final evaluation on test set
    print("\n" + "="*80)
    print("Final evaluation on test set...")
    print("="*80)

    # Load best model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = get_peft_model(model, lora_config)
    model.load_adapter(os.path.join(args.output_dir, "best_model"))
    model = model.to(device)

    test_metrics, test_predictions, test_references = evaluate_model(
        model, test_dataloader, tokenizer, device,
        args.task, max_target_length, num_beams=5
    )

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    results = {
        "task": args.task,
        "salora_config_used": args.salora_config,
        "used_rank_pattern": isinstance(rank_used, dict),
        "rank_info": {
            "type": "rank_pattern" if isinstance(rank_used, dict) else "single_rank",
            "value": rank_used if not isinstance(rank_used, dict) else f"{len(rank_used)} combinations"
        },
        "lora_alpha": lora_config.lora_alpha,
        "test_metrics": test_metrics,
    }

    # Include full SALora results if available
    if 'salora_full_config' in salora_config:
        results["salora_rank_summary"] = salora_config['salora_full_config']['rank_summary_by_module_type']
    if 'compression_strategies' in salora_config:
        results["fallback_strategies"] = salora_config['compression_strategies']

    with open(os.path.join(args.output_dir, "verification_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save sample predictions
    sample_results = []
    for i in range(min(20, len(test_predictions))):
        sample_results.append({
            "prediction": test_predictions[i],
            "reference": test_references[i],
        })

    with open(os.path.join(args.output_dir, "sample_predictions.json"), 'w') as f:
        json.dump(sample_results, f, indent=2, ensure_ascii=False)

    print(f"\nVerification complete! Results saved to {args.output_dir}")
    print("\nSummary:")
    if isinstance(rank_used, dict):
        print(f"  ✅ Used full rank_pattern: {len(rank_used)} layer-module combinations")
        ranks = list(rank_used.values())
        print(f"  Rank range: [{min(ranks)}, {max(ranks)}]")
    else:
        print(f"  Used single global rank: {rank_used}")
    print(f"  Test BLEU: {test_metrics.get('bleu', 0.0):.4f}")


if __name__ == "__main__":
    main()
