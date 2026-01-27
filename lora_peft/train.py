import math
import os
import json
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.optim.lr_scheduler as lr_scheduler
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# -------------------------------------------------------------------------
# Compatibility helpers for older PyTorch (<2.1) used on CUDA 11.7 systems.
# Transformer >=4.44 expects torch.optim.lr_scheduler.LRScheduler which
# only exists in PyTorch 2.1+. On older versions we alias it to _LRScheduler.
# -------------------------------------------------------------------------
if not hasattr(lr_scheduler, "LRScheduler") and hasattr(lr_scheduler, "_LRScheduler"):

    class LRScheduler(lr_scheduler._LRScheduler):
        """Backport of torch.optim.lr_scheduler.LRScheduler for torch < 2.1."""

        def __init__(self, optimizer, last_epoch: int = -1, verbose: bool = False):
            super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    lr_scheduler.LRScheduler = LRScheduler  # type: ignore[attr-defined]

# Import transformers module first so we can patch availability flags before
# pulling specific classes.
import transformers
import transformers.utils.import_utils as import_utils

import numpy as np

# Optional metric libraries; fall back to simple implementations if unavailable
try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
except Exception:
    nltk = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None


def _force_enable_torch_for_transformers() -> None:
    """Ensure transformers treats the current torch install as available."""

    if not import_utils.is_torch_available():
        import_utils._torch_available = True  # type: ignore[attr-defined]

        def _patched_is_torch_available() -> bool:
            return True

        import_utils.is_torch_available = _patched_is_torch_available  # type: ignore[assignment]

    # Some utilities rely on _torch_version for logging/comparisons.
    if hasattr(import_utils, "_torch_version"):
        import_utils._torch_version = torch.__version__  # type: ignore[attr-defined]


_force_enable_torch_for_transformers()

from transformers import Trainer, TrainingArguments, set_seed

from model_wrapper import create_model_and_tokenizer
from data_loader import InstructionDataset, DataCollatorForCausalLMWithPadding


@dataclass
class Args:
    # Data
    data_path: Optional[str] = None
    train_path: Optional[str] = None
    valid_path: Optional[str] = None
    test_path: Optional[str] = None
    data_format: str = "jsonl"  # jsonl or tsv
    user_field: str = "instruction"
    assistant_field: str = "output"
    system_prompt: Optional[str] = None
    max_length: int = 2048
    eval_ratio: float = 0.05

    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-3B"
    attn_impl: Optional[str] = None  # e.g., "flash_attention_2"
    bf16: bool = True
    fp16: bool = False
    use_8bit_adam: bool = False

    # LoRA (edit these to run experiments)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # if None, uses defaults in wrapper
    lora_bias: str = "none"

    # Training
    output_dir: str = "./outputs/qwen2p5_coder_lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1  # Increased from 0.03 to prevent premature LR decay
    max_grad_norm: float = 1.0  # Gradient clipping to prevent spikes
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    eval_accumulation_steps: int = 1
    # Early stopping and best model saving
    early_stopping_patience: int = 5  # Stop if eval_loss doesn't improve for N evaluations
    load_best_model_at_end: bool = True  # Load best checkpoint at end
    metric_for_best_model: str = "eval_loss"  # Monitor eval_loss
    greater_is_better: bool = False  # Lower eval_loss is better


def parse_args() -> Args:
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA on Qwen2.5-Coder-3B")

    # Data
    parser.add_argument("--data_path", type=str, default=None, help="Single dataset path (will be split)")
    parser.add_argument("--train_path", type=str, default=None, help="Path to train.jsonl")
    parser.add_argument("--valid_path", type=str, default=None, help="Path to valid.jsonl")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test.jsonl")
    parser.add_argument("--data_format", type=str, default="jsonl", choices=["jsonl", "tsv"])
    parser.add_argument("--user_field", type=str, default="instruction")
    parser.add_argument("--assistant_field", type=str, default="output")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_ratio", type=float, default=0.05)

    # Model
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--attn_impl", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use bitsandbytes paged_adamw_8bit if available")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated module names (e.g. q_proj,k_proj,v_proj,o_proj)",
    )
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])

    # Training
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen2p5_coder_lora")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (increased default to prevent premature LR decay)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm to prevent gradient spikes")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1, help="Accumulate eval tensors on CPU to reduce GPU memory")
    # Early stopping and best model
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience (number of evaluations without improvement)")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True, help="Load best checkpoint at end based on eval_loss")
    parser.add_argument("--no_load_best_model_at_end", dest="load_best_model_at_end", action="store_false", help="Disable loading best model at end")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Metric to use for best model selection")
    parser.add_argument("--greater_is_better", action="store_true", help="Whether greater metric value is better (default: False for loss)")

    ns = parser.parse_args()
    args = Args(
        data_path=ns.data_path,
        train_path=ns.train_path,
        valid_path=ns.valid_path,
        test_path=ns.test_path,
        data_format=ns.data_format,
        user_field=ns.user_field,
        assistant_field=ns.assistant_field,
        system_prompt=ns.system_prompt,
        max_length=ns.max_length,
        eval_ratio=ns.eval_ratio,
        base_model=ns.base_model,
        attn_impl=ns.attn_impl,
        bf16=ns.bf16,
        lora_r=ns.lora_r,
        lora_alpha=ns.lora_alpha,
        lora_dropout=ns.lora_dropout,
        lora_target_modules=(
            [s.strip() for s in ns.lora_target_modules.split(",") if s.strip()]
            if ns.lora_target_modules
            else None
        ),
        lora_bias=ns.lora_bias,
        output_dir=ns.output_dir,
        num_train_epochs=ns.num_train_epochs,
        per_device_train_batch_size=ns.per_device_train_batch_size,
        per_device_eval_batch_size=ns.per_device_eval_batch_size,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        learning_rate=ns.learning_rate,
        weight_decay=ns.weight_decay,
        warmup_ratio=ns.warmup_ratio,
        max_grad_norm=ns.max_grad_norm,
        logging_steps=ns.logging_steps,
        save_steps=ns.save_steps,
        eval_steps=ns.eval_steps,
        save_total_limit=ns.save_total_limit,
        seed=ns.seed,
        gradient_checkpointing=ns.gradient_checkpointing,
        dataloader_num_workers=ns.dataloader_num_workers,
        eval_accumulation_steps=ns.eval_accumulation_steps,
        early_stopping_patience=ns.early_stopping_patience,
        load_best_model_at_end=ns.load_best_model_at_end,
        metric_for_best_model=ns.metric_for_best_model,
        greater_is_better=ns.greater_is_better,
    )
    return args


def main() -> None:
    args = parse_args()

    # Build run-specific output directory using LoRA hyper-parameters
    def _shorten_targets(targets: Optional[List[str]]) -> str:
        if not targets:
            return "auto"
        short = "-".join([t.replace("_proj", "") for t in targets])
        return short[:48]

    run_name = f"lora_r{args.lora_r}_a{args.lora_alpha}_d{int(args.lora_dropout*100)}_" \
               f"tm-{_shorten_targets(args.lora_target_modules or [])}"
    base_out = args.output_dir
    output_dir = os.path.join(base_out, run_name)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)

    # Decide precision based on hardware support to avoid BF16 kernel errors on CUDA 11.7
    def bf16_supported() -> bool:
        if not torch.cuda.is_available():
            return False
        # Prefer torch API if available
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                return bool(torch.cuda.is_bf16_supported())
            except Exception:
                pass
        # Conservative fallback: require Ampere+ (SM80) and CUDA >= 11.8 typically
        major, minor = torch.cuda.get_device_capability()
        return major >= 8

    use_bf16 = bool(args.bf16 and bf16_supported())
    use_fp16 = bool(args.fp16 or (args.bf16 and not use_bf16 and torch.cuda.is_available()))

    if args.bf16 and not use_bf16 and torch.cuda.is_available():
        print("WARNING: BF16 not supported on this GPU/CUDA; falling back to FP16 mixed precision.")
    if not torch.cuda.is_available():
        use_bf16 = False
        use_fp16 = False

    # Default to SDPA attention if none specified (saves memory vs eager on many GPUs)
    attn_impl = args.attn_impl or "sdpa"

    model, tokenizer = create_model_and_tokenizer(
        base_model_name_or_path=args.base_model,
        use_bf16=use_bf16,
        attn_impl=attn_impl,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
    )

    # Build datasets: prefer explicit train/valid/test paths; fallback to split of a single path
    train_dataset = None
    eval_dataset = None
    test_dataset = None

    def build_ds(path: str):
        return InstructionDataset(
            file_path=path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            system_prompt=args.system_prompt,
            fmt=args.data_format,
            user_field=args.user_field,
            assistant_field=args.assistant_field,
        )

    if args.train_path:
        train_dataset = build_ds(args.train_path)
        if args.valid_path:
            eval_dataset = build_ds(args.valid_path)
        if args.test_path:
            test_dataset = build_ds(args.test_path)
    elif args.data_path:
        dataset = build_ds(args.data_path)
        eval_size = max(1, int(len(dataset) * args.eval_ratio)) if len(dataset) > 0 else 0
        if eval_size > 0 and eval_size < len(dataset):
            train_size = len(dataset) - eval_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset,
                [train_size, eval_size],
                generator=torch.Generator().manual_seed(args.seed),
            )
        else:
            train_dataset = dataset
            eval_dataset = None
    else:
        raise ValueError("Provide either --train_path (and optionally --valid_path/--test_path) or --data_path")

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    # Choose optimizer - attempt to use paged_adamw_8bit when requested and available
    optim_name = "adamw_torch"
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb  # noqa: F401
            optim_name = "paged_adamw_8bit"
        except Exception:
            print("WARNING: bitsandbytes not installed or not available; falling back to adamw_torch.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,  # Gradient clipping
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=("steps" if eval_dataset is not None else "no"),
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        optim=optim_name,
        report_to=["none"],
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        prediction_loss_only=True,
        eval_accumulation_steps=args.eval_accumulation_steps,
        # Best model saving and early stopping
        load_best_model_at_end=args.load_best_model_at_end if eval_dataset is not None else False,
        metric_for_best_model=args.metric_for_best_model if eval_dataset is not None else None,
        greater_is_better=args.greater_is_better,
    )

    def compute_metrics(eval_pred):
        # eval_pred is (logits, labels) or (loss, ...), depending on trainer config
        # Trainer with language modeling returns loss only by default; we compute perplexity from metrics
        # Here we just pass, as Trainer will report eval_loss; we post-process below.
        return {}

    # File logger callback writing JSONL logs under the run directory
    from transformers import TrainerCallback, EarlyStoppingCallback

    class FileLoggingCallback(TrainerCallback):
        def __init__(self, log_file: str) -> None:
            self.log_file = log_file

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": int(state.global_step), **logs}) + "\n")
            except Exception:
                pass

    # Setup callbacks
    callbacks = [FileLoggingCallback(os.path.join(output_dir, "train_logs.jsonl"))]
    
    # Add early stopping if eval dataset is available
    if eval_dataset is not None and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0,  # Stop when no improvement
            )
        )
        print(f"Early stopping enabled: patience={args.early_stopping_patience} evaluations")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    # --- Post-training: generation-based evaluation on test set (BLEU / METEOR / ROUGE-L)
    def _safe_batch_decode(arr):
        return tokenizer.batch_decode(arr, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def _compute_simple_bleu(references, hypotheses):
        # corpus-level BLEU-4 approximation (micro)
        weights = [0.25, 0.25, 0.25, 0.25]
        precisions = []
        for n in range(1, 5):
            num, den = 0, 0
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                ref_ngrams = {}
                for i in range(max(0, len(ref_tokens)-n+1)):
                    ref_ngrams[tuple(ref_tokens[i:i+n])] = ref_ngrams.get(tuple(ref_tokens[i:i+n]), 0) + 1
                hyp_ngrams = {}
                for i in range(max(0, len(hyp_tokens)-n+1)):
                    hyp_ngrams[tuple(hyp_tokens[i:i+n])] = hyp_ngrams.get(tuple(hyp_tokens[i:i+n]), 0) + 1
                match = 0
                for g, cnt in hyp_ngrams.items():
                    match += min(cnt, ref_ngrams.get(g, 0))
                num += match
                den += sum(hyp_ngrams.values())
            precisions.append((num / den) if den > 0 else 0.0)

        smooth = 1e-9
        score = 1.0
        for p in precisions:
            score *= (p + smooth) ** (1.0/4.0)

        ref_len = sum(len(r.split()) for r in references)
        hyp_len = sum(len(h.split()) for h in hypotheses)
        bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / (hyp_len + 1e-9))
        return bp * score

    def _compute_simple_meteor(reference, hypothesis):
        r_tokens = reference.split()
        h_tokens = hypothesis.split()
        common = 0
        r_count = {}
        for t in r_tokens:
            r_count[t] = r_count.get(t, 0) + 1
        for t in h_tokens:
            if r_count.get(t, 0) > 0:
                common += 1
                r_count[t] -= 1
        if common == 0:
            return 0.0
        prec = common / max(1, len(h_tokens))
        rec = common / max(1, len(r_tokens))
        beta = 1.0
        return (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-9)

    def _compute_rouge_l(reference, hypothesis):
        a = reference.split()
        b = hypothesis.split()
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if a[i] == b[j]:
                    dp[i][j] = dp[i+1][j+1] + 1
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j+1])
        lcs = dp[0][0]
        prec = lcs / max(1, n)
        rec = lcs / max(1, m)
        if prec + rec == 0:
            return 0.0
        return (2 * prec * rec) / (prec + rec)

    if test_dataset is not None and len(test_dataset) > 0:
        try:
            pred_out = trainer.predict(test_dataset, predict_with_generate=True)
            raw_preds = pred_out.predictions
            if isinstance(raw_preds, tuple):
                raw_preds = raw_preds[0]
            decoded_preds = _safe_batch_decode(raw_preds)

            label_ids = pred_out.label_ids
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            if label_ids is None:
                decoded_refs = ["" for _ in decoded_preds]
            else:
                label_ids = np.array(label_ids)
                label_ids[label_ids == -100] = pad_id
                decoded_refs = _safe_batch_decode(label_ids)

            if nltk is not None:
                refs_for_bleu = [[r.split()] for r in decoded_refs]
                hyps_for_bleu = [h.split() for h in decoded_preds]
                try:
                    smoothie = SmoothingFunction().method4
                    bleu_score = corpus_bleu(refs_for_bleu, hyps_for_bleu, smoothing_function=smoothie)
                except Exception:
                    bleu_score = _compute_simple_bleu(decoded_refs, decoded_preds)
                try:
                    meteor_scores = [meteor_score([r], h) for r, h in zip(decoded_refs, decoded_preds)]
                    meteor = float(np.mean(meteor_scores)) if len(meteor_scores) > 0 else 0.0
                except Exception:
                    meteor = float(np.mean([_compute_simple_meteor(r, h) for r, h in zip(decoded_refs, decoded_preds)]))
            else:
                bleu_score = _compute_simple_bleu(decoded_refs, decoded_preds)
                meteor = float(np.mean([_compute_simple_meteor(r, h) for r, h in zip(decoded_refs, decoded_preds)]))

            if rouge_scorer is not None:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_ls = [scorer.score(r, h)['rougeL'].fmeasure for r, h in zip(decoded_refs, decoded_preds)]
                rouge_l = float(np.mean(rouge_ls)) if len(rouge_ls) > 0 else 0.0
            else:
                rouge_l = float(np.mean([_compute_rouge_l(r, h) for r, h in zip(decoded_refs, decoded_preds)]))

            gen_metrics = {
                "BLEU": float(bleu_score),
                "METEOR": float(meteor),
                "ROUGE-L": float(rouge_l),
                "num_examples": len(decoded_preds),
            }
            try:
                with open(os.path.join(output_dir, "test_generation_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(gen_metrics, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            print("Generation metrics:", gen_metrics)
        except Exception as e:
            print("Generation-based evaluation failed:", e)

    # --- Save LoRA -> core parameter mapping for inspection
    def _extract_lora_mappings(model):
        mappings = {}
        for name, _ in model.named_parameters():
            if 'lora' in name.lower() or '.lora' in name:
                core = name
                for suf in ['.lora_A', '.lora_B', '.lora_alpha', '.lora_dropout', '.lora_bias', 'lora_']:
                    core = core.replace(suf, '')
                core = core.replace('..', '.')
                mappings[name] = core
        return mappings

    try:
        lora_map = _extract_lora_mappings(trainer.model)
        with open(os.path.join(output_dir, "lora_core_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(lora_map, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    metrics = {}
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics and eval_metrics["eval_loss"] is not None:
            try:
                eval_ppl = math.exp(eval_metrics["eval_loss"]) if eval_metrics["eval_loss"] < 20 else float("inf")
            except OverflowError:
                eval_ppl = float("inf")
            eval_metrics["eval_perplexity"] = eval_ppl
        metrics.update({f"valid_{k}": v for k, v in eval_metrics.items()})
        print({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
        # Persist validation metrics
        try:
            with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    if test_dataset is not None and len(test_dataset) > 0:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        if "eval_loss" in test_metrics and test_metrics["eval_loss"] is not None:
            try:
                test_ppl = math.exp(test_metrics["eval_loss"]) if test_metrics["eval_loss"] < 20 else float("inf")
            except OverflowError:
                test_ppl = float("inf")
            test_metrics["eval_perplexity"] = test_ppl
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        print({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
        # Persist test metrics
        try:
            with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Save PEFT adapter and tokenizer (LoRA only; base model is not re-saved)
    trainer.model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Persist run configuration for future comparison
    run_cfg = {
        "base_model": args.base_model,
        "attn_impl": attn_impl,
        "precision": {"bf16": use_bf16, "fp16": use_fp16},
        "seed": args.seed,
        "max_length": args.max_length,
        "train_sizes": {
            "train": len(train_dataset) if train_dataset is not None else 0,
            "valid": len(eval_dataset) if eval_dataset is not None else 0,
            "test": len(test_dataset) if test_dataset is not None else 0,
        },
        "optimizer": optim_name,
        "training_args": {
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "early_stopping_patience": args.early_stopping_patience,
            "load_best_model_at_end": args.load_best_model_at_end,
            "metric_for_best_model": args.metric_for_best_model,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": args.lora_target_modules,
        },
        "output_dir": output_dir,
        "run_name": run_name,
    }
    try:
        with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()


