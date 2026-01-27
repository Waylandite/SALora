from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model


def _maybe_set_pad_to_eos(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is None:
        # Use EOS as PAD for causal LM training
        tokenizer.pad_token = tokenizer.eos_token


def create_model_and_tokenizer(
    base_model_name_or_path: str = "Qwen/Qwen2.5-Coder-3B",
    use_bf16: bool = True,
    attn_impl: Optional[str] = None,
    gradient_checkpointing: bool = True,
    # LoRA params (edit these to run experiments)
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    lora_bias: str = "none",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads Qwen2.5-Coder base model and applies LoRA using PEFT.

    LoRA parameters are surfaced here for easy experimentation.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    _maybe_set_pad_to_eos(tokenizer)

    # Choose dtype: prefer BF16 when requested and supported; else FP16 on CUDA; else default
    if torch.cuda.is_available():
        if use_bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = None

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **model_kwargs,
    )

    # Reduce memory during training
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # Default target modules for common transformer linear layers
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    # Print trainable parameter ratio for quick verification
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model, tokenizer


