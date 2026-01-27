import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _ensure_eos(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.eos_token_id is None:
        # Fallback: add eos if missing
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    return int(tokenizer.eos_token_id)


def format_chat_messages(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: Optional[str],
    user_content: str,
    assistant_content: Optional[str] = None,
) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    # For training we want the full dialogue rendered as text
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        truncation=False,
    )
    return rendered


class InstructionDataset(Dataset):
    """
    A flexible dataset for instruction tuning.

    Supported formats:
    - JSONL where each line has fields for user/instruction and assistant/response.
    - TSV (tab-separated) lines: user \t assistant

    Label masking: user/context tokens are masked to -100 so loss applies only to
    assistant continuation tokens.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
        fmt: str = "jsonl",  # "jsonl" or "tsv"
        user_field: str = "instruction",
        assistant_field: str = "output",
        strip_empty: bool = True,
    ) -> None:
        self.examples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.user_field = user_field
        self.assistant_field = assistant_field
        self.eos_id = _ensure_eos(tokenizer)

        if fmt not in {"jsonl", "tsv"}:
            raise ValueError("fmt must be 'jsonl' or 'tsv'")

        if fmt == "jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    user_text = obj.get(user_field, "")
                    assistant_text = obj.get(assistant_field, "")
                    if strip_empty and (not user_text or not assistant_text):
                        continue
                    self.examples.append(
                        {"user": user_text, "assistant": assistant_text}
                    )
        else:  # tsv
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        if strip_empty:
                            continue
                        parts = parts + [""] * (2 - len(parts))
                    self.examples.append({"user": parts[0], "assistant": parts[1]})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        user_text: str = ex["user"]
        assistant_text: str = ex["assistant"]

        # Build a rendered context containing system + user only, ending with the model's "generation prompt"
        # We render two strings:
        # 1) context_str: system+user with add_generation_prompt=True (assistant start)
        # 2) full_str: system+user+assistant with add_generation_prompt=False
        # We will tokenize both to compute the boundary where labels start.

        context_str = self.tokenizer.apply_chat_template(
            (
                ([{"role": "system", "content": self.system_prompt}]
                 if self.system_prompt else [])
                + [{"role": "user", "content": user_text}]
            ),
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
        )

        full_str = format_chat_messages(
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            user_content=user_text,
            assistant_content=assistant_text,
        )

        context_ids = self.tokenizer(
            context_str, add_special_tokens=False
        )["input_ids"]
        full_ids = self.tokenizer(
            full_str, add_special_tokens=False
        )["input_ids"]

        # Ensure an EOS at end of assistant
        if not full_ids or full_ids[-1] != self.eos_id:
            full_ids = full_ids + [self.eos_id]

        # Truncate from the left if too long (keep the end since labels are at the end)
        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            # Shift both sequences keeping alignment where possible
            full_ids = full_ids[overflow:]
            context_cut = max(0, len(context_ids) - overflow)
            context_ids = context_ids[overflow:] if overflow < len(context_ids) else []
        else:
            context_cut = len(context_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Mask labels up to the assistant start boundary
        assistant_start = min(context_cut, labels.size(0))
        labels[:assistant_start] = -100

        return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: str = "longest"
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        batch_input = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels manually with label_pad_token_id
        max_len = batch_input["input_ids"].size(1)
        batch_labels = torch.full(
            (len(labels), max_len), self.label_pad_token_id, dtype=torch.long
        )
        for i, lbl in enumerate(labels):
            length = min(lbl.size(0), max_len)
            batch_labels[i, :length] = lbl[:length]

        attention_mask = (batch_input["input_ids"] != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": batch_input["input_ids"],
            "attention_mask": attention_mask,
            "labels": batch_labels,
        }


