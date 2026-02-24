"""
Data loaders for code-related tasks

Supports:
- code2nl: Code summarization (JCSD dataset)
- code2code: Assertion generation (ATLAS dataset)
- nl2code: Code generation (conCode dataset)
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
import torch


class JCSDDataLoader:
    """
    Loader for Java Code Summarization Dataset (JCSD)

    Format: {"code": "...", "summary": "..."}
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 128,
    ):
        """
        Args:
            data_dir: Directory containing train.json, dev.json, test.json
            tokenizer: Tokenizer for the model
            max_source_length: Maximum length for source code
            max_target_length: Maximum length for summary
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def load_file(self, split: str) -> List[Dict]:
        """Load data from JSON file"""
        file_path = os.path.join(self.data_dir, f"{split}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def preprocess_function(self, examples):
        """Preprocess examples for seq2seq generation"""
        # Tokenize inputs (code)
        model_inputs = self.tokenizer(
            examples['code'],
            max_length=self.max_source_length,
            truncation=True,
            padding=False,  # We'll pad dynamically in collator
        )

        # Tokenize targets (summary)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['summary'],
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_dataset(self) -> DatasetDict:
        """Load and preprocess the entire dataset"""
        # Load data
        train_data = self.load_file('train')
        dev_data = self.load_file('dev')
        test_data = self.load_file('test')

        # Convert to HuggingFace Dataset format
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(dev_data),
            'test': Dataset.from_list(test_data),
        })

        # Preprocess
        tokenized_dataset = dataset_dict.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset_dict['train'].column_names,
        )

        return tokenized_dataset


class ATLASDataLoader:
    """
    Loader for ATLAS (Assertion Generation) Dataset

    Format: {"focal_method": "...", "test_prefix": "...", "assertion": "..."}
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 128,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def load_file(self, split: str) -> List[Dict]:
        """Load data from JSON file"""
        file_path = os.path.join(self.data_dir, f"{split}.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def preprocess_function(self, examples):
        """Preprocess examples"""
        # Combine focal method and test prefix as input
        inputs = [
            f"Focal Method:\n{focal}\n\nTest Prefix:\n{prefix}"
            for focal, prefix in zip(examples['focal_method'], examples['test_prefix'])
        ]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )

        # Tokenize assertions
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['assertion'],
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_dataset(self) -> DatasetDict:
        """Load and preprocess the entire dataset"""
        train_data = self.load_file('train')
        dev_data = self.load_file('dev')
        test_data = self.load_file('test')

        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(dev_data),
            'test': Dataset.from_list(test_data),
        })

        tokenized_dataset = dataset_dict.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset_dict['train'].column_names,
        )

        return tokenized_dataset


class ConCodeDataLoader:
    """
    Loader for conCode (Code Generation) Dataset

    Format: {"nl": "...", "code": "..."}
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 128,
        max_target_length: int = 512,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def load_file(self, split: str) -> List[Dict]:
        """Load data from JSON file"""
        file_path = os.path.join(self.data_dir, f"{split}.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def preprocess_function(self, examples):
        """Preprocess examples"""
        # Tokenize NL descriptions
        model_inputs = self.tokenizer(
            examples['nl'],
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )

        # Tokenize code
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['code'],
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_dataset(self) -> DatasetDict:
        """Load and preprocess the entire dataset"""
        train_data = self.load_file('train')
        dev_data = self.load_file('dev')
        test_data = self.load_file('test')

        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(dev_data),
            'test': Dataset.from_list(test_data),
        })

        tokenized_dataset = dataset_dict.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset_dict['train'].column_names,
        )

        return tokenized_dataset


def get_data_loader(
    task_type: str,
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    max_source_length: int = 512,
    max_target_length: int = 128,
):
    """
    Factory function to get appropriate data loader

    Args:
        task_type: One of 'code2nl', 'code2code', 'nl2code'
        data_dir: Directory containing the dataset
        tokenizer: Tokenizer
        max_source_length: Max length for source
        max_target_length: Max length for target

    Returns:
        DataLoader instance
    """
    if task_type == 'code2nl':
        return JCSDDataLoader(data_dir, tokenizer, max_source_length, max_target_length)
    elif task_type == 'code2code':
        return ATLASDataLoader(data_dir, tokenizer, max_source_length, max_target_length)
    elif task_type == 'nl2code':
        return ConCodeDataLoader(data_dir, tokenizer, max_source_length, max_target_length)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# Data collator for seq2seq tasks
class DataCollatorForSeq2Seq:
    """
    Data collator for sequence-to-sequence tasks
    Handles padding for both inputs and labels
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: Optional[torch.nn.Module] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        # Separate labels from inputs
        labels = [feature["labels"] for feature in features] if "labels" in features[0] else None

        # Remove labels from features for padding
        if labels is not None:
            for feature in features:
                feature.pop("labels")

        # Pad inputs
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels if present
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            # Pad labels
            padded_labels = []
            for label in labels:
                remainder = [self.label_pad_token_id] * (max_label_length - len(label))
                padded_labels.append(label + remainder)

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch
