"""
Configuration for SALora experiments
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SALoraConfig:
    """Configuration for SALora"""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    num_labels: int = None
    task_type: str = "causal_lm"  # 'causal_lm', 'sequence_classification', 'token_classification'

    # LoRA settings
    lora_r: int = 8  # Maximum rank
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    modules_to_replace: List[str] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )

    # Architecture search settings
    arch_init_strategy: str = "uniform"  # 'uniform', 'high_rank', 'low_rank'
    lambda_spectral: float = 1e-4  # Weight for spectral intrusion penalty
    lambda_l1: float = 1e-3  # Weight for L1 regularization
    spectral_keep_ratio: float = 0.1  # Ratio of singular vectors to keep (top 10%)

    # Training settings
    learning_rate: float = 3e-4  # Learning rate for model parameters
    arch_learning_rate: float = 3e-4  # Learning rate for architecture parameters
    weight_decay: float = 0.01
    arch_weight_decay: float = 1e-3
    num_epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # Bilevel optimization settings
    unroll_steps: int = 1  # Number of inner loop steps per outer loop step
    train_split_ratio: float = 0.7  # Ratio of training data for inner loop
    valid_step: int = 100  # Validation frequency
    roll_back: bool = True  # Whether to roll back model weights after outer loop

    # Generation settings (for seq2seq tasks)
    max_source_length: int = 512
    max_target_length: int = 128
    num_beams: int = 5
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0

    # Other settings
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Dataset settings
    dataset_name: str = "jcsd"
    dataset_config: str = None
    data_dir: str = "./data/jcsd"  # Path to dataset directory

    def __post_init__(self):
        """Validate configuration"""
        assert self.lora_r > 0, "lora_r must be positive"
        assert 0 < self.train_split_ratio < 1, "train_split_ratio must be in (0, 1)"
        assert self.lambda_spectral >= 0, "lambda_spectral must be non-negative"
        assert self.lambda_l1 >= 0, "lambda_l1 must be non-negative"


@dataclass
class Code2NLConfig(SALoraConfig):
    """Configuration for Code Summarization task (JCSD)"""
    dataset_name: str = "jcsd"
    data_dir: str = "./data/jcsd"
    task_type: str = "causal_lm"
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    max_source_length: int = 512
    max_target_length: int = 128
    num_beams: int = 5
    batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 10


@dataclass
class Code2CodeConfig(SALoraConfig):
    """Configuration for Assertion Generation task (ATLAS)"""
    dataset_name: str = "atlas"
    data_dir: str = "./data/atlas"
    task_type: str = "causal_lm"
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    max_source_length: int = 512
    max_target_length: int = 128
    num_beams: int = 5
    batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 10


@dataclass
class NL2CodeConfig(SALoraConfig):
    """Configuration for Code Generation task (conCode)"""
    dataset_name: str = "concode"
    data_dir: str = "./data/concode"
    task_type: str = "causal_lm"
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    max_source_length: int = 128
    max_target_length: int = 512
    num_beams: int = 5
    batch_size: int = 4
    eval_batch_size: int = 8
    num_epochs: int = 10
