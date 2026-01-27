## Qwen2.5-Coder LoRA Fine-tuning (PEFT)

### Environment Setup

```bash
# 1) Create and activate conda env
conda create -n qwen-peft python=3.10 -y
conda activate qwen-peft

# 2) Install PyTorch (pick the wheel matching your CUDA)
# Example: CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Or CPU-only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3) Install project deps (includes nltk / rouge-score for metrics)
pip install -r requirements.txt

# 4) (Optional) Download model weights locally
python modeldownload.py --model Qwen/Qwen2.5-Coder-3B --output ./models/qwen2.5-coder-3b
```

### Environment Check

Before training, check your environment:

```bash
python check_env.py
```

This will verify:
- PyTorch installation and CUDA availability
- Required dependencies
- CUDA library paths

### Troubleshooting CUDA Issues

If you encounter `libcudart.so.12` or `libcublas.so.*` errors:

**Solution 1: Set CUDA library path**
```bash
# Find your CUDA installation (common locations)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Or if CUDA is in a different location:
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

**Solution 2: Check CUDA installation**
```bash
# Check if CUDA is installed
nvcc --version
# Or
ls /usr/local/cuda*/lib64/libcudart.so*

# Find the correct path and add to LD_LIBRARY_PATH
```

**Solution 3: Reinstall PyTorch with matching CUDA version**
```bash
# Check your CUDA version first
nvcc --version

# Install matching PyTorch (example for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Solution 4: Use CPU-only (very slow, not recommended for training)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Then remove --bf16 flag when training
```

### Data format
- JSONL: one JSON object per line with fields `instruction` and `output` (configurable)
- TSV: each line is `user\tassistant`

### Train

**For code summarization task (JCSD dataset):**

```bash
python train.py \
  --train_path /home/wuruifeng/data/wuruifeng/data/JCSD/train.jsonl \
  --valid_path /home/wuruifeng/data/wuruifeng/data/JCSD/valid.jsonl \
  --test_path /home/wuruifeng/data/wuruifeng/data/JCSD/test.jsonl \
  --data_format jsonl \
  --user_field code \
  --assistant_field comment \
  --system_prompt "You are a helpful coding assistant. Summarize the code succinctly." \
  --base_model Qwen/Qwen2.5-Coder-3B \
  --bf16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_bias none \
  --output_dir ./outputs/qwen2p5_lora_codesum \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4
```

**For general instruction tuning:**

```bash
python train.py \
  --data_path /path/to/dataset.jsonl \
  --data_format jsonl \
  --user_field instruction \
  --assistant_field output \
  --system_prompt "You are a helpful coding assistant." \
  --base_model Qwen/Qwen2.5-Coder-3B \
  --bf16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --output_dir ./outputs/qwen2p5_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4
```

### LoRA Parameters for Experiments

Key LoRA knobs for experiments (edit or pass via CLI):
- `--lora_r`: Rank of LoRA (e.g., 8, 16, 32, 64)
- `--lora_alpha`: LoRA alpha scaling (typically 2x r, e.g., 16, 32, 64)
- `--lora_dropout`: Dropout rate (e.g., 0.05, 0.1, 0.2)
- `--lora_target_modules`: Which modules to apply LoRA (comma-separated)
- `--lora_bias`: Bias handling ("none", "all", "lora_only")

**Example experiment variations:**

```bash
# Low rank experiment
--lora_r 8 --lora_alpha 16

# High rank experiment  
--lora_r 64 --lora_alpha 128

# Different dropout
--lora_dropout 0.1

# Only attention layers
--lora_target_modules q_proj,k_proj,v_proj,o_proj
```

Artifacts (adapter) and tokenizer are saved to `--output_dir`.

> Note: training only saves the LoRA adapters (not the full base model). To keep storage minimal, avoid re-downloading the base model when saving checkpoints.


