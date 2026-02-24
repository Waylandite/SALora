# SA-AutoLoRA Quick Start Guide

## 📦 Installation

```bash
# Create virtual environment
conda create -n sa_autolora python=3.9
conda activate sa_autolora

# Install dependencies
pip install torch transformers datasets
pip install betty-ml  # For bilevel optimization
pip install matplotlib seaborn tqdm
pip install peft  # For baseline comparison (optional)
```

## 🚀 Quick Start

### 1. Basic Usage (Two-Stage Training)

```bash
# Train on SST-2 with BERT-base
python experiment_script.py \
    --task sst2 \
    --model_name bert-base-uncased \
    --r_max 8 \
    --lambda_spectral 1e-4 \
    --gamma_l1 1e-3 \
    --training_mode two_stage \
    --warmup_epochs 5 \
    --search_epochs 10 \
    --batch_size 16
```

### 2. Advanced Usage (Betty Bilevel Optimization)

```bash
python experiment_script.py \
    --task mrpc \
    --model_name roberta-base \
    --r_max 16 \
    --lambda_spectral 1e-4 \
    --training_mode betty \
    --batch_size 8
```

### 3. Quick Test (Small Dataset)

```bash
# Use subset of data for quick testing
python experiment_script.py \
    --task sst2 \
    --train_size 1000 \
    --val_size 200 \
    --warmup_epochs 2 \
    --search_epochs 3 \
    --batch_size 32
```

## 📝 Code Example

```python
from transformers import AutoModelForSequenceClassification
from sa_autolora import SAAutoLoRAConfig, replace_linear_with_lora, TwoStageTrainer

# 1. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=2
)

# 2. Configure SA-AutoLoRA
config = SAAutoLoRAConfig(
    r_max=8,
    target_modules=['query', 'key', 'value', 'output'],
    lambda_spectral=1e-4,
    gamma_l1=1e-3,
)

# 3. Replace layers
model, lora_layers = replace_linear_with_lora(model, config)

# 4. Train (assuming you have data loaders)
trainer = TwoStageTrainer(
    model=model,
    lora_layers=lora_layers,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

trainer.train()

# 5. Analyze results
for name, layer in lora_layers.items():
    print(f"{name}: effective_rank={layer.get_effective_rank()}/{config.r_max}")
```

## 🔬 Hyperparameter Tuning Guide

### Starting Values

| Parameter | Recommended Start | Range to Explore | Notes |
|-----------|------------------|------------------|-------|
| `r_max` | 8 | [4, 8, 16] | Higher = more capacity but slower |
| `lambda_spectral` | 1e-4 | [0, 1e-5, 1e-4, 1e-3] | 0 = no spectral constraint |
| `gamma_l1` | 1e-3 | [1e-4, 1e-3, 1e-2] | Higher = more sparsity |
| `lr_theta` | 1e-4 | [5e-5, 1e-4, 2e-4] | LoRA parameter learning rate |
| `lr_alpha` | 1e-2 | [5e-3, 1e-2, 5e-2] | Alpha learning rate |

### Tuning Strategy

**Phase 1: Find Best λ and γ**
```bash
for lambda in 0 1e-5 1e-4 1e-3; do
  for gamma in 1e-4 1e-3 1e-2; do
    python experiment_script.py \
      --lambda_spectral $lambda \
      --gamma_l1 $gamma \
      --output_dir "./tuning/lambda_${lambda}_gamma_${gamma}"
  done
done
```

**Phase 2: Optimize Rank**
```bash
# Use best λ and γ from Phase 1
for r_max in 4 8 16; do
  python experiment_script.py \
    --r_max $r_max \
    --lambda_spectral 1e-4 \
    --gamma_l1 1e-3 \
    --output_dir "./tuning/rmax_${r_max}"
done
```

## 📊 Understanding the Results

### Output Files

After training, check `experiments/{task}_{model}_{timestamp}/`:

```
├── results.json           # Final metrics
├── model.pt              # Trained model checkpoint
├── alpha_heatmap.png     # Visualization of α values
├── training_curves.png   # Loss and accuracy curves
└── args.json             # Experiment configuration
```

### Key Metrics to Check

1. **Final Accuracy**: Compare to baseline LoRA
2. **Sparsity**: % of α values < 0.1 (target: 20-50%)
3. **Effective Ranks**: Active ranks per module
4. **Spectral Intrusion**: Should decrease during training

### Example Output

```json
{
  "final_accuracy": 0.923,
  "alpha_analysis": {
    "overall": {
      "mean": 0.42,
      "sparsity": 0.35,
      "active_ratio": 0.65
    },
    "per_module": {
      "layer.0.attention.self.query": {
        "effective_rank": 6,
        "mean": 0.58
      },
      "layer.0.attention.self.key": {
        "effective_rank": 3,
        "mean": 0.31
      }
    }
  }
}
```

## 🐛 Troubleshooting

### Issue 1: Training Loss Not Decreasing

**Symptoms**: Loss stays flat in Stage 1

**Solutions**:
- Increase `lr_theta` to 2e-4 or 5e-4
- Check if data loading is correct
- Reduce batch size if GPU memory is full

```bash
python experiment_script.py \
    --lr_theta 5e-4 \
    --batch_size 8
```

### Issue 2: All Alpha Values → 0 or 1

**Symptoms**: No sparsity pattern, all αs are extremes

**Solutions**:
- Reduce `lr_alpha` to 5e-3
- Adjust `gamma_l1` to 5e-4 (less penalty)
- Increase `warmup_epochs` to 10

```bash
python experiment_script.py \
    --lr_alpha 5e-3 \
    --gamma_l1 5e-4 \
    --warmup_epochs 10
```

### Issue 3: GPU Out of Memory

**Solutions**:
```bash
# Option 1: Reduce batch size
python experiment_script.py --batch_size 4

# Option 2: Use gradient accumulation (modify code)
# Option 3: Reduce r_max
python experiment_script.py --r_max 4
```

### Issue 4: SVD Computation Slow

**Symptoms**: Training very slow during Stage 2

**Solutions**:
- Reduce `k_ratio` from 0.1 to 0.05
- Compute projection operators less frequently (modify code)
- Use smaller model (bert-small instead of bert-base)

## 📈 Experimental Design Template

### Ablation Study

```bash
# 1. No spectral constraint (λ=0)
python experiment_script.py --lambda_spectral 0 --output_dir ./ablation/no_spectral

# 2. No L1 penalty (γ=0)
python experiment_script.py --gamma_l1 0 --output_dir ./ablation/no_l1

# 3. Full SA-AutoLoRA
python experiment_script.py --output_dir ./ablation/full

# 4. Only Q,V modules (like standard AutoLoRA)
python experiment_script.py \
    --target_modules query value \
    --output_dir ./ablation/qv_only
```

### Multi-Task Comparison

```bash
for task in sst2 mrpc qnli qqp; do
  python experiment_script.py \
    --task $task \
    --output_dir "./multitask/$task"
done
```

## 🎯 Expected Results

### Performance Targets (BERT-base on GLUE)

| Task | Baseline LoRA | SA-AutoLoRA (Expected) |
|------|--------------|----------------------|
| SST-2 | 92.0% | 92.5% - 93.0% |
| MRPC | 88.0% | 88.5% - 89.0% |
| QNLI | 91.0% | 91.5% - 92.0% |

### Parameter Efficiency

- **Baseline LoRA** (r=8, q+v only): ~0.6M params
- **SA-AutoLoRA** (r_max=8, all modules): ~0.4M effective params (after pruning)
- **Sparsity**: 30-50% of ranks pruned

## 💡 Tips for Better Results

1. **Start Small**: Use `--train_size 1000` for quick iterations
2. **Monitor Alpha**: Check if patterns make sense (e.g., attention vs FFN)
3. **Patience**: Two-stage training is more stable than bilevel
4. **Visualization**: Always check `alpha_heatmap.png` for insights
5. **Comparison**: Run baseline LoRA for fair comparison

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{sa_autolora2025,
  title={SA-AutoLoRA: Spectral-Aware Meta-Learning for Automated Multi-Module Low-Rank Adaptation},
  author={Your Name},
  year={2025}
}
```

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue!

## 📄 License

MIT License - feel free to use for research and commercial purposes.