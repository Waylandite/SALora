# SALora：用于Qwen2.5-Coder的代码任务自适应LoRA

SALora (Spectral-Aware AutoLoRA) 是针对代码相关任务使用Qwen2.5-Coder模型的实现。

## ✨ PEFT 0.7.0+支持完整的逐层逐模块Rank配置

**好消息！** PEFT 0.7.0+支持`rank_pattern`参数，允许SALora使用完整的逐层逐模块rank配置而无需任何压缩！这保留了所有发现的rank以获得最佳性能。

📖 详细文档请参阅[RANK_PATTERN_GUIDE.md](RANK_PATTERN_GUIDE.md)

## 支持的任务

| 任务类型 | 数据集 | 训练集 | 验证集 | 测试集 | 评估指标 |
|---------|--------|--------|--------|--------|---------|
| code2nl (代码摘要) | JCSD | 69,708 | 8,714 | 8,714 | BLEU, METEOR, ROUGE-L |
| code2code (断言生成) | ATLAS | 125,408 | 15,676 | 15,676 | 精确匹配, SAM, CodeBLEU |
| nl2code (代码生成) | conCode | 100,000 | 2,000 | 2,000 | Pass@k, CodeBLEU, 精确匹配 |

## 快速开始

### 安装依赖

```bash
# 安装依赖包
pip install torch transformers datasets
pip install peft>=0.7.0  # ⚠️ 需要0.7.0+版本以支持rank_pattern！
pip install nltk rouge-score

# 安装Betty框架（用于双层优化）
cd ../AutoLoRA/betty
pip install -e .
```

**注意：** 对于旧版本的PEFT（<0.7.0），SALora将自动使用压缩策略作为降级方案。

### 1. 准备数据集

按以下结构组织数据集：

```
data/
├── jcsd/          # 代码摘要
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── atlas/         # 断言生成
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── concode/       # 代码生成
    ├── train.json
    ├── dev.json
    └── test.json
```

**数据集格式：**

- **JCSD (code2nl):** `{"code": "...", "summary": "..."}`
- **ATLAS (code2code):** `{"focal_method": "...", "test_prefix": "...", "assertion": "..."}`
- **conCode (nl2code):** `{"nl": "...", "code": "..."}`

### 2. 运行SALora架构搜索

搜索最优的LoRA rank配置：

```bash
cd main

# 代码摘要 (code2nl)
python run_code_search.py \
    --task code2nl \
    --data_dir ./data/jcsd \
    --model_name Qwen/Qwen2.5-Coder-1.5B \
    --output_dir ./output_code2nl \
    --lora_r 8 \
    --num_epochs 10 \
    --batch_size 8

# 断言生成 (code2code)
python run_code_search.py \
    --task code2code \
    --data_dir ./data/atlas \
    --output_dir ./output_code2code \
    --lora_r 8

# 代码生成 (nl2code)
python run_code_search.py \
    --task nl2code \
    --data_dir ./data/concode \
    --output_dir ./output_nl2code \
    --lora_r 8
```

**输出文件：**
- `peft_config.json` - 完整配置，包含rank_pattern
- `peft_config_rank_pattern.json` - 仅rank_pattern

### 3. 使用标准PEFT进行验证

使用标准PEFT库应用发现的配置：

#### 方案A: 使用rank_pattern (PEFT 0.7.0+, 推荐)

```bash
# 默认：自动使用完整的rank_pattern
python verify_with_peft.py \
    --salora_config ./output_code2nl/peft_config.json \
    --task code2nl \
    --data_dir ./data/jcsd \
    --model_name Qwen/Qwen2.5-Coder-1.5B \
    --output_dir ./output_peft_verify
```

这会使用SALora发现的完整逐层逐模块rank，无需任何压缩！

#### 方案B: 使用降级方案（旧版PEFT或单一全局rank）

```bash
# 强制使用降级的全局rank
python verify_with_peft.py \
    --salora_config ./output_code2nl/peft_config.json \
    --task code2nl \
    --data_dir ./data/jcsd \
    --no_rank_pattern \
    ...
```

降级方案使用从所有逐层逐模块rank计算的全局中位数rank。

## 输出文件

运行SALora搜索后，会生成：

```
output_code2nl/
├── config.json                      # 训练配置
├── model.pt                         # 训练的模型权重
├── architecture.pt                  # 架构参数（alphas）
├── peft_config.json                 # PEFT配置（含rank_pattern）⭐
├── peft_config_rank_pattern.json    # 仅rank_pattern（便于加载）
├── results.json                     # 评估指标和rank汇总
└── sample_predictions.json          # 样例预测结果
```

### 分析结果

使用分析脚本检查发现的配置：

```bash
python analyze_salora_results.py ./output_code2nl/peft_config.json
```

输出示例：
```
SALora配置分析
======================================================================

✅ 使用rank_pattern (PEFT 0.7.0+)

rank_pattern: 84个层-模块组合
  层数: 12
  每层模块数: 7

各模块类型统计:
  query   : 中位数=7, 均值=6.2, 范围=[3, 8]
  key     : 中位数=2, 均值=2.1, 范围=[0, 3]  ← 经常被剪枝
  value   : 中位数=8, 均值=7.3, 范围=[4, 8]  ← 高重要性
  ...
```

### 理解`peft_config.json`

该文件包含发现的最优LoRA配置及**完整的逐层逐模块rank**：

```json
{
  "notice": "✅ 此配置使用rank_pattern (PEFT 0.7.0+)实现完整的逐层逐模块rank！",
  "version": "2.0",

  "peft_config": {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "rank_pattern": {                // ⭐ 完整的逐层逐模块rank！
      "model.layers.0.self_attn.q_proj": 7,
      "model.layers.0.self_attn.k_proj": 2,
      "model.layers.0.self_attn.v_proj": 8,
      "model.layers.0.self_attn.o_proj": 6,
      "model.layers.0.mlp.gate_proj": 7,
      "model.layers.0.mlp.up_proj": 8,
      "model.layers.0.mlp.down_proj": 6,
      "model.layers.1.self_attn.q_proj": 6,
      ...  // 所有 n_layers × 7 个模块
    },
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
  },

  "peft_config_fallback": {          // 用于旧版PEFT (<0.7.0)
    "r": 6,  // 单一全局rank（所有rank的中位数）
    ...
  },

  "rank_pattern": {                  // 单独保存便于加载
    "model.layers.0.self_attn.q_proj": 7,
    ...
  },

  "salora_full_config": {            // 完整的SALora搜索结果
    "layer_module_ranks": {
      "layer.0.query": 7,
      "layer.0.key": 2,
      ...
    },
    "rank_summary_by_module_type": {
      "query": [7, 6, 5, 6, 7, 8, 5, 6, 7, 5, 4, 3],
      "key": [2, 3, 2, 2, 1, 2, 3, 2, 1, 1, 1, 0],
      ...
    }
  }
}
```

**关键部分：**

- **`peft_config`**: 包含`rank_pattern`的即用PEFT配置（PEFT 0.7.0+）
- **`rank_pattern`**: 单独副本便于使用
- **`peft_config_fallback`**: 用于旧版PEFT的单一全局rank
- **`salora_full_config`**: SALora格式的完整搜索结果

**直接使用：**

```python
import json
from peft import LoraConfig, get_peft_model

# 加载并直接使用rank_pattern
with open('./output_code2nl/peft_config.json') as f:
    config = json.load(f)

# 方案1: 使用完整配置
lora_config = LoraConfig(**config['peft_config'])

# 方案2: 仅加载rank_pattern
with open('./output_code2nl/peft_config_rank_pattern.json') as f:
    rank_pattern = json.load(f)
lora_config = LoraConfig(
    rank_pattern=rank_pattern,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, lora_config)
```

## 核心特性

### 1. 全模块覆盖

SALora在所有Transformer模块中搜索最优rank：
- **注意力层：** Q, K, V, O投影
- **MLP层：** gate_proj, up_proj, down_proj（Qwen2架构）

### 2. 完整的逐层逐模块Rank (rank_pattern)

使用PEFT 0.7.0+，SALora保留所有发现的rank：

| 特性 | 不使用rank_pattern | 使用rank_pattern |
|------|-------------------|------------------|
| 保留的rank数 | 1个全局值 | n_layers × 7个值 |
| 层级差异 | ❌ 丢失 | ✅ 保留 |
| 模块级差异 | ❌ 丢失 | ✅ 保留 |
| 性能 | 较好的近似 | 最优 |
| 参数分配 | 均匀分布 | 最优分配 |

**示例：** 对于12层模型：
- 不使用rank_pattern: 1个rank值（例如所有都用r=6）
- 使用rank_pattern: 84个独立rank值（12层 × 7模块）

**优势：**
- 早期层获得更高rank（保留预训练知识）
- 晚期层获得更低rank（任务特定适应）
- Key投影经常被剪枝（自然的低重要性）
- Value/Up投影被保留（高重要性）

### 3. 光谱健康约束

通过测量光谱侵入来防止灾难性遗忘：

$$\mathcal{R}_{spec} = \sum_{l,j} \alpha_{l,j} \cdot \| P_l^\perp u_{l,j} \|_2^2$$

### 4. 自动剪枝

L1正则化自动剪枝低贡献模块。

## 评估指标

### Code2NL（代码摘要）

- **BLEU-1/2/3/4：** N-gram重叠
- **METEOR：** 带同义词的语义相似度
- **ROUGE-L：** 最长公共子序列

### Code2Code（断言生成）

- **精确匹配(EM)：** 精确字符串匹配
- **SAM：** 语义断言匹配（标准化比较）
- **BLEU：** N-gram相似度

### NL2Code（代码生成）

- **精确匹配：** 精确代码匹配
- **BLEU：** N-gram相似度
- **Pass@k：** （需要执行，尚未实现）

## 配置

`config.py`中的关键超参数：

```python
@dataclass
class Code2NLConfig:
    # 模型
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"

    # LoRA搜索
    lora_r: int = 8                    # 最大rank
    lora_alpha: int = 16
    lambda_spectral: float = 1e-4      # 光谱惩罚权重
    lambda_l1: float = 1e-3            # L1惩罚权重

    # 训练
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4

    # 生成
    max_source_length: int = 512
    max_target_length: int = 128
    num_beams: int = 5
```

## 高级用法

### 分析rank_pattern结果

使用分析脚本检查发现的配置：

```bash
python analyze_salora_results.py ./output_code2nl/peft_config.json
```

**输出：**
```
SALora配置分析
======================================================================

✅ 使用rank_pattern (PEFT 0.7.0+)

rank_pattern: 84个层-模块组合
  层数: 12
  每层模块数: 7

各模块类型统计:
  query   : 中位数=7, 均值=6.2, 范围=[3, 8]
  key     : 中位数=2, 均值=2.1, 范围=[0, 3]  ← 经常被剪枝
  value   : 中位数=8, 均值=7.3, 范围=[4, 8]  ← 高重要性
  output  : 中位数=6, 均值=5.5, 范围=[2, 6]
  gate    : 中位数=7, 均值=6.1, 范围=[2, 7]
  up      : 中位数=8, 均值=7.0, 范围=[3, 8]  ← 高重要性
  down    : 中位数=6, 均值=5.2, 范围=[1, 6]

层组:
  早期  (层  0- 3): 中位数=7  ← 保留预训练知识
  中期 (层  4- 7): 中位数=6
  晚期  (层  8-11): 中位数=4  ← 任务特定适应
```

### 在代码中直接使用rank_pattern

```python
import json
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 加载rank_pattern
with open('./output_code2nl/peft_config_rank_pattern.json') as f:
    rank_pattern = json.load(f)

# 创建LoraConfig
lora_config = LoraConfig(
    rank_pattern=rank_pattern,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# 应用到模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
model = get_peft_model(base_model, lora_config)

# 正常训练
...
```

### 旧版PEFT的降级方案

如果使用PEFT <0.7.0：

```bash
python verify_with_peft.py \
    --salora_config ./output_code2nl/peft_config.json \
    --no_rank_pattern \
    ...
```

这将使用全局中位数rank作为降级方案。

### 调整光谱惩罚

更高的惩罚 → 更保守（更好地保留预训练知识）
更低的惩罚 → 更激进的适应

```python
config = Code2NLConfig(lambda_spectral=1e-3)  # 更保守
```

### 不同的Qwen模型

```bash
# Qwen2.5-Coder-7B
python run_code_search.py \
    --model_name Qwen/Qwen2.5-Coder-7B \
    --task code2nl \
    --data_dir ./data/jcsd

# Qwen2.5-Coder-0.5B（用于快速测试）
python run_code_search.py \
    --model_name Qwen/Qwen2.5-Coder-0.5B \
    --task code2nl \
    --data_dir ./data/jcsd
```

## 预期结果

SALora搜索后，应该看到类似的rank分配：

```
最终rank分配:
query: [6, 7, 5, 6, 7, 8, 5, 6, 7, 5, 4, 3, ...]
key: [2, 3, 2, 2, 1, 2, 3, 2, 1, 1, 1, 0, ...]
value: [7, 8, 7, 7, 8, 8, 7, 6, 7, 6, 5, 4, ...]
output: [5, 6, 5, 5, 6, 6, 5, 5, 5, 4, 3, 2, ...]
gate: [7, 7, 6, 6, 7, 6, 5, 5, 4, 4, 3, 2, ...]
up: [8, 8, 7, 7, 8, 7, 6, 6, 5, 5, 4, 3, ...]
down: [6, 6, 6, 5, 6, 5, 5, 4, 4, 3, 2, 1, ...]
```

**观察：**
- Query和Value始终获得高rank
- Key经常被剪枝（低rank）
- MLP层（gate/up/down）获得可观的rank
- 后期层倾向于较低rank（任务特定适应）

## 与基线对比

使用rank_pattern的SALora找到稀疏、高效的配置：

| 方法 | 可训练参数 | BLEU | Rank配置 |
|------|-----------|------|---------|
| 全量微调 | 100% | XX.X | N/A |
| LoRA (r=8, 所有模块) | ~2% | XX.X | 固定：所有模块=8 |
| AutoLoRA (仅Q,V) | ~1% | XX.X | 固定：Q,V=8，其他=0 |
| **SALora (降级)** | **~1.5%** | **XX.X** | 单一：中位数=6 |
| **SALora (rank_pattern)** | **~1.5%** | **XX.X+** | **自适应：84个独立rank** |

**rank_pattern vs 降级方案：**

| 方面 | 降级方案（单一rank） | rank_pattern (PEFT 0.7.0+) |
|------|---------------------|---------------------------|
| 使用的rank数 | 1个全局值 | n_layers × 7个值 |
| 性能 | 较好的近似 | **最优**（无信息损失） |
| 参数分配 | 均匀分布 | 最优分配 |
| 层级自适应 | ❌ | ✅ 早期高，晚期低 |
| 模块级剪枝 | ❌ | ✅ 自动（key常为0-3） |
| 使用场景 | 旧版PEFT，测试 | **生产环境，研究** |

## 故障排除

### PEFT版本问题

检查PEFT版本：
```python
import peft
print(peft.__version__)  # 应该 >= 0.7.0 以支持rank_pattern
```

如需升级：
```bash
pip install --upgrade peft
```

如果使用旧版PEFT (<0.7.0)，添加`--no_rank_pattern`：
```bash
python verify_with_peft.py --no_rank_pattern ...
```

### rank_pattern无法加载

如果遇到关于`rank_pattern`参数的错误：

1. **检查PEFT版本**：必须>= 0.7.0
2. **验证配置格式**：
   ```python
   with open('peft_config.json') as f:
       config = json.load(f)
   print(config.get('version'))  # 应该是 '2.0'
   print('rank_pattern' in config)  # 应该是 True
   ```
3. **使用降级方案**：添加`--no_rank_pattern`标志

### 内存不足

减少批大小或模型规模：
```bash
python run_code_search.py --batch_size 4 --model_name Qwen/Qwen2.5-Coder-0.5B ...
```

### NLTK数据缺失

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Betty安装问题

```bash
cd ../AutoLoRA/betty
pip install -e . --no-deps
pip install torch higher
```

## 引用

```bibtex
@article{salora2024,
  title={SALora: Spectral-Aware Meta-Learning for Automated Multi-Module Low-Rank Adaptation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

MIT License
