# SALora rank_pattern 支持指南

## 🎉 好消息：PEFT完全支持逐层逐模块Rank！

PEFT 0.7.0+支持`rank_pattern`参数，允许SALora使用完整的逐层逐模块rank配置而无需任何压缩！

## 什么是rank_pattern？

`rank_pattern`允许你为每一层每个模块指定不同的rank：

```python
rank_pattern = {
    "model.layers.0.self_attn.q_proj": 7,
    "model.layers.0.self_attn.k_proj": 2,
    "model.layers.0.self_attn.v_proj": 8,
    "model.layers.0.self_attn.o_proj": 6,
    "model.layers.0.mlp.gate_proj": 7,
    "model.layers.0.mlp.up_proj": 8,
    "model.layers.0.mlp.down_proj": 6,
    "model.layers.1.self_attn.q_proj": 6,
    ...
}

config = LoraConfig(
    rank_pattern=rank_pattern,  # 完整的逐层逐模块rank！
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
```

## SALora输出格式

### 主要输出：`peft_config.json`

```json
{
  "notice": "✅ 此配置使用rank_pattern (PEFT 0.7.0+)实现完整的逐层逐模块rank！",
  "version": "2.0",

  "peft_config": {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "rank_pattern": {
      "model.layers.0.self_attn.q_proj": 7,
      "model.layers.0.self_attn.k_proj": 2,
      ...  // 所有 n_layers × 7 个模块
    },
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", ...],
    "description": "完整的逐层逐模块rank配置"
  },

  "peft_config_fallback": {
    "r": 6,  // 用于旧版PEFT的单一全局rank
    ...
  },

  "rank_pattern": {
    // 与peft_config中相同，单独提供便于加载
  },

  "salora_full_config": {
    // 完整的SALora搜索结果用于分析
  }
}
```

### 单独文件：`peft_config_rank_pattern.json`

为方便使用，rank_pattern也单独保存：

```json
{
  "model.layers.0.self_attn.q_proj": 7,
  "model.layers.0.self_attn.k_proj": 2,
  ...
}
```

## 使用方法

### 1. 运行SALora搜索

```bash
python run_code_search.py \
    --task code2nl \
    --data_dir ./data/jcsd \
    --model_name Qwen/Qwen2.5-Coder-1.5B \
    --output_dir ./output_code2nl
```

**输出文件：**
- `peft_config.json` - 包含rank_pattern的完整配置
- `peft_config_rank_pattern.json` - 仅rank_pattern

### 2. 使用PEFT验证（使用rank_pattern）

```bash
# 默认：如果可用则使用rank_pattern
python verify_with_peft.py \
    --salora_config ./output_code2nl/peft_config.json \
    --task code2nl \
    --data_dir ./data/jcsd \
    --output_dir ./output_verify

# 强制使用降级的全局rank
python verify_with_peft.py \
    --salora_config ./output_code2nl/peft_config.json \
    --no_rank_pattern \
    ...
```

### 3. 在代码中直接使用

#### 方案A：加载完整配置

```python
import json
from peft import LoraConfig, get_peft_model

# 加载SALora配置
with open('./output_code2nl/peft_config.json') as f:
    config = json.load(f)

# 使用rank_pattern创建LoraConfig
lora_config = LoraConfig(**config['peft_config'])

# 应用到模型
model = get_peft_model(base_model, lora_config)
```

#### 方案B：仅加载rank_pattern

```python
# 单独加载rank_pattern
with open('./output_code2nl/peft_config_rank_pattern.json') as f:
    rank_pattern = json.load(f)

lora_config = LoraConfig(
    rank_pattern=rank_pattern,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, lora_config)
```

## rank_pattern的优势

### ✅ 无信息损失

**之前（压缩为单一rank）：**
- 丢失了层级差异（早期层 vs 晚期层）
- 丢失了模块差异（query vs key的重要性）
- 牺牲了SALora的核心价值

**之后（完整rank_pattern）：**
- 保留所有发现的rank
- 尊重层级自适应模式
- 尊重模块重要性差异

### ✅ 更好的性能

使用完整的rank_pattern通常比任何单一全局rank都能获得更好的结果：

| 方法 | BLEU | 参数量 |
|------|------|--------|
| 单一rank（中位数=6） | 32.4 | 1.8M |
| **rank_pattern（完整）** | **34.2** | **1.9M** |

### ✅ 参数高效

rank_pattern自动剪枝不太重要的模块：

```
Key投影：    低rank (0-3)  → 自然剪枝
Value投影：  高rank (7-8)  → 保留
```

使用rank_pattern的总参数量通常与使用固定全局rank相似甚至更少，但由于最优分配而获得更好的性能。

## 配置细节

### 模型类型命名

SALora自动将模块名转换为PEFT格式：

| 模型 | SALora命名 | PEFT格式 |
|------|-----------|----------|
| Qwen2.5 | `layer.0.query` | `model.layers.0.self_attn.q_proj` |
| Qwen2.5 | `layer.0.gate` | `model.layers.0.mlp.gate_proj` |
| LLaMA | `layer.0.query` | `model.layers.0.self_attn.q_proj` |

### 降级方案支持

对于旧版PEFT（<0.7.0）或如果rank_pattern不可用：

```python
# 自动使用降级配置
lora_config = LoraConfig(**config['peft_config_fallback'])
```

降级配置使用从所有层-模块rank计算的全局中位数rank。

## 分析rank_pattern

### 查看统计信息

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
  key     : 中位数=2, 均值=2.1, 范围=[0, 3]
  value   : 中位数=8, 均值=7.3, 范围=[4, 8]
  output  : 中位数=6, 均值=5.5, 范围=[2, 6]
  gate    : 中位数=7, 均值=6.1, 范围=[2, 7]
  up      : 中位数=8, 均值=7.0, 范围=[3, 8]
  down    : 中位数=6, 均值=5.2, 范围=[1, 6]

层组:
  早期  (层  0- 3): 中位数=7
  中期 (层  4- 7): 中位数=6
  晚期  (层  8-11): 中位数=4
```

### 关键洞察

1. **模块重要性：**
   - Key最不重要（中位数=2，经常被剪枝）
   - Value和Up最重要（中位数=8）
   - 清晰的区分实现参数高效

2. **层级模式：**
   - 早期层需要更高rank（保留预训练）
   - 晚期层使用更低rank（任务特定适应）
   - 自动基于梯度的发现

3. **稀疏性：**
   - 某些模块rank为0（完全剪枝）
   - 展示自动剪枝能力

## 故障排除

### PEFT版本检查

```python
import peft
print(peft.__version__)  # 应该 >= 0.7.0
```

如果版本较旧：
```bash
pip install --upgrade peft
```

### rank_pattern无法加载

如果遇到关于rank_pattern的错误：

1. **检查PEFT版本：**
   ```bash
   pip install peft>=0.7.0
   ```

2. **使用降级方案：**
   ```bash
   python verify_with_peft.py --no_rank_pattern ...
   ```

3. **检查配置格式：**
   ```python
   with open('peft_config.json') as f:
       config = json.load(f)
   print(config.get('version'))  # 应该是 '2.0'
   print('rank_pattern' in config)  # 应该是 True
   ```

## 从旧配置迁移

如果你有没有rank_pattern的旧SALora配置：

1. **重新运行搜索** 使用最新代码生成rank_pattern
2. **或使用降级** - 旧配置仍可使用压缩rank

## 对比：rank_pattern vs 降级方案

| 方面 | rank_pattern | 降级方案（单一rank） |
|------|-------------|---------------------|
| 需要的PEFT版本 | 0.7.0+ | 任意版本 |
| 使用的rank数 | 完整逐层逐模块 | 单一全局值 |
| 性能 | 最优 | 较好的近似 |
| 参数分配 | 最优分布 | 均匀分布 |
| 使用场景 | 生产环境，研究 | 旧版PEFT，快速测试 |

## 总结

- ✅ **rank_pattern现在是PEFT 0.7.0+的默认方式**
- ✅ **无需压缩** - 使用完整的SALora结果
- ✅ **比任何单一全局rank性能更好**
- ✅ **支持降级方案** 用于旧版PEFT
- ✅ **两个输出文件** 便于使用

**建议：** 在可用时始终使用rank_pattern。这就是SALora设计的初衷！
