from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM
import os
import torch.nn as nn

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "/home/zhangxiaohong/public_user/mllm/test/model/Qwen2.5-Coder-3B-Instruct"
)

# 2. 打印模型层结构与层数
print("======== 模型层级结构 ========")
for name, module in model.named_modules():
    print(name, ":", type(module).__name__)

# 统计 transformer 层数（根据模型结构，一般是 model.layers 或 model.model.layers）
try:
    num_layers = len(model.model.layers)
except AttributeError:
    try:
        num_layers = len(model.transformer.layers)
    except AttributeError:
        num_layers = "未能自动识别，请手动检查打印结果"
print(f"\n======== 模型层数: {num_layers} ========\n")

# 3. 统计所有线性层（LoRA 常注入位置）
linear_modules = [
    name for name, module in model.named_modules()
    if isinstance(module, nn.Linear)
]
print("======== 模型中所有 Linear 层名称（可作为 LoRA target_modules） ========")
for name in linear_modules:
    print(name)
print(f"\n共发现 {len(linear_modules)} 个 nn.Linear 模块。")

# 4. 创建一个测试用 LoRA 配
