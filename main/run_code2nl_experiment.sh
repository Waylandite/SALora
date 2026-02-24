#!/bin/bash

# SALora Training Scripts for Code Tasks

# Example 1: Code Summarization (code2nl) with JCSD dataset
echo "=== Running SALora Search for Code Summarization (code2nl) ==="
python run_code_search.py \
    --task code2nl \
    --data_dir ../data/jcsd \
    --model_name Qwen/Qwen2.5-Coder-1.5B \
    --output_dir ./output_code2nl \
    --lora_r 8 \
    --num_epochs 10 \
    --batch_size 8 \
    --seed 42

# Wait for completion
echo "SALora search completed. Results saved to ./output_code2nl"

# Verify with PEFT
echo ""
echo "=== Verifying Results with Standard PEFT ==="
python verify_with_peft.py \
    --salora_config ./output_code2nl/peft_config.json \
    --task code2nl \
    --data_dir ../data/jcsd \
    --model_name Qwen/Qwen2.5-Coder-1.5B \
    --output_dir ./output_peft_verify \
    --num_epochs 10 \
    --batch_size 8 \
    --seed 42

echo ""
echo "=== Verification Complete ==="
echo "Compare results:"
echo "  - SALora search results: ./output_code2nl/results.json"
echo "  - PEFT verification: ./output_peft_verify/verification_results.json"
