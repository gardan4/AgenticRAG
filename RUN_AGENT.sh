#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Navigate to the AGENT directory
cd AgenticRAG/AGENT

# Run the training script
python src/rl_fine_tuning/train.py \
  --model "Qwen/Qwen2.5-0.5B-Instruct" \
  --gradient_checkpointing \
  --cpu_offload \
  --sequential_batching \
  --aggressive_cache_clearing \
  --num_samples 3 \
  --max_length 100 \
  --inference_no_grad