#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Navigate to the AGENT directory (where train.py lives)
cd AgenticRAG/AGENT

# Run the training script with GRPO+TRL sampler
python src/rl_fine_tuning/train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 0 \
  --max_length 1024 \
  --lr 5e-6 \
  --weight_decay 0.01 \
  --dataset data/sprint_goals_training_data-qwen-3B.jsonl \
  --use_wandb \
  --project sprint_goals_rag \
  --experiment sprint_goals_rl_tuning \
  --gradient_checkpointing \
  # Uncomment the next line if you want to offload the reference model to CPU
  # --cpu_offload \
  --batch_size 1 \
  --num_workers 4 \
  --save_checkpoints \
  --save_every 5 \
  --checkpoint_dir ./checkpoints \
  --num_samples 3 \
  --epsilon 0.1 \
  --beta 0.001 \
  --update_ref_every 10 \
  --gradient_accumulation_steps 4 \
  --max_grad_norm 1.0 \
  --temperature 0.8 \
  --top_p 0.9 \
  --log_every 5
  # To resume from a checkpoint, uncomment and edit the next line:
  # --resume_from_checkpoint ./checkpoints/model_epoch_50.pt
