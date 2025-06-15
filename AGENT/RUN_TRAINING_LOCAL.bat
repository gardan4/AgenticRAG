@echo off
python train_trl.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --dataset .\data\sprint_goals_training_data-qwen-3B.jsonl ^
  --output_dir .\trl_checkpoints ^
  --epochs 3 ^
  --batch_size 4 ^
  --lr 5e-6 ^
  --beta 0.0 ^
  --epsilon 0.1 ^
  --gradient_accumulation_steps 1