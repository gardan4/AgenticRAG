@echo off
python train_trl.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --dataset .\data\sprint_goals_training_data-qwen-3B.jsonl ^
  --output_dir .\trl_checkpoints ^
  --lr 5e-6 ^
  --beta 0.04 ^
  --epsilon 0.2 ^
  --gradient_accumulation_steps 1 ^
  --save_strategy steps ^
  --batch_size 4