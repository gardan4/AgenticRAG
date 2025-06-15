import os
import argparse
import logging
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from sentence_transformers import SentenceTransformer

from trl import GRPOConfig, GRPOTrainer

# ─── Logger Setup ───────────────────────────────────────────────────────────────
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s | %(name)s | %(levelname)s]: %(message)s")
        )
        logger.setLevel(level)
        logger.addHandler(handler)
    return logger

logger = get_logger(__name__)

# ─── Prompt Template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "You are an experienced Scrum Master. Your task is to generate clear, well-"
    "structured user stories based on sprint goals. Follow the standard user story "
    "format: 'As a [user role], I want [action], so that [benefit]'. Make sure stories "
    "are aligned with the sprint goal, specific, measurable, and achievable.\n\n"
    "Sprint Goal: {sprint_goal}\n\nGenerate appropriate user stories for this sprint goal:"
)

# ─── Reward Function Utilities ─────────────────────────────────────────────────
_model_cache = {}
def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info(f"Loaded reward model: {model_name}")
    return _model_cache[model_name]

def compute_similarity(text1: str, text2: str, model_name: str = None) -> float:
    model = get_model(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    return float(torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    ).item())

def compute_reward(generated_text: str, reference_text: str) -> float:
    if not (generated_text and reference_text):
        return 0.0
    return compute_similarity(generated_text, reference_text)

def grpo_reward_fn(completions, **kwargs):
    references = kwargs.get("reference_stories", [""] * len(completions))
    rewards = []
    for comp, ref in zip(completions, references):
        try:
            rewards.append(compute_reward(comp, ref))
        except Exception as e:
            logger.error(f"Reward error: {e}")
            rewards.append(0.0)
    return rewards

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with TRL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, required=True,
                        help="JSONL with fields 'sprint_goal' and 'formatted_issues'")
    parser.add_argument("--output_dir", type=str, default="./trl_checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Load and preprocess dataset
    raw_ds = load_dataset("json", data_files={"train": args.dataset}, split="train")
    def preprocess(ex):
        return {
            "prompt": PROMPT_TEMPLATE.format(sprint_goal=ex["sprint_goal"]),
            "reference_stories": ex.get("formatted_issues", "")
        }
    ds = raw_ds.map(preprocess, remove_columns=raw_ds.column_names)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GRPO configuration :contentReference[oaicite:0]{index=0}
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_iterations=1,
        beta=args.beta,
        epsilon=args.epsilon,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=100,
        logging_steps=10,
        save_strategy="epoch",
        use_vllm=False,              # or True if you’ve set up vLLM
        bf16=False,                  # or True if you want bfloat16
    )

    # Initialize trainer :contentReference[oaicite:1]{index=1}
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=grpo_reward_fn,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,    # ← use this instead
    )

    logger.info("Starting GRPO training…")
    trainer.train()

if __name__ == "__main__":
    main()
