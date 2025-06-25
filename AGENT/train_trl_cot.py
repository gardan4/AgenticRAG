#!/usr/bin/env python
# train_trl_cot.py
import argparse
import logging
import sys
import re
import math
import itertools
import csv

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer
from sentence_transformers import SentenceTransformer

from trl import GRPOConfig, GRPOTrainer

# ─── Logger ─────────────────────────────────────────────────────────────────────
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s | %(name)s | %(levelname)s]: %(message)s"))
        logger.setLevel(level)
        logger.addHandler(h)
    return logger

logger = get_logger(__name__)

# ─── Prompt (no <think> tag inside) ─────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "You are an experienced Scrum Master. Your task is to generate clear, well-"
    "structured user stories based on sprint goals. Follow the standard user story "
    "format: 'As a [user role], I want [action], so that [benefit]'. Make sure stories "
    "are aligned with the sprint goal, specific, measurable, and achievable.\n\n"
    "Sprint Goal: {sprint_goal}\n\n"
    "First write your private reasoning inside <think>...</think>, then output the "
    "user stories in that exact pattern, one per line.\n\n"
)

# ─── Reward-model cache ─────────────────────────────────────────────────────────
_model_cache = {}
def get_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
        logger.info(f"Loaded reward model: {name}")
    return _model_cache[name]

# ─── Helpers ────────────────────────────────────────────────────────────────────
RE_STORY = re.compile(r'^(?:[-–*•]\s*)?As a .+?, I want .+?, so that .+?\.?$', re.I)

def extract_final_output(text: str) -> str:
    """Strip everything up to and incl. </think>."""
    m = re.search(r'</think>\s*', text, flags=re.I)
    return text[m.end():].strip() if m else text.strip()

BULLET_RE = re.compile(r'^(?:[-*•–]|[0-9]+[.)])\s*')

def split_issues(ref_block: str):
    issues = []
    for raw in ref_block.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = BULLET_RE.sub('', line)

        title = None
        try:
            cell = next(csv.reader([line], quotechar='"', escapechar='\\', doublequote=True))
            title = cell[0].strip()
        except Exception:
            m = re.match(r'^(?:"?)(.*?)(?<!\\)"?\s*(?:[,;].*)?$', line)
            if m:
                title = m.group(1).replace('""', '"').replace('\\"', '"').strip()
        if title:
            issues.append(title)
    return issues

def story_lines(clean_out: str):
    """Return list of non-blank lines after </think>."""
    return [ln.strip() for ln in clean_out.splitlines() if ln.strip()]

def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

# ─── Multi-component reward ─────────────────────────────────────────────────────
def compute_multi_reward(gen_text: str, ref_block: str,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                         dup_thresh: float = 0.8, τ: float = 2.0):
    """
    Returns a single scalar reward composed of eight sub-scores:
    regex_struct, clause_pres, coverage, story_count, length_bonus
    minus redundancy_penalty and extraneous_penalty.
    """
    final = extract_final_output(gen_text)
    lines = story_lines(final)
    total_lines = len(lines)

    # Early out if nothing generated
    if total_lines == 0:
        return 0.0

    # 1 Regex-structure
    matches = [ln for ln in lines if RE_STORY.match(ln)]
    regex_struct = len(matches) / total_lines

    # 2 Clause-presence
    clausified = []
    for ln in lines:
        s = 0
        l = ln.lower()
        if "as a" in l:     s += 1/3
        if "i want" in l:   s += 1/3
        if "so that" in l:  s += 1/3
        clausified.append(s)
    clause_presence = sum(clausified) / total_lines

    # 3 Coverage (issue-recall)
    issues = split_issues(ref_block)
    coverage = 0.0
    if issues and matches:
        model = get_model(model_name)
        iss_emb = model.encode(issues, convert_to_tensor=True)
        sto_emb = model.encode(matches, convert_to_tensor=True)
        sims = []
        for ie in iss_emb:
            best = torch.max(F.cosine_similarity(ie.unsqueeze(0), sto_emb)).item()
            sims.append(best)
        coverage = sum(sims) / len(sims)

    # 4 Redundancy penalty
    redundancy = 0.0
    if len(matches) > 1:
        model = get_model(model_name)
        m_emb = model.encode(matches, convert_to_tensor=True)
        dup = 0
        for i, j in itertools.combinations(range(len(matches)), 2):
            if cosine(m_emb[i], m_emb[j]) > dup_thresh:
                dup += 1
        redundancy = dup / len(matches)

    # 5 Story-count match
    story_cnt_reward = math.exp(-abs(len(matches) - len(issues)) / τ) if issues else 0.0

    # 6 Extraneous-text penalty
    extraneous_lines = total_lines - len(matches)
    extraneous_penalty = extraneous_lines / total_lines

    # 7 Length-adequacy bonus
    bonuses = []
    for ln in matches:
        n = len(ln.split())
        if   n <= 12: bonuses.append(0.0)
        elif n >= 45: bonuses.append(1.0)
        else:         bonuses.append((n - 12) / 33)
    length_bonus = sum(bonuses) / len(bonuses) if bonuses else 0.0

    # Aggregate: average positives then subtract penalties
    pos = (regex_struct + clause_presence + coverage +
           story_cnt_reward + length_bonus) / 5.0
    neg = (redundancy + extraneous_penalty)
    return pos - neg

def grpo_reward_fn(completions, **kwargs):
    refs = kwargs.get("reference_stories", [""] * len(completions))
    rewards = []
    for comp, ref in zip(completions, refs):
        try:
            r = compute_multi_reward(comp, ref)
            rewards.append(r)
        except Exception as e:
            logger.error(f"Reward error: {e}")
            rewards.append(0.0)
    return rewards

# ─── Main training script ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning with DeepSeek-R1-Distill-Qwen-1.5B"
    )
    # —— core / dataset / optimisation flags (identical to your original) ———
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./deepseek_r1_grpo_checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-6,
                        dest="learning_rate")
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--reward_weights", type=float, nargs="+", default=None)
    parser.add_argument("--scale_rewards", action="store_true")
    parser.add_argument("--loss_type", type=str, default="bnpo",
                        choices=["grpo", "bnpo", "dr_grpo"])
    parser.add_argument("--mask_truncated_completions", action="store_true")
    parser.add_argument("--sync_ref_model", action="store_true")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.6)
    parser.add_argument("--ref_model_sync_steps", type=int, default=512)
    parser.add_argument("--use_liger_loss", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "epoch", "steps"])
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--num_completions_to_print", type=int, default=None)
    args = parser.parse_args()

    # ─── seed & tokenizer / prefix enforcement ─────────────────────────────────
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    THINK_PREFIX_IDS = tokenizer("<think>\n", add_special_tokens=False).input_ids

    def prefix_allowed_tokens(batch_id, input_ids):
        cur = input_ids.shape[-1]
        return [THINK_PREFIX_IDS[cur]] if cur < len(THINK_PREFIX_IDS) \
               else list(range(tokenizer.vocab_size))

    # ─── Dataset ───────────────────────────────────────────────────────────────
    full_ds = load_dataset("json", data_files={"data": args.dataset}, split="data")
    parts = full_ds.train_test_split(test_size=0.2, seed=args.seed)
    train_raw, val_raw = parts["train"], parts["test"]

    def preprocess(ex):
        return dict(
            prompt=PROMPT_TEMPLATE.format(sprint_goal=ex["sprint_goal"]),
            reference_stories=ex.get("formatted_issues", "")
        )

    train_ds = train_raw.map(preprocess, remove_columns=train_raw.column_names).shuffle(seed=args.seed)
    val_ds   = val_raw  .map(preprocess, remove_columns=None)                 .shuffle(seed=args.seed)

    logger.info(f"Training samples: {len(train_ds)} | Validation: {len(val_ds)}")

    # ─── GRPO config ───────────────────────────────────────────────────────────
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
        delta=args.delta,
        epsilon_high=args.epsilon_high,
        reward_weights=args.reward_weights,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        mask_truncated_completions=args.mask_truncated_completions,
        sync_ref_model=args.sync_ref_model,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        ref_model_sync_steps=args.ref_model_sync_steps,
        use_liger_loss=args.use_liger_loss,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        generate_kwargs=dict(
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_completion_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            pad_token_id=tokenizer.eos_token_id,
        ),
    )

    # ─── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=grpo_reward_fn,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info(f"Finished!  Checkpoints in {args.output_dir}")

if __name__ == "__main__":
    main()
