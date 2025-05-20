# src/rl_fine_tuning/train.py

import os
import sys
import json
import torch
import wandb
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F

# Project imports (adjust as needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_dir = os.path.dirname(os.path.dirname(current_dir))
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)

from src.reward.reward_function import compute_reward
from src.utils.logging_utils import get_logger

def logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gather: bool = True
) -> torch.Tensor:
    """
    Compute per‐token log-probs from raw logits.

    Args:
      logits:   Tensor of shape [batch_size, seq_len, vocab_size]
      labels:   Tensor of shape [batch_size, seq_len] with token IDs.
      gather:   If True, returns only log-probs of the given labels;
                otherwise returns full log-prob matrix.

    Returns:
      If gather: Tensor [batch_size, seq_len] of log-probs for each label.
      Else:      Tensor [batch_size, seq_len, vocab_size] of all log-probs.
    """
    # [batch, seq_len, vocab]
    logp = F.log_softmax(logits, dim=-1)

    if not gather:
        return logp

    # pick out the log-prob for each actual token
    # labels.unsqueeze(-1) -> [batch, seq_len, 1]
    # gather along vocab dim (-1)
    logpy = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logpy

SYSTEM_MESSAGE = (
    "You are an experienced Scrum Master. Your task is to generate clear, well-"
    "structured user stories based on sprint goals. Follow the standard user story "
    "format: 'As a [user role], I want [action], so that [benefit]'. Make sure stories "
    "are aligned with the sprint goal, specific, measurable, and achievable."
)

logger = get_logger(__name__)

class GRPOFineTuner:
    def __init__(self, model_name, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model config & tokenizer
        rev = config.get("revision", "main")
        model_config = AutoConfig.from_pretrained(
            model_name, revision=rev, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=rev, trust_remote_code=True
        )

        # Main policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=model_config,
            revision=rev, trust_remote_code=True
        ).to(self.device)
        # disable HF kv-cache so each forward keeps gradients only for that token
        self.model.config.use_cache = False

        if config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])

        # Frozen reference policy for KL penalty
        ref_dev = "cpu" if config.get("cpu_offload", False) else self.device
        print(f"Loading reference model on: {ref_dev}")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name, config=model_config,
            revision=rev, trust_remote_code=True
        ).to(ref_dev)
        self.reference_model.eval()
        self.reference_model.config.use_cache = False

        # For periodic sync
        self.iteration = 0
        self.update_every = config.get("update_ref_every", 10)

        if config.get("use_wandb", False):
            wandb.init(
                project=config["wandb_project"],
                name=config["experiment_name"]
            )
            wandb.config.update(config)

    def train_step(self, sprint_goal, reference_stories):
        # 1) Build the prompt and tokenize
        prompt = (
            f"{SYSTEM_MESSAGE}\n\n"
            f"Sprint Goal: {sprint_goal}\n\n"
            "Generate appropriate user stories for this sprint goal:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[-1]
        G = self.config.get("num_samples", 3)  # group size

        # 2) Sample G outputs in one batched generate call
        with torch.enable_grad():
            gen_out = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=self.config["max_length"],
                num_return_sequences=G,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=False,
            )

        # Separate out the generated tokens & per-step logits
        sequences = gen_out.sequences[:, prompt_len:]              # [G, gen_len]
        scores    = torch.stack(gen_out.scores, dim=1)            # [G, gen_len, vocab_size]

        # 3) Compute current-policy log-probs
        logps_curr_steps = logprobs_from_logits(scores, sequences)  # [G, gen_len]
        logps_curr_seq   = logps_curr_steps.sum(dim=1)              # [G]

        # 4) Compute reference-policy log-probs (same inputs)
        prompt_ids     = inputs["input_ids"].repeat(G, 1)            # [G, prompt_len]
        full_ids       = torch.cat([prompt_ids, sequences], dim=-1) # [G, prompt_len+gen_len]
        with torch.no_grad():
            ref_out = self.reference_model(input_ids=full_ids, return_dict=True)
        # align "next-token" logits for the generated portion:
        # we take positions [prompt_len-1 ... prompt_len+gen_len-2] to predict each generated token
        ref_logits_steps = ref_out.logits[:, prompt_len-1:-1, :]     # [G, gen_len, vocab]
        logps_ref_steps  = logprobs_from_logits(ref_logits_steps, sequences)
        logps_ref_seq    = logps_ref_steps.sum(dim=1)                # [G]

        # 5) Decode and compute rewards
        texts   = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        rewards = torch.tensor(
            [compute_reward(t, reference_stories) for t in texts],
            dtype=torch.float32, device=self.device
        )

        # 6) Group-relative advantages (z-scores)
        mean_r  = rewards.mean()
        std_r   = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r                    # [G]

        # 7) PPO-style surrogate with clipping
        #    ratio = πθ(o_i|q) / π_ref(o_i|q)
        ratio       = torch.exp(logps_curr_seq - logps_ref_seq)    # [G]
        eps         = self.config["epsilon"]
        clipped     = torch.clamp(ratio, 1 - eps, 1 + eps)
        surrogate_i = torch.min(ratio * advantages, clipped * advantages)
        loss_surr   = -surrogate_i.mean()

        # 8) KL penalty D_KL(πθ || π_ref)
        #    D_KL = E[ log πθ - log π_ref ]
        kl_term = (logps_curr_seq - logps_ref_seq).mean()
        loss_kl = self.config["beta"] * kl_term

        loss = loss_surr + loss_kl

        # 9) Backprop & optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 10) Periodically sync reference ← current
        self.iteration += 1
        if self.iteration % self.update_every == 0:
            print(f"[iter {self.iteration}] syncing reference model…")
            self.reference_model.load_state_dict(self.model.state_dict())
            # if offloading, move back to CPU
            if next(self.reference_model.parameters()).device != torch.device(ref_out.logits.device):
                self.reference_model.to(ref_out.logits.device)

        return loss.item(), texts, rewards.tolist()

    def train_loop(self, dataset):
        for epoch in range(self.config["epochs"]):
            print(f"\n=== Epoch {epoch+1}/{self.config['epochs']} ===")
            epoch_loss = 0.0
            all_rewards = []
            for sprint_goal, reference in dataset:
                loss, outputs, rews = self.train_step(sprint_goal, reference)
                epoch_loss += loss
                all_rewards.extend(rews)
            avg_loss = epoch_loss / len(dataset)
            avg_rew  = sum(all_rewards) / len(all_rewards)
            print(f"Epoch {epoch+1} done — avg_loss: {avg_loss:.4f}, avg_rew: {avg_rew:.4f}")
            if self.config.get("use_wandb", False):
                wandb.log({"epoch": epoch+1, "loss": avg_loss, "reward": avg_rew})
        print("Training complete.")

def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with TRL sampler")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project", default="sprint_goals_rag")
    parser.add_argument("--experiment", default="sprint_goals_rl_tuning")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Group size G for GRPO sampling")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL penalty coefficient")
    parser.add_argument("--update_ref_every", type=int, default=10,
                        help="Sync frozen reference every N steps")

    args = parser.parse_args()
    config = vars(args)
    config.update({
        "wandb_project": args.project,
        "experiment_name": args.experiment,
        "revision": "main",
    })

    # Locate dataset JSONL
    if args.dataset:
        path = args.dataset
    else:
        candidates = [
            os.path.join("..", "data", "sprint_goals_training_data.jsonl"),
            os.path.join("data", "sprint_goals_training_data.jsonl"),
        ]
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
        else:
            raise FileNotFoundError("Please specify --dataset path")

    print(f"Loading dataset from: {path}")
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append((obj["sprint_goal"], obj["formatted_issues"]))

    print(f"Loaded {len(data)} examples")

    trainer = GRPOFineTuner(args.model, config)
    trainer.train_loop(data)

if __name__ == "__main__":
    main()
