# src/rl_fine_tuning/train.py

import os
import sys
import json
import torch
import wandb
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import gc
from typing import List, Tuple, Dict, Any
import numpy as np

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

class SprintGoalDataset(Dataset):
    """Dataset class for sprint goals and reference stories."""
    
    def __init__(self, data_path: str):
        self.data = []
        with open(data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.data.append((obj["sprint_goal"], obj["formatted_issues"]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class GRPOFineTuner:
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model config & tokenizer
        rev = config.get("revision", "main")
        model_config = AutoConfig.from_pretrained(
            model_name, revision=rev, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=rev, trust_remote_code=True
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Main policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=model_config,
            revision=rev, trust_remote_code=True
        ).to(self.device)
        self.model.config.use_cache = False

        if config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.01)
        )

        # Frozen reference policy for KL penalty
        ref_dev = "cpu" if config.get("cpu_offload", False) else self.device
        logger.info(f"Loading reference model on: {ref_dev}")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name, config=model_config,
            revision=rev, trust_remote_code=True
        ).to(ref_dev)
        self.reference_model.eval()
        self.reference_model.config.use_cache = False

        # Initialize training state
        self.iteration = 0
        self.start_epoch = 0
        self.update_every = config.get("update_ref_every", 10)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Training metrics tracking
        self.training_metrics = {
            "losses": [],
            "rewards": [],
            "kl_divs": [],
            "advantages": []
        }

        # Load from checkpoint if specified
        if config.get("resume_from_checkpoint"):
            self.load_checkpoint(config["resume_from_checkpoint"])

        if config.get("use_wandb", False):
            wandb.init(
                project=config["wandb_project"],
                name=config["experiment_name"],
                config=config,
                resume="allow" if config.get("resume_from_checkpoint") else None
            )

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint.get('epoch', 0)
        self.iteration = checkpoint.get('iteration', 0)
        
        # Load training metrics if available
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
        
        # Update reference model to match current policy
        self.reference_model.load_state_dict(self.model.state_dict())
        if self.config.get("cpu_offload", False):
            self.reference_model.to("cpu")
        
        logger.info(f"Resumed from epoch {self.start_epoch}, iteration {self.iteration}")

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint with enhanced metadata."""
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_final:
            checkpoint_path = os.path.join(checkpoint_dir, "model_final.pt")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        
        checkpoint = {
            'epoch': epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics,
            'model_name': self.config.get('model_name', 'unknown')
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Also save a "latest" checkpoint for easy resuming
        latest_path = os.path.join(checkpoint_dir, "model_latest.pt")
        torch.save(checkpoint, latest_path)

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages using z-score normalization with stability checks."""
        with torch.no_grad():
            # Clamp rewards to prevent extreme values
            rewards = torch.clamp(rewards, -100, 100)
            
            mean_r = rewards.mean()
            std_r = rewards.std()
            
            # Add stability check for very small variance
            if std_r < 1e-6:
                # If all rewards are nearly identical, return zero advantages
                advantages = torch.zeros_like(rewards)
            else:
                advantages = (rewards - mean_r) / (std_r + 1e-8)
            
            # Clamp advantages to prevent extreme values
            advantages = torch.clamp(advantages, -5.0, 5.0)
            
        return advantages

    def train_step(self, sprint_goal: str, reference_stories: str) -> Tuple[float, List[str], List[float]]:
        """Single training step with GRPO."""
        try:
            # 1) Build the prompt and tokenize
            prompt = (
                f"{SYSTEM_MESSAGE}\n\n"
                f"Sprint Goal: {sprint_goal}\n\n"
                "Generate appropriate user stories for this sprint goal:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
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
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=self.config.get("temperature", 1.0),
                    top_p=self.config.get("top_p", 0.9),
                )

            # Separate out the generated tokens
            sequences = gen_out.sequences[:, prompt_len:]  # [G, gen_len]

            # 3) COMPUTE GRADIENTS: Forward pass through the model with require_grad=True
            prompt_ids = inputs["input_ids"].repeat(G, 1)  # [G, prompt_len]
            full_ids = torch.cat([prompt_ids, sequences], dim=-1)  # [G, prompt_len+gen_len]
            
            # Forward through policy model with gradient tracking
            policy_out = self.model(input_ids=full_ids, return_dict=True)
            policy_logits = policy_out.logits[:, prompt_len-1:-1, :]  # [G, gen_len, vocab]
            
            # Compute current-policy log-probs with gradient tracking
            logps_curr_steps = logprobs_from_logits(policy_logits, sequences)  # [G, gen_len]
            logps_curr_seq = logps_curr_steps.sum(dim=1)  # [G]

            # 4) Compute reference-policy log-probs (same inputs)
            ref_device = next(self.reference_model.parameters()).device
            full_ids_ref = full_ids.to(ref_device)
            
            with torch.no_grad():
                ref_out = self.reference_model(input_ids=full_ids_ref, return_dict=True)
            
            # Align "next-token" logits for the generated portion
            ref_logits_steps = ref_out.logits[:, prompt_len-1:-1, :]  # [G, gen_len, vocab]
            sequences_ref = sequences.to(ref_device)
            logps_ref_steps = logprobs_from_logits(ref_logits_steps, sequences_ref)
            logps_ref_seq = logps_ref_steps.sum(dim=1).to(self.device)  # [G]

            # 5) Decode and compute rewards
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            rewards = torch.tensor(
                [compute_reward(t, reference_stories) for t in texts],
                dtype=torch.float32, device=self.device
            )

            # 6) Group-relative advantages (z-scores) - DETACH to prevent reward gradients
            advantages = self.compute_advantages(rewards)

            # 7) PPO-style surrogate with clipping - FIX THE RATIO CALCULATION
            # Add numerical stability and proper detaching
            with torch.no_grad():
                ratio = torch.exp(logps_curr_seq - logps_ref_seq)  # Both should be detached for ratio
                ratio = torch.clamp(ratio, 0.1, 10.0)  # Prevent extreme ratios
            
            eps = self.config["epsilon"]
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            
            # The ratio should multiply with current policy logprobs, not be detached
            surrogate_1 = ratio * logps_curr_seq  # Keep gradient flow through current policy
            surrogate_2 = clipped * logps_curr_seq
            surrogate_i = torch.min(surrogate_1, surrogate_2) * advantages.detach()  # advantages detached
            loss_surr = -surrogate_i.mean()

            # 8) KL penalty - FIX: Make sure this doesn't explode
            kl_term = torch.clamp(logps_curr_seq - logps_ref_seq.detach(), -10, 10).mean()
            loss_kl = self.config["beta"] * kl_term

            total_loss = loss_surr + loss_kl
            
            # Add loss validation
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("NaN or Inf loss detected, skipping step")
                return 0.0, [], []

            # 9) Gradient accumulation
            scaled_loss = total_loss / self.gradient_accumulation_steps
            scaled_loss.backward()

            # Store metrics - use the unscaled loss for logging
            self.training_metrics["losses"].append(total_loss.item())
            self.training_metrics["rewards"].extend(rewards.tolist())
            self.training_metrics["kl_divs"].append(kl_term.item())
            self.training_metrics["advantages"].extend(advantages.tolist())

            return total_loss.item(), texts, rewards.tolist()  # Return unscaled loss

        except Exception as e:
            logger.error(f"Error in train_step: {e}")
            raise

    def optimizer_step(self):
        """Perform optimizer step and handle reference model updates."""
        # Check for NaN/Inf gradients before clipping
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    logger.warning("NaN or Inf gradients detected, zeroing gradients")
                    self.optimizer.zero_grad()
                    return
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Log gradient norm for debugging
        if hasattr(self, 'config') and self.config.get("log_grad_norm", False):
            logger.debug(f"Gradient norm: {total_norm}")
        
        # Gradient clipping
        if self.config.get("max_grad_norm", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config["max_grad_norm"]
            )
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.iteration += 1
        
        # Update reference model periodically
        if self.iteration % self.update_every == 0:
            logger.info(f"[iter {self.iteration}] syncing reference model...")
            self.reference_model.load_state_dict(self.model.state_dict())
            # if offloading, move back to CPU
            if self.config.get("cpu_offload", False):
                self.reference_model.to("cpu")

    def evaluate(self, dataset: List[Tuple[str, str]], num_samples: int = 5) -> Dict[str, float]:
        """Evaluate model on a subset of data."""
        self.model.eval()
        eval_rewards = []
        
        # Sample a subset for evaluation
        eval_data = dataset[:num_samples] if len(dataset) > num_samples else dataset
        
        with torch.no_grad():
            for sprint_goal, reference_stories in eval_data:
                prompt = (
                    f"{SYSTEM_MESSAGE}\n\n"
                    f"Sprint Goal: {sprint_goal}\n\n"
                    "Generate appropriate user stories for this sprint goal:"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
                
                gen_out = self.model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=self.config["max_length"],
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                generated_text = self.tokenizer.decode(
                    gen_out[0][inputs["input_ids"].shape[-1]:], 
                    skip_special_tokens=True
                )
                
                reward = compute_reward(generated_text, reference_stories)
                eval_rewards.append(reward)
        
        self.model.train()
        return {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards)
        }

    def train_loop(self, dataset: List[Tuple[str, str]]):
        """Main training loop with infinite training support."""
        # Split dataset for validation
        split_idx = int(len(dataset) * 0.9)
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        logger.info(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
        
        max_epochs = self.config.get("epochs", float('inf'))  # Support infinite training
        current_epoch = self.start_epoch
        
        try:
            while current_epoch < max_epochs:
                logger.info(f"\n=== Epoch {current_epoch+1} ===")
                
                epoch_loss = 0.0
                all_rewards = []
                step_count = 0
                
                for i, (sprint_goal, reference) in enumerate(train_data):
                    try:
                        loss, outputs, rewards = self.train_step(sprint_goal, reference)
                        epoch_loss += loss
                        all_rewards.extend(rewards)
                        step_count += 1
                        
                        # Perform optimizer step after accumulation
                        if (i + 1) % self.gradient_accumulation_steps == 0:
                            self.optimizer_step()
                        
                        # Log progress periodically
                        if (i + 1) % self.config.get("log_every", 10) == 0:
                            avg_loss = epoch_loss / step_count
                            avg_reward = np.mean(all_rewards[-len(rewards):])
                            logger.info(f"Epoch {current_epoch+1}, Step {i+1}/{len(train_data)} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
                            
                            if self.config.get("use_wandb", False):
                                wandb.log({
                                    "step_loss": loss,
                                    "step_reward": avg_reward,
                                    "iteration": self.iteration,
                                    "epoch": current_epoch + 1
                                })
                        
                        # Memory cleanup
                        if (i + 1) % 50 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                    except KeyboardInterrupt:
                        logger.info("Training interrupted by user")
                        self.save_checkpoint(current_epoch, is_final=True)
                        return
                    except Exception as e:
                        logger.error(f"Error at step {i}: {e}")
                        continue
                
                # Epoch summary
                avg_loss = epoch_loss / len(train_data) if step_count > 0 else 0
                avg_reward = np.mean(all_rewards) if all_rewards else 0
                
                # Evaluation
                eval_metrics = self.evaluate(val_data) if val_data else {}
                
                logger.info(f"Epoch {current_epoch+1} Summary:")
                logger.info(f"  Average Loss: {avg_loss:.4f}")
                logger.info(f"  Average Reward: {avg_reward:.4f}")
                if eval_metrics:
                    logger.info(f"  Eval Reward: {eval_metrics['eval_reward_mean']:.4f} ± {eval_metrics['eval_reward_std']:.4f}")
                
                # Log to wandb
                if self.config.get("use_wandb", False):
                    log_dict = {
                        "epoch": current_epoch + 1,
                        "train_loss": avg_loss,
                        "train_reward": avg_reward,
                        "kl_div": np.mean(self.training_metrics["kl_divs"][-len(train_data):]) if self.training_metrics["kl_divs"] else 0,
                    }
                    log_dict.update(eval_metrics)
                    wandb.log(log_dict)
                
                # Save checkpoint
                if self.config.get("save_checkpoints", True):  # Default to True for infinite training
                    save_freq = self.config.get("save_every", 5)
                    if (current_epoch + 1) % save_freq == 0:
                        self.save_checkpoint(current_epoch + 1)
                
                current_epoch += 1
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(current_epoch, is_final=True)
        
        logger.info("Training complete.")

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with checkpoint support")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=0,  # 0 means infinite
                        help="Number of epochs (0 for infinite training)")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--project", default="sprint_goals_rag")
    parser.add_argument("--experiment", default="sprint_goals_rl_tuning")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    
    # Checkpoint arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_checkpoints", action="store_true", default=True,
                        help="Save model checkpoints")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint_dir", default="./checkpoints",
                        help="Directory to save checkpoints")
    
    # Training parameters
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--update_ref_every", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()
    config = vars(args)
    config.update({
        "wandb_project": args.project,
        "experiment_name": args.experiment,
        "revision": "main",
        "model_name": args.model,  # Store model name for checkpoint metadata
    })

    # Handle infinite training
    if args.epochs == 0:
        config["epochs"] = float('inf')
        logger.info("Starting infinite training mode (use Ctrl+C to stop and save)")
    
    # Locate dataset
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
            raise FileNotFoundError("Please specify --dataset path or ensure data file exists")

    logger.info(f"Loading dataset from: {path}")
    
    dataset = SprintGoalDataset(path)
    data = [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))]
    
    logger.info(f"Loaded {len(data)} examples")

    trainer = GRPOFineTuner(args.model, config)
    trainer.train_loop(data)

    # Final save
    if config.get("save_checkpoints", True):
        trainer.save_checkpoint("final", is_final=True)

if __name__ == "__main__":
    main()