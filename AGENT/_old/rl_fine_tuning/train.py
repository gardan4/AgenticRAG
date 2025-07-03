# src/rl_fine_tuning/train.py

import os
import sys
import json
import torch
import wandb
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import gc
from typing import List, Tuple, Dict, Any
import numpy as np


# Project imports (adjust as needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_dir = os.path.dirname(os.path.dirname(current_dir))
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)

from _old.reward.reward_function import compute_reward
from _old.utils.logging_utils import get_logger

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
    """Dataset class for sprint goals and reference stories with improved data handling."""
    
    def __init__(self, data_path: str, split: str = "train", train_split: float = 0.9):
        """
        Initialize dataset with train/val split capability.
        
        Args:
            data_path: Path to the JSONL data file
            split: Either "train" or "val" 
            train_split: Fraction of data to use for training
        """
        self.data = []
        self.split = split
        
        # Load all data first
        all_data = []
        with open(data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                all_data.append((obj["sprint_goal"], obj["formatted_issues"]))
        
        # Split data deterministically
        split_idx = int(len(all_data) * train_split)
        
        if split == "train":
            self.data = all_data[:split_idx]
        elif split == "val":
            self.data = all_data[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        logger.info(f"Loaded {len(self.data)} examples for {split} split")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sprint_goal, formatted_issues = self.data[idx]
        return {
            'sprint_goal': sprint_goal,
            'reference_stories': formatted_issues,
            'idx': idx
        }

def collate_fn(batch):
    """Custom collate function for the DataLoader."""
    return {
        'sprint_goals': [item['sprint_goal'] for item in batch],
        'reference_stories': [item['reference_stories'] for item in batch],
        'indices': [item['idx'] for item in batch]
    }

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
        epoch_value = checkpoint.get('epoch', 0)
        if isinstance(epoch_value, str):
            if epoch_value == "final":
                self.start_epoch = 0  # Start from 0 if loading final checkpoint
            else:
                try:
                    self.start_epoch = int(epoch_value)
                except ValueError:
                    self.start_epoch = 0
        else:
            self.start_epoch = int(epoch_value)
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
        logger.info(f"Checkpoint saved to {checkpoint_path} (iteration: {self.iteration})")
        
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
            scaled_loss.backward()            # Store metrics - use the unscaled loss for logging
            self.training_metrics["losses"].append(total_loss.item())
            self.training_metrics["rewards"].extend(rewards.tolist())
            self.training_metrics["kl_divs"].append(kl_term.item())
            self.training_metrics["advantages"].extend(advantages.tolist())
            
            # Log detailed RL metrics to W&B if enabled
            if self.config.get("use_wandb", False):
                # Basic loss components
                wandb.log({
                    "step_total_loss": total_loss.item(),
                    "step_surrogate_loss": loss_surr.item(),
                    "step_kl_loss": loss_kl.item(),
                    "step_kl_term": kl_term.item(),
                    "iteration": self.iteration,
                }, step=self.iteration)
                
                # Reward statistics
                reward_stats = {
                    "step_reward_mean": rewards.mean().item(),
                    "step_reward_std": rewards.std().item(),
                    "step_reward_min": rewards.min().item(),
                    "step_reward_max": rewards.max().item(),
                }
                wandb.log(reward_stats, step=self.iteration)
                
                # Advantage statistics
                advantage_stats = {
                    "step_advantage_mean": advantages.mean().item(),
                    "step_advantage_std": advantages.std().item(),
                    "step_advantage_min": advantages.min().item(),
                    "step_advantage_max": advantages.max().item(),
                }
                wandb.log(advantage_stats, step=self.iteration)
                
                # Policy statistics
                logp_stats = {
                    "step_logp_current_mean": logps_curr_seq.mean().item(),
                    "step_logp_current_std": logps_curr_seq.std().item(),
                    "step_logp_ref_mean": logps_ref_seq.mean().item(),
                    "step_logp_ref_std": logps_ref_seq.std().item(),
                }
                wandb.log(logp_stats, step=self.iteration)
                
                # Ratio statistics (importance sampling)
                ratio_stats = {
                    "step_ratio_mean": ratio.mean().item(),
                    "step_ratio_std": ratio.std().item(),
                    "step_ratio_min": ratio.min().item(),
                    "step_ratio_max": ratio.max().item(),
                }
                wandb.log(ratio_stats, step=self.iteration)
                
                # PPO clipping statistics
                clipping_stats = {
                    "step_clipped_ratio_mean": clipped.mean().item(),
                    "step_clipping_fraction": (ratio != clipped).float().mean().item(),
                    "step_epsilon": eps,
                    "step_beta": self.config["beta"],
                }
                wandb.log(clipping_stats, step=self.iteration)

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
          # Log gradient norm and optimizer statistics
        if self.config.get("use_wandb", False):
            wandb.log({
                "gradient_norm": total_norm,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "optimizer_step": self.iteration + 1,
            }, step=self.iteration)
        
        if hasattr(self, 'config') and self.config.get("log_grad_norm", False):
            logger.debug(f"Gradient norm: {total_norm}")
        
        # Gradient clipping
        clipped_norm = total_norm
        if self.config.get("max_grad_norm", 0) > 0:
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config["max_grad_norm"]
            )
            
            # Log clipping statistics
            if self.config.get("use_wandb", False):
                wandb.log({
                    "gradient_norm_clipped": clipped_norm,
                    "gradient_clipped": clipped_norm < total_norm,
                    "gradient_clip_ratio": clipped_norm / total_norm if total_norm > 0 else 1.0,
                }, step=self.iteration)
        
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

    def evaluate(self, dataloader: DataLoader, num_samples: int = 5) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        eval_rewards = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= num_samples:
                    break
                
                sprint_goals = batch['sprint_goals']
                reference_stories_list = batch['reference_stories']
                
                for sprint_goal, reference_stories in zip(sprint_goals, reference_stories_list):
                    if samples_processed >= num_samples:
                        break
                        
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
                    samples_processed += 1
        
        self.model.train()
        return {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards)
        }

    def create_dataloaders(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        # Create datasets
        train_dataset = SprintGoalDataset(data_path, split="train", train_split=0.9)
        val_dataset = SprintGoalDataset(data_path, split="val", train_split=0.9)
        
        # Create dataloaders
        batch_size = self.config.get("batch_size", 1)
        num_workers = self.config.get("num_workers", 0)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        return train_loader, val_loader

    def train_loop(self, data_path: str):
        """Main training loop with DataLoader support."""
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(data_path)
        
        logger.info(f"Training on {len(train_loader.dataset)} examples, validating on {len(val_loader.dataset)} examples")
        logger.info(f"Batch size: {train_loader.batch_size}, Steps per epoch: {len(train_loader)}")
        
        max_epochs = self.config.get("epochs", float('inf'))  # Support infinite training
        current_epoch = self.start_epoch
        
        try:
            while current_epoch < max_epochs:
                logger.info(f"\n=== Epoch {current_epoch+1} ===")
                
                epoch_loss = 0.0
                all_rewards = []
                step_count = 0
                
                # Training loop with DataLoader
                for batch_idx, batch in enumerate(train_loader):
                    sprint_goals = batch['sprint_goals']
                    reference_stories_list = batch['reference_stories']
                    
                    # Process each item in the batch
                    batch_loss = 0.0
                    batch_rewards = []
                    
                    for sprint_goal, reference_stories in zip(sprint_goals, reference_stories_list):
                        try:
                            loss, outputs, rewards = self.train_step(sprint_goal, reference_stories)
                            batch_loss += loss
                            batch_rewards.extend(rewards)
                            
                        except Exception as e:
                            logger.error(f"Error processing item in batch {batch_idx}: {e}")
                            continue
                    
                    # Average loss over batch items
                    if len(sprint_goals) > 0:
                        batch_loss /= len(sprint_goals)
                        epoch_loss += batch_loss
                        all_rewards.extend(batch_rewards)
                        step_count += 1
                    
                    # Perform optimizer step after accumulation
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer_step()
                      # Log progress periodically
                    if (batch_idx + 1) % self.config.get("log_every", 10) == 0:
                        avg_loss = epoch_loss / step_count if step_count > 0 else 0
                        avg_reward = np.mean(batch_rewards) if batch_rewards else 0
                        logger.info(f"Epoch {current_epoch+1}, Batch {batch_idx+1}/{len(train_loader)} - Avg Loss: {avg_loss:.4f}, Batch Reward: {avg_reward:.4f}")
                        
                        if self.config.get("use_wandb", False):
                            # Basic batch metrics
                            batch_metrics = {
                                "batch_loss": batch_loss,
                                "batch_reward_mean": avg_reward,
                                "iteration": self.iteration,
                                "epoch": current_epoch + 1,
                                "batch": batch_idx + 1,
                                "batch_size": len(sprint_goals),
                                "progress_pct": ((batch_idx + 1) / len(train_loader)) * 100,
                            }
                            
                            # Add batch reward statistics if available
                            if batch_rewards:
                                batch_metrics.update({
                                    "batch_reward_std": np.std(batch_rewards),
                                    "batch_reward_min": np.min(batch_rewards),
                                    "batch_reward_max": np.max(batch_rewards),
                                })
                            
                            # Add recent training metrics averages
                            recent_window = min(50, len(self.training_metrics["losses"]))
                            if recent_window > 0:
                                batch_metrics.update({
                                    "recent_loss_mean": np.mean(self.training_metrics["losses"][-recent_window:]),
                                    "recent_reward_mean": np.mean(self.training_metrics["rewards"][-recent_window:]) if self.training_metrics["rewards"] else 0,
                                    "recent_kl_mean": np.mean(self.training_metrics["kl_divs"][-recent_window:]) if self.training_metrics["kl_divs"] else 0,
                                    "recent_advantage_mean": np.mean(self.training_metrics["advantages"][-recent_window:]) if self.training_metrics["advantages"] else 0,
                                })
                            
                            wandb.log(batch_metrics)
                    
                    # Memory cleanup
                    if (batch_idx + 1) % 50 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Epoch summary
                avg_loss = epoch_loss / len(train_loader) if step_count > 0 else 0
                avg_reward = np.mean(all_rewards) if all_rewards else 0
                
                # Evaluation
                eval_metrics = self.evaluate(val_loader) if len(val_loader.dataset) > 0 else {}
                
                logger.info(f"Epoch {current_epoch+1} Summary:")
                logger.info(f"  Average Loss: {avg_loss:.4f}")
                logger.info(f"  Average Reward: {avg_reward:.4f}")
                if eval_metrics:
                    logger.info(f"  Eval Reward: {eval_metrics['eval_reward_mean']:.4f} ± {eval_metrics['eval_reward_std']:.4f}")
                  # Log comprehensive epoch statistics to wandb
                if self.config.get("use_wandb", False):
                    # Get epoch-specific metrics (last len(train_loader) steps)
                    epoch_window = min(len(train_loader) * self.config.get("num_samples", 3), len(self.training_metrics["losses"]))
                    
                    epoch_log_dict = {
                        "epoch": current_epoch + 1,
                        "train_loss": avg_loss,
                        "train_reward_mean": avg_reward,
                        "samples_processed": len(all_rewards),
                        "batches_processed": step_count,
                    }
                    
                    # Detailed reward statistics
                    if all_rewards:
                        epoch_log_dict.update({
                            "train_reward_std": np.std(all_rewards),
                            "train_reward_min": np.min(all_rewards),
                            "train_reward_max": np.max(all_rewards),
                            "train_reward_q25": np.percentile(all_rewards, 25),
                            "train_reward_q75": np.percentile(all_rewards, 75),
                        })
                    
                    # KL divergence statistics
                    if self.training_metrics["kl_divs"] and epoch_window > 0:
                        epoch_kl = self.training_metrics["kl_divs"][-epoch_window:]
                        epoch_log_dict.update({
                            "epoch_kl_mean": np.mean(epoch_kl),
                            "epoch_kl_std": np.std(epoch_kl),
                            "epoch_kl_min": np.min(epoch_kl),
                            "epoch_kl_max": np.max(epoch_kl),
                        })
                    
                    # Advantage statistics
                    if self.training_metrics["advantages"] and epoch_window > 0:
                        epoch_advantages = self.training_metrics["advantages"][-epoch_window:]
                        epoch_log_dict.update({
                            "epoch_advantage_mean": np.mean(epoch_advantages),
                            "epoch_advantage_std": np.std(epoch_advantages),
                            "epoch_advantage_min": np.min(epoch_advantages),
                            "epoch_advantage_max": np.max(epoch_advantages),
                        })
                    
                    # Loss statistics
                    if self.training_metrics["losses"] and epoch_window > 0:
                        epoch_losses = self.training_metrics["losses"][-epoch_window:]
                        epoch_log_dict.update({
                            "epoch_loss_std": np.std(epoch_losses),
                            "epoch_loss_min": np.min(epoch_losses),
                            "epoch_loss_max": np.max(epoch_losses),
                        })
                    
                    # Training progress and efficiency metrics
                    epoch_log_dict.update({
                        "total_iterations": self.iteration,
                        "current_lr": self.optimizer.param_groups[0]['lr'],
                        "reference_model_updates": self.iteration // self.update_every,
                    })
                    
                    # Add evaluation metrics
                    epoch_log_dict.update(eval_metrics)
                    
                    wandb.log(epoch_log_dict)
                
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

def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with DataLoader support")
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
    
    # DataLoader arguments
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    
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
    parser.add_argument("--top_p", type=float, default=0.9)    # Logging arguments
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log training progress every N batches")
    parser.add_argument("--log_grad_norm", action="store_true",
                        help="Enable gradient norm logging")
    parser.add_argument("--log_detailed_metrics", action="store_true", default=True,
                        help="Log detailed RL metrics to W&B")

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
            os.path.join("..", "data", "sprint_goals_training_data-qwen-3B.jsonl"),
            os.path.join("data", "sprint_goals_training_data-qwen-3B.jsonl"),
        ]
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
        else:
            raise FileNotFoundError("Please specify --dataset path or ensure data file exists")

    logger.info(f"Loading dataset from: {path}")

    trainer = GRPOFineTuner(args.model, config)
    trainer.train_loop(path)

    # Final save
    if config.get("save_checkpoints", True):
        trainer.save_checkpoint("final", is_final=True)

if __name__ == "__main__":
    main()