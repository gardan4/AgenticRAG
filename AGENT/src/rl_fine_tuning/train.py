# src/rl_fine_tuning/grpo_train.py

import torch
import wandb
import argparse
import os
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
agent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to AGENT
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)

# Now imports will work when running from AGENT/
from src.reward.reward_function import compute_reward
from src.utils.logging_utils import get_logger

# Replace the existing SYSTEM_MESSAGE with this new one
SYSTEM_MESSAGE = "You are an experienced Scrum Master. Your task is to generate clear, well-structured user stories based on sprint goals. Follow the standard user story format: 'As a [user role], I want [action], so that [benefit]'. Make sure stories are aligned with the sprint goal, specific, measurable, and achievable."

logger = get_logger(__name__)

class GRPOFineTuner:
    def __init__(self, model_name, config):
        self.config = config
        
        # Determine device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Pin the revision to avoid auto-updating to a new version.
        revision = config.get("revision", "main")  # set a commit hash or tag

        # Load configuration with custom code enabled.
        model_config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True
        )

        # Load the tokenizer with pinned revision.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True
        )
        
        # Load the model and move it to the selected device.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            revision=revision,
            trust_remote_code=True
        ).to(self.device)

        # Initialize the optimizer.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        
        # Load the reference model as a frozen snapshot, and move it to device.
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            revision=revision,
            trust_remote_code=True
        ).to(self.device)
        self.reference_model.eval()
        
        self.iteration = 0
        
        if config.get("use_wandb", False):
            wandb.init(project=config["wandb_project"], name=config["experiment_name"])
            wandb.config.update(config)

    def differentiable_sample(self, inputs):
        """
        Generate a sequence token-by-token in a differentiable manner.
        Returns the generated token IDs and the total log probability of the sequence.
        """
        generated_tokens = []
        total_log_prob = 0.0
        # Ensure the inputs are on the correct device.
        current_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        for _ in range(self.config["max_length"]):
            outputs = self.model(**current_inputs)  # outputs.logits shape: (1, seq_len, vocab_size)
            logits = outputs.logits
            # Get logits for the next token (last position)
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            # Sample a token from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (1, 1)
            # Compute log probability for the sampled token
            token_log_prob = torch.log(probs.gather(1, next_token) + 1e-8)
            total_log_prob = total_log_prob + token_log_prob.squeeze()
            token_id = next_token.item()
            generated_tokens.append(token_id)
            if token_id == self.tokenizer.eos_token_id:
                break
            # Append the sampled token to the current input IDs for the next step.
            current_input_ids = torch.cat([current_inputs["input_ids"], next_token], dim=-1)
            current_inputs = {"input_ids": current_input_ids.to(self.device)}
        return generated_tokens, total_log_prob

    def train_step(self, prompt, reference):
        """
        One training step:
          1. For the given prompt, sample a group of outputs.
          2. Compute rewards and differentiable log probabilities.
          3. Normalize rewards to compute relative advantages.
          4. Compute probability ratios between current and reference (old) log probabilities.
          5. Compute the clipped surrogate objective and add a KL penalty.
          6. Backpropagate and update the model.
        """
        # Use the chat template: prepend the system message and format as a sprint goal task
        chat_prompt = f"{SYSTEM_MESSAGE}\n\nSprint Goal: {prompt}\n\nGenerate appropriate user stories for this sprint goal:"
        print(f"\n=== Chat Prompt: {chat_prompt} ===")
    
        inputs = self.tokenizer(chat_prompt, return_tensors='pt').to(self.device)
        
        group_outputs = []
        rewards = []
        log_probs = []
        old_log_probs = []

        num_samples = self.config.get("num_samples", 5)
        for i in range(num_samples):
            generated_tokens, sample_log_prob = self.differentiable_sample(inputs)
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            group_outputs.append(generated_text)
            reward = compute_reward(generated_text, reference)
            rewards.append(reward)
            log_probs.append(sample_log_prob)
            old_log_probs.append(sample_log_prob.detach())
            print(f"\n-- Sample {i+1} --")
            print(f"Output: {generated_text}")
            print(f"Reward: {reward:.4f}")
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        log_probs_tensor = torch.stack(log_probs).to(self.device)
        old_log_probs_tensor = torch.stack(old_log_probs).to(self.device)

        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward

        prob_ratios = torch.exp(log_probs_tensor - old_log_probs_tensor)
        epsilon = self.config.get("epsilon", 0.2)
        clipped_ratios = torch.clamp(prob_ratios, 1 - epsilon, 1 + epsilon)

        surrogate_obj = torch.min(prob_ratios * advantages, clipped_ratios * advantages)
        kl_penalty = (old_log_probs_tensor - log_probs_tensor).mean()
        beta = self.config.get("beta", 0.01)
        loss = -surrogate_obj.mean() + beta * kl_penalty

        print("\n== Training Step Summary ==")
        print(f"Mean Reward: {mean_reward.item():.4f}")
        print(f"Advantages: {advantages.tolist()}")
        print(f"Probability Ratios: {prob_ratios.tolist()}")
        print(f"Loss: {loss.item()}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iteration += 1
        update_ref_every = self.config.get("update_ref_every", 10)
        if self.iteration % update_ref_every == 0:
            self.reference_model.load_state_dict(self.model.state_dict())

        if self.config.get("use_wandb", False):
            wandb.log({"loss": loss.item(), "avg_reward": rewards_tensor.mean().item()})

        return loss.item(), group_outputs, rewards

    def train_loop(self, dataset):
        """
        Training loop over the dataset.
        Dataset is a list of (prompt, reference) pairs.
        """
        for epoch in range(self.config["epochs"]):
            print(f"\n--- Epoch {epoch + 1}/{self.config['epochs']} ---")
            total_loss = 0.0
            total_rewards = []
            for prompt, reference in dataset:
                loss, outputs, rewards = self.train_step(prompt, reference)
                total_loss += loss
                total_rewards.extend(rewards)
            avg_loss = total_loss / len(dataset)
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"\nEpoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}")
            if self.config.get("use_wandb", False):
                wandb.log({"epoch": epoch + 1, "loss": avg_loss, "avg_reward": avg_reward})
        print("\nTraining complete.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model using GRPO fine-tuning')
    parser.add_argument('--model', default="Qwen/Qwen2.5-0.5B-Instruct", 
                        help='Model name to use (default: Qwen/Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='Number of epochs to train (default: 1)')
    parser.add_argument('--max_length', type=int, default=150, 
                        help='Maximum token length for generation (default: 150)')
    parser.add_argument('--lr', type=float, default=1e-5, 
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--dataset', default=None, 
                        help='Path to dataset JSONL file (default: ../data/sprint_goals_training_data.jsonl)')
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Enable Weights & Biases logging')
    parser.add_argument('--project', default='sprint_goals_rag',
                        help='Weights & Biases project name (default: sprint_goals_rag)')
    parser.add_argument('--experiment', default='sprint_goals_rl_tuning',
                        help='Experiment name for Weights & Biases (default: sprint_goals_rl_tuning)')
    args = parser.parse_args()

    # Set up configuration
    config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "num_samples": 5,
        "epsilon": 0.2,
        "beta": 0.01,
        "update_ref_every": 10,
        "use_wandb": args.use_wandb,
        "wandb_project": args.project,
        "experiment_name": args.experiment,
        "revision": "main",
    }
    # Use the model specified in arguments
    model_name = args.model

    # Determine the path to the JSONL file based on current directory
    if args.dataset:
        jsonl_path = args.dataset
    else:
        # Try to locate the dataset file
        possible_paths = [
            os.path.join('..', 'data', 'sprint_goals_training_data.jsonl'),  # When in src/rl_fine_tuning
            os.path.join('data', 'sprint_goals_training_data.jsonl'),        # When in AGENT root
            os.path.join('.', 'data', 'sprint_goals_training_data.jsonl')    # Alternative for AGENT root
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                jsonl_path = path
                break
        else:
            raise FileNotFoundError(
                "Could not find sprint_goals_training_data.jsonl. Please specify the path with --dataset")

    # Load the dataset
    print(f"Loading dataset from: {jsonl_path}")
    sprint_dataset = []
    with open(jsonl_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                # Format as (prompt, reference) pairs where prompt is sprint_goal and reference is formatted_issues
                sprint_dataset.append((data['sprint_goal'], data['formatted_issues']))
    
    print(f"Loaded {len(sprint_dataset)} sprint goal training examples")

    # Initialize and train the model
    trainer = GRPOFineTuner(model_name, config)
    trainer.train_loop(sprint_dataset)

if __name__ == "__main__":
    main()
