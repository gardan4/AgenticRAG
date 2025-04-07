# src/rl_fine_tuning/grpo_train.py

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.reward.reward_function import compute_reward
from src.utils.logging_utils import get_logger

from transformers import AutoConfig




logger = get_logger(__name__)

class GRPOFineTuner:
    def __init__(self, model_name, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.quantization_config = None  # remove the problematic config key

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        # Create a reference policy (a frozen snapshot of the current model)
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True
        )
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
        # Work with the input tensor (assume batch size 1 for simplicity)
        current_inputs = inputs
        for _ in range(self.config["max_length"]):
            outputs = self.model(**current_inputs)  # outputs.logits shape: (1, seq_len, vocab_size)
            logits = outputs.logits
            # Get logits for the next token (last position)
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            # Sample a token from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)  # shape (1, 1)
            # Compute log probability for the sampled token
            token_log_prob = torch.log(probs.gather(1, next_token) + 1e-8)
            total_log_prob = total_log_prob + token_log_prob.squeeze()
            token_id = next_token.item()
            generated_tokens.append(token_id)
            if token_id == self.tokenizer.eos_token_id:
                break
            # Append the sampled token to the current input IDs for the next step
            current_input_ids = torch.cat([current_inputs["input_ids"], next_token], dim=-1)
            current_inputs = {"input_ids": current_input_ids}
        return generated_tokens, total_log_prob

    def train_step(self, prompt, reference):
        """
        One training step:
          1. For the given prompt, sample a group of outputs.
          2. Compute rewards and differentiable log probabilities.
          3. Normalize rewards to compute relative advantages.
          4. Compute probability ratios between current and old (reference) log probabilities.
          5. Compute the clipped surrogate objective and add a KL penalty.
          6. Backpropagate and update the model.
        """
        # Tokenize prompt (assume batch size 1)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        group_outputs = []
        rewards = []
        log_probs = []
        old_log_probs = []

        num_samples = self.config.get("num_samples", 5)
        for _ in range(num_samples):
            generated_tokens, sample_log_prob = self.differentiable_sample(inputs)
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            group_outputs.append(generated_text)
            reward = compute_reward(generated_text, reference)
            rewards.append(reward)
            log_probs.append(sample_log_prob)
            # Save the log probability from the generation as the "old" log probability (detached)
            old_log_probs.append(sample_log_prob.detach())

        # Convert lists to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        log_probs_tensor = torch.stack(log_probs)        # current log probabilities
        old_log_probs_tensor = torch.stack(old_log_probs)    # reference log probabilities

        # Compute group baseline: mean and std of rewards
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8  # avoid division by zero
        advantages = (rewards_tensor - mean_reward) / std_reward  # relative advantages

        # Compute probability ratios: ratio = exp(new_log_prob - old_log_prob)
        prob_ratios = torch.exp(log_probs_tensor - old_log_probs_tensor)
        epsilon = self.config.get("epsilon", 0.2)
        clipped_ratios = torch.clamp(prob_ratios, 1 - epsilon, 1 + epsilon)

        # Surrogate objective: use the minimum between unclipped and clipped ratios multiplied by advantages
        surrogate_obj = torch.min(prob_ratios * advantages, clipped_ratios * advantages)

        # Compute a simple KL divergence penalty
        # For a more complete implementation, KL should be computed over the full token distributions.
        kl_penalty = (old_log_probs_tensor - log_probs_tensor).mean()

        beta = self.config.get("beta", 0.01)
        # Total loss: negative surrogate objective plus KL penalty weighted by beta
        loss = -surrogate_obj.mean() + beta * kl_penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update the reference model to the current model
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
        Dataset is a list of (prompt, reference answer) pairs.
        """
        for epoch in range(self.config["epochs"]):
            total_loss = 0.0
            total_rewards = []
            for prompt, reference in dataset:
                loss, outputs, rewards = self.train_step(prompt, reference)
                total_loss += loss
                total_rewards.extend(rewards)
            avg_loss = total_loss / len(dataset)
            avg_reward = sum(total_rewards) / len(total_rewards)
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}, Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
            if self.config.get("use_wandb", False):
                wandb.log({"epoch": epoch+1, "loss": avg_loss, "avg_reward": avg_reward})
        logger.info("Training complete.")

def main():
    # Example configuration
    config = {
        "lr": 1e-5,
        "epochs": 3,
        "max_length": 50,
        "num_samples": 5,
        "epsilon": 0.2,
        "beta": 0.01,
        "update_ref_every": 10,
        "use_wandb": False,
        "wandb_project": "deepseek_rl_project",
        "experiment_name": "run1_grpo_r1",
    }
    # Replace with your model name or local path
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # Example dataset: list of (prompt, reference) pairs
    dataset = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote 1984?", "George Orwell")
    ]
    trainer = GRPOFineTuner(model_name, config)
    trainer.train_loop(dataset)

if __name__ == "__main__":
    main()
