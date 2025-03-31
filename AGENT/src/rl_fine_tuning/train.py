# src/rl_fine_tuning/train.py

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.reward.reward_function import compute_reward
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class RLFineTuner:
    def __init__(self, model_name, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        
        # Initialize experiment tracking if needed
        if config.get("use_wandb", False):
            wandb.init(project=config["wandb_project"], name=config["experiment_name"])
            wandb.config.update(config)

    def train_step(self, prompt, reference):
        """
        One training step:
        1. Model generates an output
        2. Compute reward
        3. Backpropagate reward signal
        """
        # 1. Generate
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=self.config["max_length"])
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 2. Compute reward
        reward = compute_reward(generated_text, reference)

        # 3. Calculate loss based on reward (placeholder; real RL approach differs)
        # This naive approach won't give correct RL backprop but demonstrates the structure
        loss = -torch.log(torch.tensor(reward + 1e-6))  # Negative log-likelihood of reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), generated_text, reward

    def train_loop(self, dataset):
        """
        Example: dataset is a list of (prompt, reference) pairs.
        """
        for epoch in range(self.config["epochs"]):
            total_loss = 0.0
            total_reward = 0.0
            for prompt, reference in dataset:
                loss, gen_text, reward = self.train_step(prompt, reference)
                total_loss += loss
                total_reward += reward
                
            avg_loss = total_loss / len(dataset)
            avg_reward = total_reward / len(dataset)
            
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")

            # Log metrics to WandB
            if self.config.get("use_wandb", False):
                wandb.log({"loss": avg_loss, "reward": avg_reward, "epoch": epoch+1})
        
        logger.info("Training complete.")
        return

def main():
    # Example configuration
    config = {
        "lr": 1e-5,
        "epochs": 3,
        "max_length": 50,
        "use_wandb": True,
        "wandb_project": "deepseek_rl_project",
        "experiment_name": "run1_rl_finetuning",
    }

    # Replace with your model name or local path
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Example dataset
    dataset = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote 1984?", "George Orwell")
    ]
    
    trainer = RLFineTuner(model_name, config)
    trainer.train_loop(dataset)

if __name__ == "__main__":
    main()
