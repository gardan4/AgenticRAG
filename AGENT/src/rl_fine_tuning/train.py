# src/rl_fine_tuning/train.py

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
        
        # Enable gradient checkpointing if requested
        if config.get("gradient_checkpointing", False):
            print("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable()

        # Initialize the optimizer.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        
        # Load the reference model - possibly on CPU to save VRAM
        ref_device = "cpu" if config.get("cpu_offload", False) else self.device
        print(f"Loading reference model on: {ref_device}")
        
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            revision=revision,
            trust_remote_code=True
        ).to(ref_device)
        self.reference_model.eval()
        
        self.iteration = 0
        
        if config.get("use_wandb", False):
            wandb.init(project=config["wandb_project"], name=config["experiment_name"])
            wandb.config.update(config)

    def differentiable_sample(self, inputs):
        """
        Generate a sequence token-by-token in a memory-efficient way with gradient control.
        """
        generated_tokens = []
        total_log_prob = 0.0
        all_logits = []
        all_next_tokens = []
        
        # Ensure the inputs are on the correct device
        current_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        use_inference_no_grad = self.config.get("inference_no_grad", False)
        
        # First generate the sequence without gradients to save memory
        if use_inference_no_grad:
            with torch.no_grad():
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.amp.autocast(device_type=device_type, enabled=True):
                    for _ in range(self.config.get("max_length", 100)):
                        # Forward pass
                        outputs = self.model(**current_inputs)
                        next_token_logits = outputs.logits[:, -1, :]
                        
                        # Store logits for later gradient computation
                        all_logits.append(next_token_logits.detach().clone())
                        
                        # Sample token
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        all_next_tokens.append(next_token.detach().clone())
                        
                        token_id = next_token.item()
                        generated_tokens.append(token_id)
                        
                        if token_id == self.tokenizer.eos_token_id:
                            break
                        
                        # Update input IDs for next step
                        current_input_ids = torch.cat([current_inputs["input_ids"], next_token], dim=-1)
                        current_inputs = {"input_ids": current_input_ids.to(self.device)}
                        
                        # Free memory
                        del outputs, next_token_logits
            
            # Now recompute log probabilities with gradients enabled
            with torch.enable_grad():
                for logits, token in zip(all_logits, all_next_tokens):
                    # Ensure we have tensors that require gradient
                    logits_with_grad = logits.detach().requires_grad_()
                    probs = torch.softmax(logits_with_grad, dim=-1)
                    token_log_prob = torch.log(probs.gather(1, token) + 1e-8)
                    total_log_prob = total_log_prob + token_log_prob.squeeze()
        
        else:
            # Original implementation with gradients enabled
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=True):
                for _ in range(self.config.get("max_length", 100)):
                    outputs = self.model(**current_inputs)
                    next_token_logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    token_id = next_token.item()
                    generated_tokens.append(token_id)
                    
                    token_log_prob = torch.log(probs.gather(1, next_token) + 1e-8)
                    total_log_prob = total_log_prob + token_log_prob.squeeze()
                    
                    if token_id == self.tokenizer.eos_token_id:
                        break
                    
                    current_input_ids = torch.cat([current_inputs["input_ids"], next_token], dim=-1)
                    current_inputs = {"input_ids": current_input_ids.to(self.device)}
                    
                    del outputs, next_token_logits
        
        # Clear cache
        if torch.cuda.is_available() and self.config.get("aggressive_cache_clearing", False):
            torch.cuda.empty_cache()
        
        return generated_tokens, total_log_prob

    def compute_sequence_log_prob(self, inputs, token_sequence, device):
        """
        Compute log probability of a given token sequence under the reference model.
        """
        total_log_prob = 0.0
        current_inputs = inputs
        
        with torch.no_grad():
            for i in range(len(token_sequence)):
                # Get the current token
                token = token_sequence[i]
                
                # Forward pass with reference model
                outputs = self.reference_model(**current_inputs)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Get probability of the actual token
                probs = torch.softmax(next_token_logits, dim=-1)
                token_tensor = torch.tensor([[token]], device=device)
                token_log_prob = torch.log(probs.gather(1, token_tensor) + 1e-8)
                total_log_prob = total_log_prob + token_log_prob.squeeze()
                
                # Update inputs for next token (if not the last token)
                if i < len(token_sequence) - 1:
                    next_token = torch.tensor([[token]], device=device)
                    current_input_ids = torch.cat([current_inputs["input_ids"], next_token], dim=-1)
                    current_inputs = {"input_ids": current_input_ids}
                
                # Free memory
                del outputs, next_token_logits
                
        return total_log_prob

    def train_step(self, prompt, reference):
        """
        One training step with memory optimizations.
        """
        # Use the chat template: prepend the system message and format as a sprint goal task
        chat_prompt = f"{SYSTEM_MESSAGE}\n\nSprint Goal: {prompt}\n\nGenerate appropriate user stories for this sprint goal:"
        print(f"\n=== Chat Prompt: {chat_prompt} ===")

        inputs = self.tokenizer(chat_prompt, return_tensors='pt')
        
        group_outputs = []
        rewards = []
        log_probs = []
        old_log_probs = []

        # Use potentially reduced number of samples
        num_samples = self.config.get("num_samples", 3)  # Default reduced from 5 to 3
        
        # Handle sequential vs. parallel processing
        if self.config.get("sequential_batching", False):
            print("Using sequential batching to save memory")
            for i in range(num_samples):
                # Clear cache before each sample
                if torch.cuda.is_available() and self.config.get("aggressive_cache_clearing", False):
                    torch.cuda.empty_cache()
                    
                # Process one sample at a time
                sample_inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate with current model and get log probability
                generated_tokens, sample_log_prob = self.differentiable_sample(sample_inputs)
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Store outputs and reward
                group_outputs.append(generated_text)
                reward = compute_reward(generated_text, reference)
                rewards.append(reward)
                
                # Current model log probabilities
                log_probs.append(sample_log_prob)
                
                # Get log probabilities from reference model (this is the key change!)
                ref_device = "cpu" if self.config.get("cpu_offload", False) else self.device
                with torch.no_grad():
                    # We need to compute the same sequence's probability under the reference model
                    ref_inputs = {k: v.to(ref_device) for k, v in inputs.items()}
                    ref_log_prob = self.compute_sequence_log_prob(ref_inputs, generated_tokens, ref_device)
                    old_log_probs.append(ref_log_prob)
                    
                print(f"\n-- Sample {i+1} --")
                print(f"Output: {generated_text}")
                print(f"Reward: {reward:.4f}")
                
                # Clear sample from memory
                del generated_tokens, sample_log_prob
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # else:
        #     # Fixed parallel batch processing
        #     for i in range(num_samples):
        #         # Clear cache before processing to minimize memory usage
        #         if torch.cuda.is_available() and self.config.get("aggressive_cache_clearing", False):
        #             torch.cuda.empty_cache()
                    
        #         # Generate with current model
        #         inputs_gpu = {k: v.to(self.device) for k, v in inputs.items()}
        #         generated_tokens, sample_log_prob = self.differentiable_sample(inputs_gpu)
        #         generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
        #         # Store outputs and reward
        #         group_outputs.append(generated_text)
        #         reward = compute_reward(generated_text, reference)
        #         rewards.append(reward)
        #         log_probs.append(sample_log_prob)
                
        #         # Compute log probabilities using reference model (FIX HERE)
        #         ref_device = "cpu" if self.config.get("cpu_offload", False) else self.device
        #         with torch.no_grad():
        #             ref_inputs = {k: v.to(ref_device) for k, v in inputs.items()}
        #             ref_log_prob = self.compute_sequence_log_prob(ref_inputs, generated_tokens, ref_device)
        #             old_log_probs.append(ref_log_prob)
                    
        #         print(f"\n-- Sample {i+1} --")
        #         print(f"Output: {generated_text}")
        #         print(f"Reward: {reward:.4f}")
                
        #         # Free memory
        #         del generated_tokens, sample_log_prob
        
        # Move to appropriate devices and convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        if self.config.get("cpu_offload", False):
            # Process on CPU first, then move only needed tensors to GPU
            log_probs_tensor = torch.stack([lp.to(self.device) for lp in log_probs])
            old_log_probs_tensor = torch.stack([olp.to(self.device) for olp in old_log_probs])
        else:
            log_probs_tensor = torch.stack(log_probs).to(self.device)
            old_log_probs_tensor = torch.stack(old_log_probs).to(self.device)

        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward

        prob_ratios = torch.exp(log_probs_tensor - old_log_probs_tensor)
        # Add a debug print to verify ratios
        print(f"Debug - Raw prob ratios: {prob_ratios.tolist()}")
        
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
        
        # Clear cache after gradient update
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss.item(), group_outputs, rewards

    def train_loop(self, dataset):
        """
        Training loop over the dataset with memory optimizations.
        """
        for epoch in range(self.config["epochs"]):
            print(f"\n--- Epoch {epoch + 1}/{self.config['epochs']} ---")
            total_loss = 0.0
            total_rewards = []
            
            # Process dataset with potential batch size limits
            batch_size = self.config.get("batch_size", 1)  # Default to 1 for memory efficiency
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                for prompt, reference in batch:
                    loss, outputs, rewards = self.train_step(prompt, reference)
                    total_loss += loss
                    total_rewards.extend(rewards)
                    
                    # Clear cache between examples
                    if torch.cuda.is_available() and self.config.get("aggressive_cache_clearing", False):
                        torch.cuda.empty_cache()
                        
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
    parser.add_argument('--max_length', type=int, default=100,  # Reduced from 150 to 100
                        help='Maximum token length for generation (default: 100)')
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
                        
    # Add memory optimization arguments
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--cpu_offload', action='store_true',
                        help='Keep reference model on CPU to save GPU memory')
    parser.add_argument('--sequential_batching', action='store_true',
                        help='Process samples sequentially to save memory')
    parser.add_argument('--aggressive_cache_clearing', action='store_true',
                        help='Aggressively clear CUDA cache to reduce memory fragmentation')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples per prompt (default: 3)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of prompts to process at once (default: 1)')
    parser.add_argument('--inference_no_grad', action='store_true',
                    help='Disable gradients during inference to save memory')
                    
                        
    args = parser.parse_args()

    # Set up configuration
    config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "num_samples": args.num_samples,
        "epsilon": 0.2,
        "beta": 0.01,
        "update_ref_every": 10,
        "use_wandb": args.use_wandb,
        "wandb_project": args.project,
        "experiment_name": args.experiment,
        "revision": "main",
        # Add memory optimization flags
        "gradient_checkpointing": args.gradient_checkpointing,
        "cpu_offload": args.cpu_offload,
        "sequential_batching": args.sequential_batching,
        "aggressive_cache_clearing": args.aggressive_cache_clearing,
        "batch_size": args.batch_size,
        "inference_no_grad": args.inference_no_grad
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