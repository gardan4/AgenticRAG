# src/evaluation/evaluate.py

from src.baseline_agent.base_agent import BaseAgent
from transformers import AutoTokenizer, AutoModelForCausalLM
from _old.reward.reward_function import compute_reward

def evaluate_model(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    total_reward = 0
    for prompt, reference in dataset:
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(**inputs, max_length=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_reward += compute_reward(generated_text, reference)

    return total_reward / len(dataset)

def main():
    dataset = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote 1984?", "George Orwell")
    ]
    
    baseline_agent = BaseAgent("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    baseline_reward = 0
    for prompt, ref in dataset:
        gen_text = baseline_agent.generate(prompt, max_length=50)
        baseline_reward += compute_reward(gen_text, ref)
    baseline_avg_reward = baseline_reward / len(dataset)
    print(f"Baseline Average Reward: {baseline_avg_reward:.2f}")

    # Evaluate the fine-tuned model
    finetuned_model_path = "./path_to_finetuned_model"
    finetuned_avg_reward = evaluate_model(finetuned_model_path, dataset)
    print(f"Fine-tuned Model Average Reward: {finetuned_avg_reward:.2f}")

if __name__ == "__main__":
    main()
