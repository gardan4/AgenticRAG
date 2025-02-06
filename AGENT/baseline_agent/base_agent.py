# src/baseline_agent/base_agent.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseAgent:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        # Use the huggingface model name or local checkpoint path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_length: int = 50):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
