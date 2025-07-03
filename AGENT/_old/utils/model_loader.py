# src/utils/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str, quantize: str = None, device_map: str = "auto"):
    """
    Load a causal LM with optional quantization.
    
    :param model_name: A string like "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
                      or local path to the checkpoint.
    :param quantize:  None, "8bit", or "4bit".
    :param device_map: Typically "auto" to shard across GPUs (if multiple).
    :return: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantize == "8bit":
        print("Loading model in 8-bit precision with bitsandbytes...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,               # Enable 8-bit
            device_map=device_map,           # e.g. "auto" or {0: ...} for single GPU
        )
    elif quantize == "4bit":
        print("Loading model in 4-bit precision with bitsandbytes...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,               # Enable 4-bit
            device_map=device_map,           # e.g. "auto"
            torch_dtype=torch.float16,       # Usually best to keep compute dtype at 16-bit
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        # Default FP16 or FP32 load
        # If you want standard FP16 on GPU (and your GPU supports it), you can do:
        # torch_dtype=torch.float16
        # device_map="auto"
        print("Loading model in standard precision (no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
        )

    return tokenizer, model
