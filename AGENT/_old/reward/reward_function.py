# src/reward/reward_function.py

import torch
from typing import Optional
from sentence_transformers import SentenceTransformer
import os

# Global model cache to avoid reloading
_model_cache = {}

def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads and caches SentenceTransformer models to avoid reloading them repeatedly.
    
    Args:
        model_name: Name of the SentenceTransformer model to use
    
    Returns:
        SentenceTransformer model
    """
    global _model_cache
    
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
        print(f"Loaded {model_name}")
    
    return _model_cache[model_name]

def compute_similarity(text1: str, text2: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    """
    Compute semantic similarity between two texts using SentenceTransformer embeddings.
    
    Args:
        text1: First text
        text2: Second text
        model_name: SentenceTransformer model to use
    
    Returns:
        Similarity score between 0 and 1
    """
    model = get_model(model_name)
    
    # Compute embeddings
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    
    return similarity.item()

def compute_reward(generated_text: str, reference_text: str, 
                   model_name: Optional[str] = None) -> float:
    """
    Compute a reward based on semantic similarity between model output and reference.
    
    Args:
        generated_text: Text generated by the model being trained
        reference_text: Reference/ground truth text
        model_name: Name of the SentenceTransformer model to use
    
    Returns:
        Similarity score between 0 and 1
    """
    # Use environment variable if available, otherwise use default
    if model_name is None:
        model_name = os.environ.get("REWARD_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # If reference text is empty, no meaningful similarity can be computed
    if not reference_text or not generated_text:
        return 0.0
    
    # Calculate semantic similarity
    similarity_score = compute_similarity(generated_text, reference_text, model_name)
    
    return similarity_score

#TEST CODE
if __name__ == "__main__":
    print("Testing reward function with example data...")
    
    # Example sprint goals - similar but with different wording
    reference_goal = "Improve system performance and fix critical bugs in the API"
    generated_goals = [
        "Enhance system performance while addressing critical API bugs",
        "Fix core API issues and optimize overall system performance",
        "Resolve major bugs in the API and improve performance",
        "Add new features to the user interface and update documentation",  # Less similar
        "Deploy the application to production environment"  # Not similar
    ]
    
    # Test with default model
    print("\nTesting with default model (all-MiniLM-L6-v2):")
    for i, goal in enumerate(generated_goals):
        score = compute_reward(goal, reference_goal)
        print(f"Example {i+1} - Score: {score:.4f}")
        print(f"Reference: {reference_goal}")
        print(f"Generated: {goal}")
        print("-" * 50)
    
    # Test edge cases
    print("\nTesting edge cases:")
    print(f"Empty generated text: {compute_reward('', reference_goal)}")
    print(f"Empty reference text: {compute_reward(generated_goals[0], '')}")
    print(f"Empty both: {compute_reward('', '')}")
