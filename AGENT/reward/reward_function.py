# src/reward/reward_function.py

def compute_reward(generated_text: str, reference_text: str) -> float:
    """
    Compute a reward based on correctness of the model output.
    Here, you might measure exact match or partial overlap with reference.
    """
    # Example: A simplistic check for substring presence
    if reference_text.lower() in generated_text.lower():
        return 1.0
    return 0.0
