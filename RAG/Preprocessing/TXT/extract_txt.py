from unstructured.partition.text import partition_text

def txt_to_elements(file_path: str):
    """
    Extracts elements from a txt file.
    
    Args:
    file_path (str): Path to the txt file for information extraction.
    
    Returns:
    A list of Unstructured Elements
    """
    return partition_text(filename=file_path)
