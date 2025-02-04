from unstructured.partition.docx import partition_docx

def docx_to_elements(file_path: str):
    """
    Extracts elements from a docx file
    
    Args:
    file_path (String): Path to the docx file for information extraction.
    
    Returns:
    A list of Unstructured Elements
    """
    return partition_docx(filename=file_path)