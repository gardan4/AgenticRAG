from unstructured.partition.docx import partition_docx
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def docx_to_elements(file_path: str):
    """
    Extracts elements from a docx file
    
    Args:
    file_path (String): Path to the docx file for information extraction.
    
    Returns:
    A list of Unstructured Elements
    """

    try:
        return partition_docx(filename=file_path)
    except Exception as e:
        logger.info(f"Exception:")
        logger.info(e)
        print(e)
        return []