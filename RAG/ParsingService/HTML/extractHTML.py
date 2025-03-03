from unstructured.partition.html.partition import partition_html
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def html_to_elements(file_path: str):
    """
    Extracts elements from an html file
    
    Args:
    file_path (String): Path to the html file for information extraction.
    
    Returns:
    A list of Unstructured Elements
    """

    try:
        return partition_html(filename=file_path)
    except Exception as e:
        logger.info(f"Exception:")
        logger.info(e)
        print(e)
        return []

