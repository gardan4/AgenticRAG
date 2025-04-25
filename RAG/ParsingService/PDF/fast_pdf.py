#For paid version
from unstructured.staging.base import dict_to_elements
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
import unstructured_client
from unstructured.partition.auto import partition
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def pdf_to_elements_fast(pdf_path):
    """
    Extracts elements from a pdf file by utilizing the low resolution (fast) pdf extractor without the need of an API
    
    Args:
    pdf_path (String): Path to the pdf file for information extraction.
    
    Returns:
    A list of Unstructured Elements
    """
    try:
        elements = partition(filename=pdf_path,strategy='fast')
        return  elements
    except Exception as e:
        logger.info(f"Exceptuion:")
        logger.info(e)
        print(e)
        return []