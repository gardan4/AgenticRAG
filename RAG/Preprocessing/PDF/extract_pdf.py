from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from unstructured_inference.inference.layout import DocumentLayout
from unstructured.partition.pdf_image.ocr import process_file_with_ocr

#For free version
from unstructured.partition.auto import partition
import os

#Needs installed poppler and tesseract locally
def pdf_to_elements(pdf_path) :
    """
    Extracts elements from a pdf file by utilizing yolox through unstructured.io repo
    
    Args:
    pdf_path (String): Path to the pdf file for information extraction.
    
    Returns:
    A list of Unstructured Elements
    """
    try:
        elements = partition(filename=pdf_path,strategy='hi_res',skip_infer_table_types=[])
        i = -1
        for el in elements:
            i+=1
            if(el.category =='Table'): #If a table is found it will try to convert it to html for better context understanding when it is later needed
                elements[i].text=elements[i].metadata.text_as_html
            if(el.category =='Header'):
                elements[i].category="Title"
        return  elements
    except Exception as e:
        print(e)
        return []
