from unstructured.partition.html.partition import partition_html
from typing import Optional

def html_to_elements(file_path: Optional[str]=None, url: Optional[str]=None):
    """
    Extracts elements from an html file or webpage
    If url is provided it will extract webpage, otherwise will look for a local html file
    
    Args:
    file_path (String): Path to the html file for information extraction.
    url (String): The webpage that should get its html extracted
    
    Returns:
    A list of Unstructured Elements
    """
    if(url!=None):
        return partition_html(url=url)
    else:
        return partition_html(filename=file_path)