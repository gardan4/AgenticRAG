#This class extends Unstructured.io's ElementMetadata class and allows for additional fields to be added
from unstructured.documents.elements import ElementMetadata

class CustomElementMetadata(ElementMetadata):
    def __init__(self, filename, page_number=0, coordinates=None, languages=None, trust_score=50):
        # Call parent constructor to retain existing functionality
        super().__init__(filename=filename, page_number=page_number, coordinates=coordinates, languages=languages)
        
        #Custom fields:
        self.trust_score = trust_score