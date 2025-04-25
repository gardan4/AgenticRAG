#This class extends Unstructured.io's ElementMetadata class and allows for additional fields to be added
from unstructured.documents.elements import ElementMetadata

class CustomElementMetadata(ElementMetadata):
    def __init__(self, filename="Unknown", page_number=0, coordinates=None, languages=None, trust_score=50):
        # Call parent constructor to retain existing functionality
        super().__init__(filename=filename, page_number=page_number, coordinates=coordinates, languages=languages)
        
        #Custom fields:
        self.trust_score = trust_score

    def to_dict(self):
        base_metadata = super().to_dict()  # Get the default metadata dictionary
        # Add custom fields
        base_metadata["trust_score"] = self.trust_score
        return base_metadata