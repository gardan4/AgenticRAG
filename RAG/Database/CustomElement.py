from unstructured.documents.elements import Element, CoordinateSystem, CoordinatesMetadata
from CustomElementMetadata import CustomElementMetadata
from typing import Optional

class CustomElement(Element):
    "Allows for the CustomElementMetadata class to be used with the Element class"
    def __init__(
        self,
        text: Optional[str] = None,
        category: Optional[str] = None,
        filename: Optional[str] = None,
        element_id: Optional[str] = None,
        coordinates: Optional[tuple[tuple[float, float], ...]] = None,
        coordinate_system: Optional[CoordinateSystem] = None,
        metadata: Optional[CustomElementMetadata] = None,
        detection_origin: Optional[str] = None,
    ):
        self.text = text

        self.category = category

        self.filename = filename

        if element_id is not None and not isinstance(element_id, str):
            raise ValueError("element_id must be of type str or None.")

        self._element_id = element_id

        self.metadata = CustomElementMetadata() if metadata is None else metadata

        if coordinates is not None or coordinate_system is not None:
            self.metadata.coordinates = CoordinatesMetadata(
                points=coordinates, system=coordinate_system
            )

        self.metadata.detection_origin = detection_origin
        self.text = self.text if hasattr(self, "text") else ""
