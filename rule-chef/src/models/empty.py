from typing import Dict, List
from src.models.base import NERModel

class EmptyNER(NERModel):
    def predict(self, text: str) -> List[Dict]:
        return []