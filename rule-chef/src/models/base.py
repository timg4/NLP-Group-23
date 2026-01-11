from typing import Dict, List

class NERModel:
    def fit(self, train_texts: List[str], train_spans: List[List[Dict]]):
        return self

    def predict(self, text: str) -> List[Dict]:
        raise NotImplementedError