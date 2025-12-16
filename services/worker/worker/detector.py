# shim for detector used by worker; mirrors API's model_loader but simpler
import numpy as np

class DetectionModel:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self._model = None
        self._names = {0: "person", 1: "shelf", 2: "product_label"}

    def predict(self, img: np.ndarray):
        return []
