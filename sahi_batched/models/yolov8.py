from sahi.models.yolov8 import Yolov8DetectionModel as Yolov8DetectionModelBase
from typing import List
import numpy as np

class Yolov8DetectionModel(Yolov8DetectionModelBase):
    def perform_inference(self, images: List[np.ndarray]):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        prediction_result = self.model.predict(source=images, verbose=False, device=self.device)

        prediction_result = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold] for result in prediction_result
        ]

        self._original_predictions = prediction_result
