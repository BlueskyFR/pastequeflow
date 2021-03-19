from abc import ABC, abstractmethod
from typing import List, Union

from tensorflow.keras import Model
from tensorflow.keras.metrics import Metric


class IModel(ABC):
    @abstractmethod
    def get_model(self, output_classes_count: int, metrics: Union[List[Metric], List[str]]) -> Model:
        """Returns a compiled model

        Args:
            output_classes_count (int): the number of output classes
            metrics (Union[List[Metric], List[str]]): A list of metrics to use for the model compilation

        Returns:
            Model: A compiled keras model
        """
        pass
