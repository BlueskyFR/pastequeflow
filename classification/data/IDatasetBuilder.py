from abc import ABC, abstractmethod
from typing import Tuple, List

from numpy import ndarray
from tensorflow.data import Dataset


class IDatasetBuilder(ABC):
    @abstractmethod
    def get_train_val_datasets(self, x_y_data: List[Tuple[str, int]], classes: List[str]) -> Tuple[Dataset, Dataset]:
        pass

    @abstractmethod
    def get_testing_dataset(self, x_data: List[str], classes: ndarray) -> Dataset:
        pass
