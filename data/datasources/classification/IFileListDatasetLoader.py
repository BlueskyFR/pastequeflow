from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from numpy import ndarray
from pandas import DataFrame
from tensorflow.python.data import Dataset


class IFileListDatasetLoader(ABC):

    @property
    @abstractmethod
    def classes(self) -> ndarray:
        pass

    @property
    @abstractmethod
    def class_mappings(self) -> Dict[int, str]:
        pass

    @property
    @abstractmethod
    def classes_repartition(self) -> DataFrame:
        pass

    @property
    @abstractmethod
    def weights(self) -> Dict[int, float]:
        pass

    @abstractmethod
    def get_training_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_validation_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_testing_dataset(self) -> Dataset:
        pass
