from abc import ABC, abstractmethod

from tensorflow.keras import Sequential
from tensorflow.python.data import Dataset


class IDatasetPreprocessor(ABC):
    @abstractmethod
    def prepare(self, dataset: Dataset, shuffle: bool = False, augment: bool = False) -> Dataset:
        pass