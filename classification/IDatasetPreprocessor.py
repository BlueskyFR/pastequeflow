from abc import ABC, abstractmethod
from typing import Tuple

from tensorflow.keras import Sequential
from tensorflow.python.data import Dataset


class IDatasetPreprocessor(ABC):
    @abstractmethod
    def prepare(self, ds: Dataset, ds_len: int, shuffle: bool = False, augment: bool = False) -> Dataset:
        pass
    
    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass
    
    @property
    @abstractmethod
    def resulting_shape(self) -> Tuple[int, int, int]:
        pass
