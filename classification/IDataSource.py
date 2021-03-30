from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union

from numpy import ndarray
from pandas import DataFrame
from tensorflow.python.data import Dataset


class IDataSource(ABC):

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
    def classes_repartition(self) -> Union[DataFrame, None]:
        pass

    @property
    @abstractmethod
    def weights(self) -> Union[Dict[int, float], None]:
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
    
    def get_all_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        return (
            self.get_training_dataset(),
            self.get_validation_dataset(),
            self.get_testing_dataset()
        )
    
    @property
    @abstractmethod
    def train_ds_len(self) -> int:
        pass
    
    @property
    @abstractmethod
    def val_ds_len(self) -> int:
        pass
    
    @property
    @abstractmethod
    def test_ds_len(self) -> int:
        pass
    
    def get_all_lengths(self) -> Tuple[int, int, int]:
        return (
            self.train_ds_len,
            self.val_ds_len,
            self.test_ds_len
        )
        