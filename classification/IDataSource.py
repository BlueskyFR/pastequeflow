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
