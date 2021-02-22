from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from numpy import ndarray
from pandas import DataFrame
from tensorflow.python.data import Dataset


class IDataLoader(ABC):

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

    # The following three methods both return a tuple (str, Dataset): (img_dir_path, x_y_str_dataset)
    # The str corresponds to the folder in which the images have to be read

    @abstractmethod
    def get_training_dataset(self) -> Tuple[str, Dataset]:
        pass

    @abstractmethod
    def get_validation_dataset(self) -> Tuple[str, Dataset]:
        pass

    @abstractmethod
    def get_testing_dataset(self) -> Tuple[str, Dataset]:
        pass
