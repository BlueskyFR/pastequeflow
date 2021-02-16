from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from numpy import ndarray
from pandas import DataFrame


class IDatasource(ABC):

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
    def get_train_val_data(self) -> List[Tuple[str, str]]:
        pass

    @abstractmethod
    def get_testing_data(self) -> List[str]:
        pass
