from typing import List, Dict, Tuple
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from tensorflow.python.data import Dataset

from .. import IDataSource


class CSVLoader(IDataSource):
    def __init__(self, train_val_csv_path: str, test_csv_path: str, x_col: str, y_col: str):
        self.__test_csv_path = test_csv_path
        self.__x_col = x_col
        self.__y_col = y_col

        # Load the data from the training CSV file
        # Note that we do not load the testing data here, because it is lazy loaded from the `get_testing_data` function
        df = self.__load_csv(path=train_val_csv_path)
        self.__train_val_data = df[[x_col, y_col]]

        # Cache class list (multiple calls to it)
        y_values = self.__train_val_data[self.__y_col]
        self.__classes = np.unique(y_values.to_numpy()).astype(np.str)
        # Cache the classes repartition (multiple calls to it)
        self.__classes_repartition = y_values.value_counts()

    @property
    def classes(self) -> ndarray:
        # Return cached class list
        return self.__classes

    @property
    def class_mappings(self) -> Dict[int, str]:
        # Create mappings form the class list
        return {i: class_name for i, class_name in enumerate(self.__classes)}

    @property
    def classes_repartition(self) -> DataFrame:
        return self.__classes_repartition

    @property
    def weights(self) -> Dict[int, float]:
        sample_count = self.__train_val_data[self.__y_col].size
        class_count = len(self.__classes)

        return {
            i: sample_count / (self.__classes_repartition[class_name] * class_count)
            for i, class_name in enumerate(self.__classes)
        }

    def get_train_val_data(self) -> List[List[str]]:  #List[Tuple[str, str]]:
        # Convert the dataframe to a numpy tuples array: [(x, y), ...]
        # return self.__train_val_data.astype(str).to_records(index=False).tolist()
        return self.__train_val_data.astype(str).values.tolist()

    def get_testing_data(self) -> List[str]:
        df = self.__load_csv(path=self.__test_csv_path)
        # Convert the dataframe to a numpy tuples array: [(x, y), ...]
        return df[self.__x_col].to_list()

    def get_training_dataset(self) -> Tuple[str, Dataset]:
        pass

    def get_validation_dataset(self) -> Tuple[str, Dataset]:
        pass

    def get_testing_dataset(self) -> Tuple[str, Dataset]:
        pass

    def __load_csv(self, path):
        df = pd.read_csv(path)
        # Replace `train/00000.image.jpg` by `00000.image.jpg`
        df[self.__x_col].replace(to_replace=".*/", value="", regex=True, inplace=True)

        return df
