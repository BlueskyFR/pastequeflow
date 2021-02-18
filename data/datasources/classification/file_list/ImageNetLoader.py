import json
from typing import Tuple, List, Dict
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from tensorflow.python.data import Dataset

from .. import IFileListDatasetLoader


class ImageNetLoader(IFileListDatasetLoader):
    def __init__(self, train_dir: str, val_dir: str, class_mappings_path: str, test_size: int):
        """
        :param train_dir: The directory containing the training images (~1.6 millions images)
        :param val_dir: The directory containing the validation images (~52 thousands images)
        :param class_mappings_path: The path to imagenet_class_index.json containing the class mappings
                                    under the format { "class_num [0-999]": [ "folder_name", "class_name" ], ... }.
                                    You can download it at
                                    https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
        :param test_size: Since no labeled test dataset is provided officially, specify the image count that will be
                          moved from the training set (the bigger one) to a new test dataset
        """
        # Load classes
        json_mappings = json.load(open(class_mappings_path))
        self.__class_mappings = {i: full_class_name for i, full_class_name in json_mappings.items()}
        self.__classes = [word_id for _, word_id in json_mappings.values()]

        self.__train_data = Dataset.list_files()

        # Cache the classes repartition
        #self.__classes


    @property
    def classes(self) -> ndarray:
        pass

    @property
    def class_mappings(self) -> Dict[int, str]:
        pass

    @property
    def classes_repartition(self) -> DataFrame:
        pass

    @property
    def weights(self) -> Dict[int, float]:
        pass

    def get_training_dataset(self) -> List[Tuple[str, str]]:
        pass

    def get_testing_data(self) -> List[str]:
        pass
