import json
from pathlib import Path
from typing import Tuple, List, Dict
from numpy import ndarray
from pandas import DataFrame
from tensorflow.python.data import Dataset

from . import IDataLoader


class FileListFromClassDir(IDataLoader):

    def __init__(
            self,
            train_dir: str,
            val_dir: str = None, validation_split: int = .2,
            test_dir: str = None, testing_split: int = .1,
            class_mappings_path: str = None
    ):
        """
        :param train_dir: The directory containing the training images (~1.6 millions images)
        :param val_dir: The directory containing the validation images (~52 thousands images)
        :param validation_split: The proportion of the training set to keep for validation if val_dir is None.
                                 Defaults to .2
        :param test_dir:
        :param testing_split:
        :param class_mappings_path: The path to the class mappings (ex.: imagenet_class_index.json) json file
                                    under the format { "class_num [0-999]": [ "folder_name", "class_name" ], ... }.
                                    You can download the ImageNet class mappings at:
                                    https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
        """
        # List train files
        train_ds = Dataset \
            .list_files(Path(train_dir) / "*/*", shuffle=False) \
            .shuffle() #\
            #.can we get len(train_ds) to get its length for the shuffle buffer?

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

    def get_validation_dataset(self) -> Tuple[str, Dataset]:
        pass

    def get_testing_dataset(self) -> Tuple[str, Dataset]:
        pass
