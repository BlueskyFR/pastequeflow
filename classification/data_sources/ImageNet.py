from typing import List, Dict, Tuple, Union
from numpy import ndarray
from pandas import DataFrame
from tensorflow.python.data import Dataset

import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset

from .. import IDataSource

AUTOTUNE = tf.data.AUTOTUNE

class ImageNet(IDataSource):
    def __init__(
            self,
            train_dir: str,
            class_mappings_path: str,
            val_dir: str = None, validation_split: float = .2,
            test_dir: str = None, testing_split: float = .1
    ):
        # Safety checks
        if val_dir is None and validation_split is None \
            or test_dir is None and testing_split is None:
                raise ValueError("both val_dir and validation_split "
                                 "or test_dir and testing_split cannot be None!")
        
        self._train_dir = train_dir
        self._class_mappings_path = class_mappings_path
        self._val_dir = val_dir
        self._validation_split = validation_split
        self._test_dir = test_dir
        self._testing_split = testing_split
        
        # Load the datasets and the mappings
        self._load_datasets()
        self._load_class_mappings()
        self._map_datasets()
    
    def _load_datasets(self) -> None:
        train_ds = self._get_file_list_dataset(self._train_dir, shuffle=True)
        file_count = len(train_ds)
        print(f"{file_count} files loaded from the training dataset!")
        
        # Load the validation dataset if a directory was specified,
        # otherwise split the training dataset
        if self._val_dir is not None:
            val_ds = self._get_file_list_dataset(self._val_dir)
            print(f"{len(val_ds)} files loaded from the validation dataset!")
        else:
            val_size = int(file_count * self._validation_split)
            val_ds = train_ds.take(val_size)
            train_ds = train_ds.skip(val_size)
        
        # Same thing for the testing dataset
        if self._test_dir is not None:
            test_ds = self._get_file_list_dataset(self._test_dir)
        else:
            test_size = int(file_count * self._testing_split)
            test_ds = train_ds.take(test_size)
            train_ds = train_ds.skip(test_size)
        
        # Store the datasets in the class object
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._test_ds = test_ds
        
    def _get_file_list_dataset(self, directory: str, shuffle: bool = False) -> Dataset:
        # Get the file list from directory
        ds = Dataset.list_files(str(Path(directory) / "*/*"), shuffle=False)
        if shuffle:
            # Shuffle the filenames
            ds = ds.shuffle(len(ds), reshuffle_each_iteration=False)
        
        return ds
    
    def _load_class_mappings(self):
        json_mappings = json.load(open(self._class_mappings_path))
        self._class_mappings = {
            class_id: class_names[1]
            for class_id, class_names in json_mappings.items()
        }
        self._synset_wnip_classes = np.array([
            class_names[0]
            for _, class_names in json_mappings.items()
        ])
        self._classes = np.array([word_id for _, word_id in json_mappings.values()])
    
    def _map_datasets(self):
        def get_label(file_path: tf.Tensor) -> tf.Tensor:
            # Extract the synset wnid
            synset_wnid = tf.strings.regex_replace(
                file_path,
                pattern=r"(n\d+)_\d+.JPEG$",
                rewrite="\1" # Rewrite by replacing all by the first parenthesized group
            )
            
            one_hot = synset_wnid == self._synset_wnip_classes
            return tf.argmax(one_hot)
        
        def decode_img(img) -> tf.Tensor:
            # Convert the compressed string to a 3D uint8 tensor
            #img = tf.io.decode_jpeg(img, channels=3)
            return tf.io.decode_jpeg(img, channels=3)
            # Resize the image to the desired size
            #return tf.image.resize(img, self._images_size)
        
        def process_path(file_path: tf.Tensor):
            label = get_label(file_path)
            # Load the raw data from the file as a string
            img = tf.io.read_file(file_path)
            img = decode_img(img)
            return img, label
        
        # Save the datasets lengths since map breaks len(ds)
        self._train_ds_len =len(self._train_ds)
        self._val_ds_len = len(self._val_ds)
        self._test_ds_len = len(self._test_ds)
        
        # Map the 3 datasets files to their corresponding (image, label) pairs
        self._train_ds = self._train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        self._val_ds = self._val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        self._test_ds = self._test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    @property
    def classes(self) -> ndarray:
        return self._classes

    @property
    def class_mappings(self) -> Dict[int, str]:
        return self._class_mappings

    @property
    def classes_repartition(self) -> Union[DataFrame, None]:
        # We are not getting the classes repartition at the moment here
        return None

    @property
    def weights(self) -> Union[Dict[int, float], None]:
        # We are not getting the classes repartition at the moment here
        return None

    def get_training_dataset(self) -> Dataset:
        return self._train_ds

    def get_validation_dataset(self) -> Dataset:
        return self._val_ds

    def get_testing_dataset(self) -> Dataset:
        return self._test_ds
    
    @property
    def train_ds_len(self) -> int:
        return self._train_ds_len
    
    @property
    def val_ds_len(self) -> int:
        return self._val_ds_len
    
    @property
    def test_ds_len(self) -> int:
        return self._test_ds_len