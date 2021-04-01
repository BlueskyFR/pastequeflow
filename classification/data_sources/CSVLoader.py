from typing import List, Dict, Tuple, Union

import tensorflow as tf
from tensorflow.python.data import Dataset
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from .. import IDataSource

AUTOTUNE = tf.data.AUTOTUNE

class CSVLoader(IDataSource):
    def __init__(
        self,
        train_csv_path: str,
        x_col: str,
        y_col: str,
        train_base_path: str = None,
        val_csv_path: str = None, val_base_path: str = None, validation_split: float = .2,
        test_csv_path: str = None, test_base_path: str = None, testing_split: float = .1
    ):
        # Safety checks
        if val_csv_path is None and validation_split is None:
            raise ValueError("Both val_csv_path and validation_split cannot be None!")
        if test_csv_path is None and testing_split is None:
            raise ValueError("or test_csv_path and testing_split cannot be None!")
        if val_csv_path is None and val_base_path is not None:
            raise ValueError("val_base_path can only be set if val_csv_path is not None!")
        if test_csv_path is None and test_base_path is not None:
            raise ValueError("test_base_path can only be set if test_csv_path is not None!")
        
        self._train_csv_path = train_csv_path
        self._x_col = x_col
        self._y_col = y_col
        self._train_base_path = train_base_path
        self._val_csv_path = val_csv_path
        self._val_base_path = val_base_path
        self._validation_split = validation_split
        self._test_csv_path = test_csv_path
        self._test_base_path = test_base_path
        self._testing_split = testing_split
        
        # Load the datasets and the mappings
        self._load_datasets()
        self._load_class_mappings()
        self._map_datasets()
    
    def _load_datasets(self) -> None:
        train_df = pd.read_csv(self._train_csv_path)
        # Save the y values for later (needed for some properties)
        self._y_values = train_df[self._y_col]
        train_ds = self._csv_df_to_dataset(train_df, shuffle=True)
        file_count = len(train_ds)
        print(f"{file_count} files loaded from the training dataset!")
        
        # Load the validation dataset if a directory was specified,
        # otherwise split the training dataset
        if self._val_csv_path is not None:
            val_ds = self._csv_df_to_dataset(pd.read_csv(self._val_csv_path))
            print(f"{len(val_ds)} files loaded from the validation dataset!")
        else:
            val_size = int(file_count * self._validation_split)
            val_ds = train_ds.take(val_size)
            train_ds = train_ds.skip(val_size)
        
        # Same thing for the testing dataset
        if self._test_csv_path is not None:
            test_ds = self._csv_df_to_dataset(pd.read_csv(self._test_csv_path), is_test_csv=True)
        else:
            test_size = int(file_count * self._testing_split)
            test_ds = train_ds.take(test_size)
            train_ds = train_ds.skip(test_size)
        
        # Store the datasets in the class object
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._test_ds = test_ds
    
    def _csv_df_to_dataset(self, df: DataFrame, shuffle: bool = False, is_test_csv: bool = False) -> Dataset:
        # Replace `train/00000.image.jpg` by `00000.image.jpg`
        #f[self.__x_col].replace(to_replace=".*/", value="", regex=True, inplace=True)
        
        # Some aliases to make code easier to read
        x_col = self._x_col
        y_col = self._y_col
        
        if not y_col in df:
            if not is_test_csv:
                raise Exception(f"❌ The {y_col} column is missing in the training and/or validation csv file(s)!")
            
            print(f"👉 The y column (\"{y_col}\") was not found in the testing csv, so it will be omitted!")
            # Only keep the x column
            df = df[[x_col]]
            # Convert the dataframe to a tf.data.Dataset
            ds = Dataset.from_tensor_slices((df[x_col].values))
            
        else:
            # Only keep the x and y columns
            df = df[[x_col, y_col]]
            # Convert the dataframe to a tf.data.Dataset
            ds = Dataset.from_tensor_slices((df[x_col].values, df[y_col].values))
        
        if shuffle:
            # Shuffle the filenames
            ds = ds.shuffle(len(ds), reshuffle_each_iteration=False)
        
        return ds

    def _load_class_mappings(self):
        self._classes = np.unique(self._y_values.to_numpy()).astype(np.str)
        self._class_mappings = {
            i: class_name
            for i, class_name in enumerate(self._classes)
        }
        self._classes_repartition = self._y_values.value_counts().to_dict()
        
        # Weights computation
        sample_count = self._y_values.size
        class_count = len(self._classes)
        self._weights = {
            i: sample_count / (self._classes_repartition[class_name] * class_count)
            for i, class_name in enumerate(self._classes)
        }
    
    def _map_datasets(self):
        def get_label(y) -> tf.Tensor:
            one_hot = y == self._classes
            #return tf.cast(one_hot, tf.float32)
            return tf.argmax(one_hot)
        
        def decode_img(img) -> tf.Tensor:
            # Convert the compressed string to a 3D uint8 tensor
            #img = tf.io.decode_jpeg(img, channels=3)
            return tf.io.decode_jpeg(img, channels=3)
            # Resize the image to the desired size
            #return tf.image.resize(img, self._images_size)
        
        def process_tuple(x: tf.Tensor, base_path: str = None, y: tf.Tensor = None):
            # Load the raw data from the file as a string
            if base_path is not None:
                x = tf.strings.join([base_path, x])
            img = tf.io.read_file(x)
            img = decode_img(img)
            if y is None:
                return img
            label = get_label(y)
            return img, label
        
        # Save the datasets lengths since map breaks len(ds)
        self._train_ds_len = len(self._train_ds)
        self._val_ds_len = len(self._val_ds)
        self._test_ds_len = len(self._test_ds)
        
        # Map the 3 datasets files to their corresponding (image, label) pairs
        self._train_ds = self._train_ds.map(
            lambda x, y: process_tuple(x, self._train_base_path, y),
            num_parallel_calls=AUTOTUNE
        )
        val_base_path = self._val_base_path if self._val_base_path is not None else self._train_base_path
        self._val_ds = self._val_ds.map(
            lambda x, y: process_tuple(x, val_base_path, y),
            num_parallel_calls=AUTOTUNE
        )
        test_base_path = self._test_base_path if self._test_base_path is not None else self._train_base_path
        self._test_ds = self._test_ds.map(
            lambda x, y = None: process_tuple(x, test_base_path, y),
            num_parallel_calls=AUTOTUNE
        )

    @property
    def classes(self) -> ndarray:
        return self._classes

    @property
    def class_mappings(self) -> Dict[int, str]:
        return self._class_mappings

    @property
    def classes_repartition(self) -> Union[Dict[str, int], None]:
        return self._classes_repartition

    @property
    def weights(self) -> Union[Dict[int, float], None]:
        return self._weights

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
        