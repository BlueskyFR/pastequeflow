from math import pi
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow.python.data import Dataset

AUTOTUNE = tf.data.AUTOTUNE

from .. import IDatasetPreprocessor

class CharImagesPreprocessor(IDatasetPreprocessor):

    def __init__(self, desired_img_width: int, desired_img_height: int, batch_size: int, workers: int = AUTOTUNE) -> None:
        self._batch_size = batch_size
        self._workers = workers
        # As of now, we consider the channel count to always be 3
        #TODO: ask for the channel count as argument
        self._resulting_shape = (desired_img_width, desired_img_height, 3)
        
        self._resize_and_rescale = Sequential([
            Resizing(width=desired_img_width, height=desired_img_height),
            #Rescaling(1. / 255)
        ])
        
        self._data_augmentation = Sequential([
            Normalization(),
            RandomContrast(0.3),  # Random contrast factor picked between [1.0 - 0.1, 1.0 + 0.1]
            RandomZoom((0, -0.3)),  # A negative value means zooming in
            RandomRotation(pi / 12),  # 15°
            RandomTranslation(0.15, 0.15)
        ])

    def prepare(self, ds: Dataset, ds_len: int, shuffle: bool = False, augment: bool = False) -> Dataset:
        #TODO: move the general calls such as shuffle, batch, prefetch etc. to Parent class
        print(f"Preparing dataset of {ds_len} elements...")
        
        # Resize and rescale all datasets
        ds = ds.map(
            lambda x, y = None: 
                (self._resize_and_rescale(x), y) if y is not None
                else self._resize_and_rescale(x),
            num_parallel_calls=self._workers
        )
        
        # Cache all the datasets
        ds = ds.cache()
        
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)#ds_len)
            
        # Batch all datasets
        ds = ds.batch(self._batch_size)    
        
        # Use data augmentation only on the training set
        if augment:
            ds = ds.map(
                lambda x, y = None:
                    (self._data_augmentation(x, training=True), y) if y is not None
                    else self._data_augmentation(x, training=True),
                num_parallel_calls=self._workers
            )
        
        # Use buffured prefetching on all datasets
        return ds.prefetch(buffer_size=self._workers)
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def resulting_shape(self) -> Tuple[int, int, int]:
        return self._resulting_shape
