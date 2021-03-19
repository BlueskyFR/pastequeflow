from math import pi

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow.python.data import Dataset

AUTOTUNE = tf.data.AUTOTUNE

from .. import IDatasetPreprocessor

class ImageNetPreprocessor(IDatasetPreprocessor):
    
    def __init__(self, images_width: int, images_height: int, batch_size: int) -> None:
        self._batch_size = batch_size
        
        self._resize_and_rescale = Sequential([
            Resizing(width=images_width, height=images_height),
            Rescaling(1. / 255)
        ])
        
        self._data_augmentation = Sequential([
            Normalization(),
            RandomContrast(0.1),  # Random contrast factor picked between [1.0 - 0.1, 1.0 + 0.1]
            RandomZoom((0, -0.3)),  # A negative value means zooming in
            RandomRotation(pi / 18),  # 10°
            RandomTranslation(0.1, 0.1)
        ])

    def prepare(self, ds: Dataset, shuffle: bool = False, augment: bool = False) -> Dataset:
        #TODO: move the general calls such as shuffle, batch, prefetch etc. to Parent class
        # Resize and rescale all datasets
        ds = ds.map(
            lambda x, y: (self._resize_and_rescale(x), y),
            num_parallel_calls=AUTOTUNE
        )
        
        if shuffle:
            ds = ds.shuffle(len(ds))
        
        # Batch all datasets
        ds = ds.batch(self._batch_size)
        
        # Use data augmentation only on the training set
        if augment:
            ds = ds.map(
                lambda x, y: (self._data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE
            )
        
        # Use buffured prefetching on all datasets
        return ds.prefetch(buffer_size=AUTOTUNE)
