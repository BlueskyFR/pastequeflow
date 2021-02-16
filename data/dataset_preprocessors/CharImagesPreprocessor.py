from math import pi

from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow.python.data import Dataset

from pastequeflow.data import IDatasetPreprocessor


class CharImagesPreprocessor(IDatasetPreprocessor):

    def data_augmentation_layers(self) -> Sequential:
        return Sequential([
            Normalization(),
            RandomContrast(0.1),  # Random contrast factor picked between [1.0 - 0.1, 1.0 + 0.1]
            RandomZoom((0, -0.3)),  # A negative value means zooming in
            RandomRotation(pi / 18),  # 10°
            RandomTranslation(0.1)
        ])

    def prepare(self, dataset: Dataset, shuffle: bool = False, augment: bool = False) -> Dataset:
        pass
