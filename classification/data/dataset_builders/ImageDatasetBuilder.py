from functools import partial
from pathlib import Path
from typing import Tuple, List

from numpy import ndarray
import tensorflow as tf
from tensorflow.python.data import Dataset

AUTOTUNE = tf.data.AUTOTUNE

from .. import IDatasetBuilder


class ImageDatasetBuilder(IDatasetBuilder):
    def __init__(self, train_img_dir: str, test_img_dir: str, validation_split: float = 0.2):
        self._train_dir = str(Path(train_img_dir))
        self._test_dir = str(Path(test_img_dir))
        self.__val_split = validation_split

    # We define the following methods + the _train_dir attribute as protected instead of private because AutoGraph does
    # not yet handle mangled names (aka private members, aka arguments beginning but not also ending with "__")
    def _get_label(self, y: tf.Tensor, classes: ndarray):
        one_hot = y == classes
        # Convert the one hot encoding to int
        return tf.argmax(one_hot)

    def _decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        return tf.image.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        # img = tf.image.resize(img, [img_height, img_width])

    def get_train_val_datasets(self, x_y_data: List[List[str]], classes: ndarray) -> Tuple[Dataset, Dataset]:
        image_count = len(x_y_data)
        list_ds = Dataset \
            .from_tensor_slices(x_y_data) \
            .shuffle(image_count, reshuffle_each_iteration=False)  # We will shuffle in the preparation step

        # for f in list_ds.take(5):
        #     print(f.numpy())

        val_size = int(image_count * self.__val_split)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)

        # print(f"total train/val image count: {image_count}")
        # print(f"              train_ds size: {tf.data.experimental.cardinality(train_ds).numpy()}")
        # print(f"                val_ds size: {tf.data.experimental.cardinality(val_ds).numpy()}")

        # Map the dataset image names + string labels to actual images and int labels
        def process_path(tensor: tf.Tensor):
            img_name = tensor[0]
            y = tensor[1]
            label = self._get_label(y, classes)
            # Load the raw data from the file as a string
            img = tf.io.read_file(self._train_dir + "/" + img_name)
            img = self._decode_img(img)
            return img, label

        train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

        # for image, label in train_ds.take(1):
        #     print("Image shape:", image)
        #     print("Label:", label)

        return train_ds, val_ds

    def get_testing_dataset(self, x_y_data: List[str], classes: ndarray) -> Dataset:
        image_count = len(x_y_data)
        test_ds = Dataset.from_tensor_slices(x_y_data)

        # Map the dataset image names to actual images
        def process_path(img_name: tf.Tensor):
            # Load the raw data from the file as a string
            img = tf.io.read_file(self._test_dir + "/" + img_name)
            img = self._decode_img(img)
            return img

        return test_ds.map(process_path)
