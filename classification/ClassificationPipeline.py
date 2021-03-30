from typing import List, Union
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras import Model
import math

from . import IDataSource, IDatasetPreprocessor
from .. import IPipeline, IModel

class Pipeline(IPipeline):
    def __init__(
        self,
        data_source: IDataSource,
        dataset_preprocessor: IDatasetPreprocessor,
        model: IModel,
        metrics: Union[List[Metric], List[str]],
        epochs: int
        ) -> None:
        
        self._data_source = data_source
        self._preprocessor = dataset_preprocessor
        self._model = model
        self._metrics = metrics
        self._epochs = epochs
    
    #TODO: display metrics
    
    def run(self):
        # Load the data
        train_ds, val_ds, test_ds = self._data_source.get_all_datasets()
        train_ds_len, val_ds_len, test_ds_len = self._data_source.get_all_lengths()
        
        # Preprocess the data
        train_ds = self._preprocessor.prepare(train_ds, ds_len=train_ds_len, shuffle=True, augment=True)
        val_ds = self._preprocessor.prepare(val_ds, ds_len=val_ds_len)
        test_ds = self._preprocessor.prepare(test_ds, ds_len=test_ds_len)
        
        # Get the compiled model
        class_count = len(self._data_source.classes)
        model = self._model.get_model(
            input_shape=self._preprocessor.resulting_shape,
            output_classes_count=class_count,
            metrics=self._metrics
        )
        
        # TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)
        tf.profiler.experimental.server.start(6007)
        
        bc = self._preprocessor.batch_size
        print(f"[train] Steps per epoch: {math.ceil(train_ds_len / bc)}")
        print(f"[val] Steps per epoch: {math.ceil(val_ds_len / bc)}")
        print(f"[test] Steps per epoch: {math.ceil(test_ds_len / bc)}")
        
        # Train the model
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self._epochs,
            callbacks=[tensorboard_callback]
        )