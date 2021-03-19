from typing import List, Union
from tensorflow.keras.metrics import Metric
from tensorflow.keras import Model

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
        
        # Preprocess the data
        train_ds = self._preprocessor.prepare(train_ds, shuffle=True, augment=True)
        val_ds = self._preprocessor.prepare(val_ds)
        test_ds = self._preprocessor.prepare(test_ds)
        
        # Get the compiled model
        class_count = len(self._data_source.classes)
        model = self._model.get_model(
            output_classes_count=class_count,
            metrics=self._metrics
        )
        
        # Train the model
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self._epochs
        )