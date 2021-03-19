from typing import List, Union
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Metric

from ... import IModel

class EfficientNetB4(IModel):
    def get_model(self, output_classes_count: int, metrics: Union[List[Metric], List[str]]) -> Model:
        model = tf.keras.applications.EfficientNetB4(weights="imagenet", classes=output_classes_count)
        
        model.compile(
            optimizer=tf.optimizers.Adam, #TODO: check existing optimizers and use the best
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=metrics
        )
        
        return model