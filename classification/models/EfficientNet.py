from typing import List, Tuple, Union
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Metric

from ... import IModel

class EfficientNetB4(IModel):
    def get_model(self, input_shape: Tuple[int, ...], output_classes_count: int, metrics: Union[List[Metric], List[str]]) -> Model:
        model = tf.keras.applications.EfficientNetB4(
            weights="imagenet",
            include_top=False,
            classes=output_classes_count,
            input_tensor=tf.keras.Input(shape=input_shape),
            pooling="max"
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(), #TODO: check existing optimizers and use the best
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=metrics
        )
        
        return model