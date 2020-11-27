from tensorflow_core.python.keras.metrics import MeanMetricWrapper, sparse_categorical_accuracy
import numpy as np
import tensorflow as tf


class SparseCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, class_weight=None, name='sparse_categorical_accuracy', dtype=None):
        super(SparseCategoricalAccuracy, self).__init__(
            sparse_categorical_accuracy, name, dtype=dtype)
        self.class_weight = tf.convert_to_tensor(np.array(class_weight)[:, np.newaxis]) if class_weight else None

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.class_weight is not None and sample_weight is None:
            sample_weight = tf.gather_nd(self.class_weight, tf.cast(y_true, tf.int32))
            sample_weight = tf.reshape(sample_weight, tf.shape(y_true))
        return super(SparseCategoricalAccuracy, self).update_state(y_true, y_pred, sample_weight)