import tensorflow as tf
import numpy as np


class SparseCategoricalMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, class_weight=None, name='sparse_categorical_mean_iou', dtype=None):
        self.class_weight = tf.convert_to_tensor(np.array(class_weight)[:, np.newaxis]) if class_weight else None
        super(SparseCategoricalMeanIoU, self).__init__(num_classes, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.keras.backend.argmax(y_pred, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
        if self.class_weight is not None and sample_weight is None:
            sample_weight = tf.gather_nd(self.class_weight, tf.cast(y_true, tf.int32))
            sample_weight = tf.reshape(sample_weight, tf.shape(y_pred))
        return super(SparseCategoricalMeanIoU, self).update_state(y_true, y_pred, sample_weight)
