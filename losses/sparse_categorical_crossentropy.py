import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import backend as K


class SparseCategoricalCrossentropy(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, class_weight=None, name=None):
        self.class_weight = np.array(class_weight)[:, np.newaxis] if class_weight else None
        super(SparseCategoricalCrossentropy, self).__init__(name=name)

    def call(self, y_true, y_pred):
        sample_weight = tf.gather_nd(self.class_weight, tf.cast(y_true, tf.int32))
        losses = super(SparseCategoricalCrossentropy, self).call(y_true, y_pred)
        return losses_utils.compute_weighted_loss(
            losses, sample_weight, reduction=self._get_reduction())
