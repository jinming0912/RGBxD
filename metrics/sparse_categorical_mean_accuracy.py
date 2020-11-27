import numpy as np
import tensorflow as tf


class SparseCategoricalMeanAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, class_weight=None, name='sparse_categorical_mean_accuracy', dtype=None):
        super(SparseCategoricalMeanAccuracy, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.class_weight = tf.convert_to_tensor(np.array(class_weight)[:, np.newaxis]) if class_weight else None
        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=tf.initializers.zeros,
            dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
        if self.class_weight is not None and sample_weight is None:
            sample_weight = tf.gather_nd(self.class_weight, tf.cast(y_true, tf.int32))
            sample_weight = tf.reshape(sample_weight, tf.shape(y_pred))

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None and sample_weight.shape.ndims > 1:
            sample_weight = tf.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=tf.float64)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.diag_part(self.total_cm), dtype=self._dtype)

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(sum_over_col, 0), dtype=self._dtype))

        acc = tf.math.divide_no_nan(true_positives, sum_over_col)

        return tf.math.divide_no_nan(
            tf.reduce_sum(acc, name='mean_acc'), num_valid_entries)

    def reset_states(self):
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(SparseCategoricalMeanAccuracy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))