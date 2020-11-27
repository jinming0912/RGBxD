import logging
import tensorflow as tf


class LearningRateMetric(tf.keras.metrics.Metric):
    def __init__(self, optimizer, name=None, dtype=None, **kwargs):
        super(LearningRateMetric, self).__init__(name=name, dtype=dtype, **kwargs)
        self.optimizer = optimizer

    def update_state(self, *args, **kwargs):
        return None

    def result(self):
        lr = self.optimizer.lr
        if isinstance(lr, tf.Variable):
            return lr
        elif hasattr(lr, 'current_lr'):
            return lr.current_lr
        else:
            logging.error("No attr named \'current_lr\'")
            return None