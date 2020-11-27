import tensorflow as tf


class PolynomialDecay(tf.keras.optimizers.schedules.PolynomialDecay):
    """A LearningRateSchedule that uses a polynomial decay schedule."""

    def __init__(
            self,
            initial_learning_rate,
            decay_steps,
            end_learning_rate=0.0001,
            power=1.0,
            cycle=False,
            name=None):
        super(PolynomialDecay, self).__init__(initial_learning_rate=initial_learning_rate,
                                              decay_steps=decay_steps,
                                              end_learning_rate=end_learning_rate,
                                              power=power,
                                              cycle=cycle,
                                              name=name)
        self.current_lr = initial_learning_rate

    def __call__(self, step):
        self.current_lr = super(PolynomialDecay, self).__call__(step)
        return self.current_lr

