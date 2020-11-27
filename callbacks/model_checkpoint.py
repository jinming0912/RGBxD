import os

import tensorflow as tf


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 **kwargs):
        super(ModelCheckpoint, self).__init__(filepath,
                                              monitor=monitor,
                                              verbose=verbose,
                                              save_best_only=save_best_only,
                                              save_weights_only=save_weights_only,
                                              mode=mode,
                                              save_freq=save_freq,
                                              **kwargs)
        self.last_filepath = None

    def _save_model(self, epoch, logs):
        """Saves the model.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}
        if self.monitor not in logs.keys():
            return

        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    tf.compat.v1.logging.warning('Can save best model only with %s available, '
                                                 'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))

                        # Delete last file
                        if self.last_filepath is not None:
                            if os.path.exists(self.last_filepath):
                                os.remove(self.last_filepath)
                        self.last_filepath = filepath

                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

            self._maybe_remove_file()

