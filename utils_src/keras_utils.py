"""
A collection of general purpose Keras utility procedures for machine-learning.

Author: Yuval Atzmon
"""

import warnings
import numpy as np
import os

from keras.callbacks import Callback
import keras
import keras.backend as K


def layer_name_to_ids(name, model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    return [i for i, w_name in enumerate(names) if name in w_name]

def slice_2d_tensor_to_list(T, ranges_ids, axis=1):
    groups_ids = [0] + ranges_ids.tolist() + [T.shape.as_list()[1]]
    groups_sizes = np.diff(groups_ids)
    split_list = []
    for n in range(len(groups_ids) - 1):
        # slc = slice(groups_ids[n], groups_ids[n + 1])

        if axis==1:
            sliced_shape = (groups_sizes[n],)
            split_list += [keras.layers.Lambda(
                lambda x: x[:, groups_ids[n]:groups_ids[n + 1]],
                output_shape=sliced_shape)(T)]
        else:
            raise(ValueError('Axis not supported: axis={}'.format(axis)))

    return split_list

def sparse_categorical_loss(y_true, y_pred):
    """Sparse categorical loss
    # Arguments
        y_true: An integer tensor (will be converted to one-hot).
        y_pred: A tensor resulting from y_pred
    # Returns
        Output tensor.
    """
    output_shape = y_pred.get_shape()
    num_classes = int(output_shape[1])
    # Represent as one-hot
    y_true = K.cast(K.flatten(y_true), 'int64')
    y_true = K.one_hot(y_true, num_classes)

    # Call loss function
    res = getattr(keras.losses, 'categorical_crossentropy')(y_true, y_pred)
    return res


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

# For debugging: Log to a text file the updates on every batch
class LogUpdatesCallback(Callback):
    def __init__(self, model, pathname, start_epoch=0, end_epoch=float('inf')):
        self.model = model
        self.pathname = pathname
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.pre_weights = None
        self.do_logging = False
        if start_epoch == 0:
            self.do_logging = True

    def on_epoch_begin(self, epoch, logs=None):
        if epoch>=self.start_epoch and epoch<self.end_epoch:
            self.do_logging = True
        else:
            self.do_logging = False

    def on_batch_begin(self, batch, logs=None):
        if self.do_logging:
            self.pre_weights = self.model.get_weights()

    def on_batch_end(self, batch, logs=None):

        if self.do_logging:
            current_weights = self.model.get_weights()
            for i, W in enumerate(current_weights):
                update = W - self.pre_weights[i]
                shape = 'x'.join([str(x) for x in W.shape])
                # Generate filename based on batch_number and layer
                fname = 'update_L%d_%s_batch_%d.csv'%(i, shape, batch)
                full_fname = os.path.join(self.pathname, fname )
                # Dump updates to file
                np.savetxt(full_fname, update, delimiter=",", fmt='%1.7f')
                if batch%10 == 0:
                    print('Batch %d: Saved updates under %s'%(batch, full_fname))





class EarlyBreaking(Callback):
    """Stop training when a monitored quantity is doesn't exceed
       a threshold for some time
    # Arguments
        monitor: quantity to be monitored.
        threshold: Threshold to exceed. Setting to None disables this callback
        patience: min number of epochs to wait before checking the threshold
        verbose: verbosity mode.
        mode: one of {min, max}. In `min` mode,
            training will stop when the quantity
            monitored is above the threshold; in `max`
            mode it will stop when the quantity
            monitored is below the threshold; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor, mode, threshold, patience):
        super(EarlyBreaking, self).__init__()

        self.monitor = monitor
        self.threshold = threshold
        self.patience = patience

        if mode not in ['min', 'max']:
            raise ValueError('EarlyBreaking mode %s is unknown, ' % mode)
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if self.threshold is None:
            return
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early breaking is disabled because it is conditioned on '
                'metric `%s` which is not available. Available metrics are: %s'
                % (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current, self.best):
            print(f'DEBUG Early breaking: best={self.best}')
            self.best = current
            self.wait = 0
        if epoch >= self.patience:
            if not self.monitor_op(self.best, self.threshold):
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(
                    'Early breaking! Best monitored value {}={} didn''t exceed'
                    ' threshold {} for the first {} epochs: '.format(
                        self.monitor, self.best, self.threshold, self.stopped_epoch))
