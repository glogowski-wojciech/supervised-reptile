"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)


class ProgressiveOmniglotColumn:
    def __init__(self, x, num_classes, optimizer=DEFAULT_OPTIMIZER, laterals_in=None, **optim_kwargs):
        self.NUM_LAYERS = 5
        self.outputs = {-1: x}
        self.num_classes = num_classes
        self.optimizer = optimizer

        if laterals_in is None:
            laterals_in = {0: [], 1: [], 2: [], 3: []}
        out0 = self.convModule(x)
        x = self.convToConvAdapter(out0, laterals_in[0])
        out1 = self.convModule(x)
        x = self.convToConvAdapter(out1, laterals_in[1])
        out2 = self.convModule(x)
        x = self.convToConvAdapter(out2, laterals_in[2])
        out3 = self.convModule(x)
        x = self.convToLinearAdapter(out3, laterals_in[3])
        self.logits, self.label_ph, self.loss, self.predictions, self.minimize_op = \
            self.linearModule(x, **optim_kwargs)

        self.laterals = {
            0: out0,
            1: out1,
            2: out2,
            3: out3,
        }

    def convModule(self, x):
        with tf.name_scope('ConvModule'):
            out = tf.layers.conv2d(x, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
            return out

    def linearModule(self, x, **optim_kwargs):
        with tf.name_scope('LinearModule'):
            logits = tf.layers.dense(x, self.num_classes)
            label_ph = tf.placeholder(tf.int32, shape=(None,))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph,
                                                                  logits=logits)
            predictions = tf.argmax(logits, axis=-1)
        if self.optimizer is not None:
            minimize_op = self.optimizer(**optim_kwargs).minimize(loss)
        else:
            minimize_op = None
        return logits, label_ph, loss, predictions, minimize_op

    def convToConvAdapter(self, x, laterals):
        with tf.name_scope('ConvToConvAdapter'):
            if len(laterals) == 0:
                return x
            scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
            lateral = tf.concat(scaled_laterals, 3)
            lateral = tf.layers.conv2d(lateral, 64, 1, padding='same')
            lateral = tf.nn.relu(lateral)
            out = tf.concat([x, lateral], 3)
            return out

    def convToLinearAdapter(self, x, laterals):
        with tf.name_scope('ConvToLinearAdapter'):
            x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
            if len(laterals) == 0:
                return x
            scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
            lateral = tf.concat(scaled_laterals, 1)
            lateral = tf.layers.conv2d(lateral, 64, 1, padding='same')
            lateral = tf.nn.relu(lateral)
            lateral = tf.reshape(lateral, (-1, int(np.prod(lateral.get_shape()[1:]))))
            out = tf.concat([x, lateral], 1)
            return out


class ProgressiveMiniImageNetColumn:
    def __init__(self, x, num_classes, optimizer=DEFAULT_OPTIMIZER, laterals_in=None, **optim_kwargs):
        self.NUM_LAYERS = 5
        self.outputs = {-1: x}
        self.num_classes = num_classes
        self.optimizer = optimizer

        if laterals_in is None:
            laterals_in = {0: [], 1: [], 2: [], 3: []}
        out0 = self.convModule(x)
        x = self.convToConvAdapter(out0, laterals_in[0])
        out1 = self.convModule(x)
        x = self.convToConvAdapter(out1, laterals_in[1])
        out2 = self.convModule(x)
        x = self.convToConvAdapter(out2, laterals_in[2])
        out3 = self.convModule(x)
        x = self.convToLinearAdapter(out3, laterals_in[3])
        self.logits, self.label_ph, self.loss, self.predictions, self.minimize_op = \
            self.linearModule(x, **optim_kwargs)

        self.laterals = {
            0: out0,
            1: out1,
            2: out2,
            3: out3,
        }

    def convModule(self, x):
        with tf.name_scope('ConvModule'):
            out = tf.layers.conv2d(x, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
            return out

    def linearModule(self, x, **optim_kwargs):
        with tf.name_scope('LinearModule'):
            logits = tf.layers.dense(x, self.num_classes)
            label_ph = tf.placeholder(tf.int32, shape=(None,))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph,
                                                                  logits=logits)
            predictions = tf.argmax(logits, axis=-1)
        if self.optimizer is not None:
            minimize_op = self.optimizer(**optim_kwargs).minimize(loss)
        else:
            minimize_op = None
        return logits, label_ph, loss, predictions, minimize_op

    def convToConvAdapter(self, x, laterals):
        with tf.name_scope('ConvToConvAdapter'):
            if len(laterals) == 0:
                return x
            scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
            lateral = tf.concat(scaled_laterals, 3)
            lateral = tf.layers.conv2d(lateral, 32, 1, padding='same')
            lateral = tf.nn.relu(lateral)
            out = tf.concat([x, lateral], 3)
            return out

    def convToLinearAdapter(self, x, laterals):
        with tf.name_scope('ConvToLinearAdapter'):
            x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
            if len(laterals) == 0:
                return x
            scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
            lateral = tf.concat(scaled_laterals, 1)
            lateral = tf.layers.conv2d(lateral, 32, 1, padding='same')
            lateral = tf.nn.relu(lateral)
            lateral = tf.reshape(lateral, (-1, int(np.prod(lateral.get_shape()[1:]))))
            out = tf.concat([x, lateral], 1)
            return out


def merge_laterals(laterals_list):
    out = {}
    keys = set()
    for laterals in laterals_list:
        keys.update(laterals.keys())
    for key in keys:
        out[key] = []
    for laterals in laterals_list:
        for key, val in laterals.items():
            out[key].append(val)
    return out


# pylint: disable=R0903
class ProgressiveOmniglotModel:
    """
    Progressive n-column model for Omniglot. Check https://arxiv.org/abs/1606.04671 for more details.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        self.input = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        with tf.name_scope('column0'):
            self.column0 = ProgressiveOmniglotColumn(self.input, num_classes, optimizer=None, **optim_kwargs)
        laterals1 = merge_laterals([self.column0.laterals])
        with tf.name_scope('column1'):
            self.column1 = ProgressiveOmniglotColumn(
                self.input, num_classes, optimizer, laterals1, **optim_kwargs
            )
        self.logits = self.column1.logits
        self.label_ph = self.column1.label_ph
        self.loss = self.column1.loss
        self.predictions = self.column1.predictions
        self.minimize_op = self.column1.minimize_op


# pylint: disable=R0903
class ProgressiveMiniImageNetModel:
    """
    Progressive n-column model for Mini-ImageNet. Check https://arxiv.org/abs/1606.04671 for more details.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3)) 
        self.input = self.input_ph
        with tf.name_scope('column0'):
            self.column0 = ProgressiveMiniImageNetColumn(self.input, num_classes, optimizer=None, **optim_kwargs)
        laterals1 = merge_laterals([self.column0.laterals])
        with tf.name_scope('column1'):
            self.column1 = ProgressiveMiniImageNetColumn(
                self.input, num_classes, optimizer, laterals1, **optim_kwargs
            )
        self.logits = self.column1.logits
        self.label_ph = self.column1.label_ph
        self.loss = self.column1.loss
        self.predictions = self.column1.predictions
        self.minimize_op = self.column1.minimize_op

