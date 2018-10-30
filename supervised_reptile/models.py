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
        out = tf.layers.conv2d(x, 64, 3, strides=2, padding='same')
        out = tf.layers.batch_normalization(out, training=True)
        out = tf.nn.relu(out)
        return out

    def linearModule(self, x, **optim_kwargs):
        logits = tf.layers.dense(x, self.num_classes)
        label_ph = tf.placeholder(tf.int32, shape=(None,))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph,
                                                              logits=logits)
        predictions = tf.argmax(logits, axis=-1)
        minimize_op = self.optimizer(**optim_kwargs).minimize(loss)
        return logits, label_ph, loss, predictions, minimize_op

    def convToConvAdapter(self, x, laterals):
        if len(laterals) == 0:
            return x
        scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
        lateral = tf.concat(scaled_laterals, 1)
        lateral = tf.layers.conv2d(lateral, 64, 3, strides=2, padding='same')
        lateral = tf.nn.relu(lateral)
        out = tf.concat([x, lateral], 1)
        return out

    def convToLinearAdapter(self, x, laterals):
        x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
        if len(laterals) == 0:
            return x
        scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
        lateral = tf.concat(scaled_laterals, 1)
        lateral = tf.layers.conv2d(lateral, 64, 3, strides=2, padding='same')
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
        out = tf.layers.conv2d(x, 32, 3, padding='same')
        out = tf.layers.batch_normalization(out, training=True)
        out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
        out = tf.nn.relu(out)
        return out

    def linearModule(self, x, **optim_kwargs):
        logits = tf.layers.dense(x, self.num_classes)
        label_ph = tf.placeholder(tf.int32, shape=(None,))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph,
                                                              logits=logits)
        predictions = tf.argmax(logits, axis=-1)
        minimize_op = self.optimizer(**optim_kwargs).minimize(loss)
        return logits, label_ph, loss, predictions, minimize_op

    def convToConvAdapter(self, x, laterals):
        if len(laterals) == 0:
            return x
        scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
        lateral = tf.concat(scaled_laterals, 1)
        lateral = tf.layers.conv2d(lateral, 32, 3, padding='same')
        lateral = tf.nn.relu(lateral)
        out = tf.concat([x, lateral], 1)
        return out

    def convToLinearAdapter(self, x, laterals):
        x = tf.reshape(x, (-1, int(np.prod(x.get_shape()[1:]))))
        if len(laterals) == 0:
            return x
        scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
        lateral = tf.concat(scaled_laterals, 1)
        lateral = tf.layers.conv2d(lateral, 32, 3, padding='same')
        lateral = tf.nn.relu(lateral)
        lateral = tf.reshape(lateral, (-1, int(np.prod(lateral.get_shape()[1:]))))
        out = tf.concat([x, lateral], 1)
        return out


# pylint: disable=R0903
class ProgressiveOmniglotModel:
    """
    Progressive n-column model for Omniglot. Check https://arxiv.org/abs/1606.04671 for more details.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        self.input = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        self.column0 = ProgressiveOmniglotColumn(self.input, num_classes, optimizer, **optim_kwargs)
        self.logits = self.column0.logits
        self.label_ph = self.column0.label_ph
        self.loss = self.column0.loss
        self.predictions = self.column0.predictions
        self.minimize_op = self.column0.minimize_op


# pylint: disable=R0903
class ProgressiveMiniImageNetModel:
    """
    Progressive n-column model for Mini-ImageNet. Check https://arxiv.org/abs/1606.04671 for more details.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3)) 
        self.input = self.input_ph
        self.column0 = ProgressiveMiniImageNetColumn(self.input, num_classes, optimizer, **optim_kwargs)
        self.logits = self.column0.logits
        self.label_ph = self.column0.label_ph
        self.loss = self.column0.loss
        self.predictions = self.column0.predictions
        self.minimize_op = self.column0.minimize_op

