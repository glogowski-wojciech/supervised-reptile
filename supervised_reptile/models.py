"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

NUM_LAYERS = 5
NUM_COLUMNS = 2


class ProgressiveOmniglotColumn:
    def __init__(self, x, num_classes, laterals_in=None, lateral_map=None):
        self.outputs = {-1: x}
        self.num_classes = num_classes

        if laterals_in is None:
            laterals_in = {}
            for i in range(NUM_LAYERS - 1):
                laterals_in[i] = []
        else:
            print('laterals_in', laterals_in)
            laterals_in = dict(laterals_in)
            print('laterals_in', laterals_in)
            assert len(lateral_map) == NUM_LAYERS - 1
            for i in range(NUM_LAYERS - 1):
                assert lateral_map[i] in ['o', 'x']
                if lateral_map[i] == 'o':
                    laterals_in[i] = []
        out0 = self.convModule(x)
        x = self.convToConvAdapter(out0, laterals_in[0])
        out1 = self.convModule(x)
        x = self.convToConvAdapter(out1, laterals_in[1])
        out2 = self.convModule(x)
        x = self.convToConvAdapter(out2, laterals_in[2])
        out3 = self.convModule(x)
        x = self.convToLinearAdapter(out3, laterals_in[3])
        self.logits, self.label_ph, self.loss, self.predictions = self.linearModule(x)

        self.laterals = {
            0: out0,
            1: out1,
            2: out2,
            3: out3,
        }

    def convModule(self, x):
        with tf.name_scope('ConvMod'):
            out = tf.layers.conv2d(x, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
            return out

    def linearModule(self, x):
        with tf.name_scope('LinMod'):
            logits = tf.layers.dense(x, self.num_classes)
            label_ph = tf.placeholder(tf.int32, shape=(None,))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph,
                                                                  logits=logits)
            predictions = tf.argmax(logits, axis=-1)
        return logits, label_ph, loss, predictions

    def convToConvAdapter(self, x, laterals):
        with tf.name_scope('C2CAdapt'):
            if len(laterals) == 0:
                return x
            scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
            lateral = tf.concat(scaled_laterals, 3)
            lateral = tf.layers.conv2d(lateral, 64, 1, padding='same')
            lateral = tf.nn.relu(lateral)
            out = tf.concat([x, lateral], 3)
            return out

    def convToLinearAdapter(self, x, laterals):
        with tf.name_scope('C2LAdapt'):
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
    def __init__(self, x, num_classes, laterals_in=None, lateral_map=None):
        laterals_in = dict(laterals_in)
        self.outputs = {-1: x}
        self.num_classes = num_classes

        if laterals_in is None:
            laterals_in = []
            for i in range(NUM_LAYERS - 1):
                laterals_in[i] = []
        else:
            assert len(lateral_map) == NUM_LAYERS - 1
            for i in range(NUM_LAYERS - 1):
                assert lateral_map[i] in ['o', 'x']
                if lateral_map[i] == 'o':
                    laterals_in[i] = []
        out0 = self.convModule(x)
        x = self.convToConvAdapter(out0, laterals_in[0])
        out1 = self.convModule(x)
        x = self.convToConvAdapter(out1, laterals_in[1])
        out2 = self.convModule(x)
        x = self.convToConvAdapter(out2, laterals_in[2])
        out3 = self.convModule(x)
        x = self.convToLinearAdapter(out3, laterals_in[3])
        self.logits, self.label_ph, self.loss, self.predictions = self.linearModule(x)

        self.laterals = {
            0: out0,
            1: out1,
            2: out2,
            3: out3,
        }

    def convModule(self, x):
        with tf.name_scope('ConvMod'):
            out = tf.layers.conv2d(x, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
            return out

    def linearModule(self, x):
        with tf.name_scope('LinMod'):
            logits = tf.layers.dense(x, self.num_classes)
            label_ph = tf.placeholder(tf.int32, shape=(None,))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph,
                                                                  logits=logits)
            predictions = tf.argmax(logits, axis=-1)
        return logits, label_ph, loss, predictions

    def convToConvAdapter(self, x, laterals):
        with tf.name_scope('C2CAdapt'):
            if len(laterals) == 0:
                return x
            scaled_laterals = [tf.Variable(1.0) * lateral for lateral in laterals]
            lateral = tf.concat(scaled_laterals, 3)
            lateral = tf.layers.conv2d(lateral, 32, 1, padding='same')
            lateral = tf.nn.relu(lateral)
            out = tf.concat([x, lateral], 3)
            return out

    def convToLinearAdapter(self, x, laterals):
        with tf.name_scope('C2LAdapt'):
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


def minimize_op(loss, optimizer, var_scope0, var_scope1, learning_rate0, learning_rate1, **optim_kwargs):
    with tf.name_scope('Opt0'):
        col0_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope0)
        minimize_op0 = optimizer(learning_rate0, **optim_kwargs).minimize(loss, var_list=col0_vars)
    with tf.name_scope('Opt1'):
        col1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope1)
        minimize_op1 = optimizer(learning_rate1, **optim_kwargs).minimize(loss, var_list=col1_vars)
    with tf.control_dependencies([minimize_op0, minimize_op1]):
        return tf.no_op()


# pylint: disable=R0903
class ProgressiveOmniglotModel:
    """
    Progressive n-column model for Omniglot. Check https://arxiv.org/abs/1606.04671 for more details.
    """
    def __init__(self, num_classes, lateral_map, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        print(type(lateral_map), lateral_map)
        assert len(lateral_map) == (NUM_LAYERS - 1) * NUM_COLUMNS * (NUM_COLUMNS - 1) / 2
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        self.input = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        with tf.name_scope('Net'):
            with tf.variable_scope('Col0Vars'):
                self.column0 = ProgressiveOmniglotColumn(self.input, num_classes)
            laterals1 = merge_laterals([self.column0.laterals])
            with tf.variable_scope('Col1Vars'):
                self.column1 = ProgressiveOmniglotColumn(
                    self.input, num_classes, laterals1, lateral_map[0:NUM_LAYERS - 1]
                )
        self.logits = self.column1.logits
        self.label_ph = self.column1.label_ph
        self.loss = self.column1.loss
        self.predictions = self.column1.predictions
        self.minimize_op = minimize_op(self.loss, optimizer, 'Col0Vars', 'Col1Vars', **optim_kwargs)


# pylint: disable=R0903
class ProgressiveMiniImageNetModel:
    """
    Progressive n-column model for Mini-ImageNet. Check https://arxiv.org/abs/1606.04671 for more details.
    """
    def __init__(self, num_classes, lateral_map, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        assert len(lateral_map) == (NUM_LAYERS - 1) * NUM_COLUMNS * (NUM_COLUMNS - 1) / 2
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3)) 
        self.input = self.input_ph
        with tf.name_scope('Net'):
            with tf.variable_scope('Col0Vars'):
                self.column0 = ProgressiveMiniImageNetColumn(self.input, num_classes)
            laterals1 = merge_laterals([self.column0.laterals])
            with tf.variable_scope('Col1Vars'):
                self.column1 = ProgressiveMiniImageNetColumn(
                    self.input, num_classes, laterals1, lateral_map[0:NUM_LAYERS - 1]
                )
        self.logits = self.column1.logits
        self.label_ph = self.column1.label_ph
        self.loss = self.column1.loss
        self.predictions = self.column1.predictions
        self.minimize_op = minimize_op(self.loss, optimizer, 'Col0Vars', 'Col1Vars', **optim_kwargs)

