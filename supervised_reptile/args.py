"""
Command-line argument parsing.
"""

import argparse
from functools import partial

import tensorflow as tf

from .reptile import Reptile, FOML

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='model_checkpoint')
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=5, type=int)
    parser.add_argument('--train-shots', help='shots in a training batch', default=0, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=5, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=20, type=int)
    parser.add_argument('--replacement', help='sample with replacement', action='store_true')
    parser.add_argument('--learning-rate', help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta-step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta-step-final', help='meta-training step size by the end',
                        default=0.1, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval-batch', help='eval inner batch size', default=5, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=50, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=10000, type=int)
    parser.add_argument('--eval-interval', help='train steps per eval', default=10, type=int)
    parser.add_argument('--weight-decay', help='weight decay rate', default=1, type=float)
    parser.add_argument('--transductive', help='evaluate all samples at once', action='store_true')
    parser.add_argument('--foml', help='use FOML instead of Reptile', action='store_true')
    parser.add_argument('--foml-tail', help='number of shots for the final mini-batch in FOML',
                        default=None, type=int)
    parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true')
    parser.add_argument('--allow-growth', help='allow_growth gpu option', action='store_true')
    parser.add_argument('--mode', help='mode for reproducing results', default='', type=str)
    return parser

def model_kwargs(parsed_args):
    """
    Build the kwargs for model constructors from the
    parsed command-line arguments.
    """
    res = {'learning_rate': parsed_args.learning_rate}
    if parsed_args.sgd:
        res['optimizer'] = tf.train.GradientDescentOptimizer
    return res

def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args)
    }

def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'weight_decay_rate': parsed_args.weight_decay,
        'num_samples': parsed_args.eval_samples,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args)
    }

def default_args():
    return {
        'pretrained': False,
        'seed': 0,
        'checkpoint': 'model_checkpoint',
        'classes': 5,
        'shots': 5,
        'train_shots': 0,
        'inner_batch': 5,
        'inner_iters': 20,
        'replacement': False,
        'learning_rate': 1e-3,
        'meta_step': 0.1,
        'meta_step_final': 0.1,
        'meta_batch': 1,
        'meta_iters': 400000,
        'eval_batch': 5,
        'eval_iters': 50,
        'eval_samples': 10000,
        'eval_interval': 10,
        'weight_decay': 1,
        'transductive': False,
        'foml': False,
        'foml_tail': None,
        'sgd': False,
        'allow_growth': False,
    }

def create_omniglot_mode(shots, classes, transductive):
    assert shots in [1, 5]
    assert classes in [5, 20]
    assert transductive in [False, True]
    name = 'o' + str(shots) + str(classes) + ('t' if transductive else '')
    cl5 = classes == 5
    return {
        'mode_name': name,
        'classes': classes,
        'shots': shots,
        'checkpoint': 'ckpt_' + name,
        'transductive': transductive,
        'train_shots': 10,
        'meta_step': 1.0,
        'meta_step_final': 0.0,
        'meta_batch': 5,
        'eval_iters': 50,
        'learning_rate': 0.001 if cl5 else 0.0005,
        'inner_batch': 10 if cl5 else 20,
        'inner_iters': 5 if cl5 else 10,
        'meta_iters': 100000 if cl5 else 200000,
        'eval_batch': 5 if cl5 else 10,
    }

def create_miniimagenet_mode(shots, transductive):
    assert shots in [1, 5]
    assert transductive in [False, True]
    name = 'm' + str(shots) + '5' + ('t' if transductive else '')
    sh1 = shots == 1
    return {
        'mode_name': name,
        'classes': 5,
        'shots': shots,
        'checkpoint': 'ckpt_' + name,
        'transductive': transductive,
        'learning_rate': 0.001,
        'inner_batch': 10,
        'inner_iters': 8,
        'train_shots': 15,
        'meta_step': 1.0,
        'meta_step_final': 0.0,
        'meta_iters': 100000,
        'meta_batch': 5,
        'eval_iters': 50,
        'eval_batch': 5 if sh1 else 15
    }


def reproduce(mode):
    args = default_args()
    modes = {}
    for shots in [1, 5]:
        for classes in [5, 20]:
            for transductive in [False, True]:
                m = create_omniglot_mode(shots, classes, transductive)
                modes[m['mode_name']] = m
    for shots in [1, 5]:
        for transductive in [False, True]:
            m = create_miniimagenet_mode(shots, transductive)
            modes[m['mode_name']] = m
    print(modes)
    assert mode in modes
    args.update(modes[mode])
    return args

def _args_reptile(parsed_args):
    if parsed_args.foml:
        return partial(FOML, tail_shots=parsed_args.foml_tail)
    return Reptile
