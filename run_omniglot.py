"""
Train a model on Omniglot.
"""

import random

import neptune
import tensorflow as tf

from supervised_reptile.args import model_kwargs, train_kwargs, evaluate_kwargs, neptune_args
from supervised_reptile.eval import evaluate
from supervised_reptile.models import ProgressiveOmniglotModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train import train

def main():
    """
    Load data and train a model on it.
    """
    context = neptune.Context()
    context.integrate_with_tensorflow()
    final_train_channel = context.create_channel('final_train_accuracy', neptune.ChannelType.NUMERIC)
    final_test_channel = context.create_channel('final_test_accuracy', neptune.ChannelType.NUMERIC)
    args = neptune_args(context)
    print('args:\n', args)
    random.seed(args.seed)

    train_set, test_set = split_dataset(read_dataset(args.omniglot_src))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = ProgressiveOmniglotModel(args.classes, **model_kwargs(args))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = args.allow_growth
    with tf.Session(config=config) as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        final_train_accuracy = evaluate(sess, model, train_set, **eval_kwargs)
        final_test_accuracy = evaluate(sess, model, test_set, **eval_kwargs)
        print('final_train_accuracy:', final_train_accuracy)
        print('final_test_accuracy:', final_test_accuracy)
        final_train_channel.send(final_train_accuracy)
        final_test_channel.send(final_test_accuracy)

if __name__ == '__main__':
    main()
