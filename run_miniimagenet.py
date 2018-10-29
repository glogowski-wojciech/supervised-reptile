"""
Train a model on miniImageNet.
"""

import random

import neptune
import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs, neptune_args
from supervised_reptile.eval import evaluate
from supervised_reptile.models import MiniImageNetModel
from supervised_reptile.miniimagenet import read_dataset
from supervised_reptile.train import train

def main():
    """
    Load data and train a model on it.
    """
    context = neptune.Context()
    context.integrate_with_tensorflow()
    final_train_channel = context.create_channel('final_train_accuracy', neptune.ChannelType.NUMERIC)
    final_val_channel = context.create_channel('final_val_accuracy', neptune.ChannelType.NUMERIC)
    final_test_channel = context.create_channel('final_test_accuracy', neptune.ChannelType.NUMERIC)
    args = neptune_args(context)
    print('args:', args)
    random.seed(args.seed)

    train_set, val_set, test_set = read_dataset(args.dataset)
    model = MiniImageNetModel(args.classes, **model_kwargs(args))

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
        final_val_accuracy = evaluate(sess, model, val_set, **eval_kwargs)
        final_test_accuracy = evaluate(sess, model, test_set, **eval_kwargs)
        print('final_train_accuracy:', final_train_accuracy)
        print('final_val_accuracy:', final_val_accuracy)
        print('final_test_accuracy:', final_test_accuracy)
        final_train_channel.send(final_train_accuracy)
        final_val_channel.send(final_val_accuracy)
        final_test_channel.send(final_test_accuracy)

if __name__ == '__main__':
    main()
