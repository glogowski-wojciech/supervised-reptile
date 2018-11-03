"""
Helpers for evaluating models.
"""

from .reptile import Reptile
from .variables import weight_decay

# pylint: disable=R0913,R0914
def evaluate(sess,
             model,
             dataset,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             weight_decay_rate=1,
             reptile_fn=Reptile):
    """
    Evaluate a model on a dataset.
    """
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    total_correct0 = 0
    total_correct1 = 0
    for _ in range(num_samples):
        correct0, correct1 = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                          model.minimize_op, model,
                                          num_classes=num_classes, num_shots=num_shots,
                                          inner_batch_size=eval_inner_batch_size,
                                          inner_iters=eval_inner_iters, replacement=replacement)
        total_correct0 += correct0
        total_correct1 += correct1
    accuracy0 = total_correct0 / (num_samples * num_classes)
    accuracy1 = total_correct1 / (num_samples * num_classes)
    return accuracy0, accuracy1
