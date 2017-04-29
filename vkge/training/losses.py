# -*- coding: utf-8 -*-

import tensorflow as tf
import sys


def logistic_loss(scores, targets):
    """
    Logistic loss as used in [1]

    [1] http://jmlr.org/proceedings/papers/v48/trouillon16.pdf

    :param scores: (N,) Tensor containing scores of examples.
    :param targets: (N,) Tensor containing {0, 1} targets of examples.
    :return: Loss value.
    """
    logistic_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=targets)
    loss = tf.reduce_sum(logistic_losses)
    return loss


def hinge_loss(scores, targets, margin=1):
    """
    Hinge loss.
    :param scores: (N,) Tensor containing scores of examples.
    :param targets: (N,) Tensor containing {0, 1} targets of examples.
    :param margin: float representing the margin in the hinge loss relu(margin - logits * (2 * targets - 1))
    :return: Loss value.
    """
    hinge_losses = tf.nn.relu(margin - scores * (2 * targets - 1))

    loss = tf.reduce_sum(hinge_losses)
    return loss


# Aliases
logistic = logistic_loss
hinge = hinge_loss


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown loss function: {}'.format(function_name))
    return getattr(this_module, function_name)
