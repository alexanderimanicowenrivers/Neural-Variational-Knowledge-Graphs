# -*- coding: utf-8 -*-

import tensorflow as tf
import vkge
import logging

logger = logging.getLogger(__name__)

flags = tf.app.flags
flags.DEFINE_float("epsilon", 1e-7, "Adam optimiser epsilon decay rate [1e-7]")
flags.DEFINE_integer("embedding_size", 300, "The dimension of graph embeddings [300]")
flags.DEFINE_string("file_name", 'model', "file name for tensorboard file ['--']")
flags.DEFINE_integer("no_batches", 10, "Number of batches [10]")
flags.DEFINE_string("dataset", 'wn18rr', "Determines the Knowledge Base dataset [wn18]")
flags.DEFINE_boolean("projection", False, "Alternate between using a projection to constrain variance embedding to sum to one [False]")
flags.DEFINE_boolean("alt_prior", False, "Define the use of unit prior or alternative  [False]")
flags.DEFINE_string("score_func", 'Quaternator', "Defines score function: Quaternator")
flags.DEFINE_string("distribution", 'normal', "Defines the distribution, either 'normal' or 'vmf', which is von-Mises Fisher distribution")
flags.DEFINE_float("lr", 0.01, "Learning rate for optimiser [0.001]")
flags.DEFINE_integer("negsamples", 1, "Number of negative samples [1]")
FLAGS = flags.FLAGS



def main(_):
    logger.warning("creating model with flags \t {}".format(flags.FLAGS.__flags))
    vkge.Quaternator(embedding_size=FLAGS.embedding_size, no_batches=FLAGS.no_batches, dataset=FLAGS.dataset,
                  lr=FLAGS.lr,score_func=FLAGS.score_func,epsilon=FLAGS.epsilon,file_name=FLAGS.file_name)


if __name__ == '__main__':
    tf.app.run()