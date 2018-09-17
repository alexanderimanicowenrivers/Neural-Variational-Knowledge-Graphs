# -*- coding: utf-8 -*-

import tensorflow as tf
import vkge
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
#
import logging

logger = logging.getLogger(__name__)

flags = tf.app.flags
flags.DEFINE_float("epsilon", 1e-5, "Initalised epsilon of variables [1e-5]")
flags.DEFINE_integer("embedding_size", 200, "The dimension of graph embeddings [50]")
flags.DEFINE_string("file_name", '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/18_6_25', "file name for tensorboard file ['--']")
flags.DEFINE_integer("no_batches", 100, "Number of batches [10]")
flags.DEFINE_string("dataset", 'wn18', "Alternate updates around each distribution[wn18]")
flags.DEFINE_boolean("projection", False, "Alternate between using a projection on the means [False]")
flags.DEFINE_boolean("alt_prior", True, "Define the use of unit prior or alternative  [True]")
flags.DEFINE_string("score_func", 'DistMult', "Defines score function [dismult]")
flags.DEFINE_string("distribution", 'normal', "Defines the distribution, either normal or von mises")
flags.DEFINE_float("lr", 0.01, "Choose optimiser loss, select the margin for hinge loss [1]")
flags.DEFINE_integer("negsamples", 0, "Number of negative samples [0]")
FLAGS = flags.FLAGS


def main(_):
    logger.warn('creating model with flags \t {0}'.format(flags.FLAGS.__flags))


    vkge.modelB(embedding_size=FLAGS.embedding_size,distribution=FLAGS.distribution, epsilon=FLAGS.epsilon, no_batches=FLAGS.no_batches,dataset=FLAGS.dataset, negsamples=FLAGS.negsamples,
              lr=FLAGS.lr, file_name=FLAGS.file_name, alt_prior=FLAGS.alt_prior, projection=FLAGS.projection, score_func=FLAGS.score_func)


if __name__ == '__main__':
    tf.app.run()