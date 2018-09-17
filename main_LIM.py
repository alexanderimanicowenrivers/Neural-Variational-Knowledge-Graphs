# -*- coding: utf-8 -*-

import tensorflow as tf
import vkge
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
#
import logging

logger = logging.getLogger(__name__)

flags = tf.app.flags
flags.DEFINE_float("epsilon", 1e-7, "Adam optimiser epsilon decay rate [1e-7]")
flags.DEFINE_integer("embedding_size", 300, "The dimension of graph embeddings [300]")
flags.DEFINE_string("file_name", '~/', "file name for tensorboard file ['--']")
flags.DEFINE_integer("no_batches", 10, "Number of batches [10]")
flags.DEFINE_string("dataset", 'wn18', "Determines the Knowledge Base dataset [wn18]")
flags.DEFINE_boolean("projection", False, "Alternate between using a projection to constrain variance embedding to sum to one [False]")
flags.DEFINE_boolean("alt_prior", False, "Define the use of unit prior or alternative  [False]")
flags.DEFINE_string("score_func", 'DistMult', "Defines score function: DistMult, ComplEx or TransE [DistMult]")
flags.DEFINE_string("distribution", 'normal', "Defines the distribution, either 'normal' or 'vmf', which is von-Mises Fisher distribution")
flags.DEFINE_float("lr", 0.001, "Learning rate for optimiser [0.001]")
flags.DEFINE_integer("negsamples", 1, "Number of negative samples [1]")
FLAGS = flags.FLAGS


def main(_):

    vkge.LIM(embedding_size=FLAGS.embedding_size,distribution=FLAGS.distribution, epsilon=FLAGS.epsilon, no_batches=FLAGS.no_batches,dataset=FLAGS.dataset, negsamples=FLAGS.negsamples,
              lr=FLAGS.lr, file_name=FLAGS.file_name, alt_prior=FLAGS.alt_prior, projection=FLAGS.projection, score_func=FLAGS.score_func)


if __name__ == '__main__':
    tf.app.run()