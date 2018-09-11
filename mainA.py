# -*- coding: utf-8 -*-

import tensorflow as tf
import vkge
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
#
import logging

logger = logging.getLogger(__name__)

flags = tf.app.flags
flags.DEFINE_float("mean_c", 6, "Constant for mean embeddings initalisation [0.1]")
flags.DEFINE_float("init_sig", 6.0, "Initalised variance of variables [6.0]")
flags.DEFINE_float("epsilon", 1e-5, "Initalised epsilon of variables [1e-5]")
flags.DEFINE_float("mog_split", 0.5, "Split between spike and slab [6.0]")
flags.DEFINE_integer("re_reun", 1, "used for multiple runs of the same parameter settings")
flags.DEFINE_integer("embedding_size", 200, "The dimension of graph embeddings [50]")
flags.DEFINE_string("file_name", '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/18_6_25', "file name for tensorboard file ['--']")
flags.DEFINE_integer("no_batches", 100, "Number of batches [10]")
flags.DEFINE_boolean("alt_updates", True, "Switch between alternating updates in training [True]")
flags.DEFINE_boolean("alt_cost", True, "Switch for compression cost to be used in training [True]")
flags.DEFINE_boolean("static_mean", False,
                     "Switch as to if mean is fixed at 0 or can train with random normal init [False]")
flags.DEFINE_string("dataset", 'wn18', "Alternate updates around each distribution[wn18]")
flags.DEFINE_boolean("Sigma_alt", True, "Alternate between two different standard dev calculations [True]")
flags.DEFINE_boolean("projection", False, "Alternate between using a projection on the means [False]")
flags.DEFINE_boolean("alt_opt", True, "Define the use of unit prior or alternative  [True]")
flags.DEFINE_string("score_func", 'DistMult', "Defines score function [dismult]")
flags.DEFINE_string("distribution", 'normal', "Defines the distribution, either normal or von mises")
flags.DEFINE_float("lr", 0.01, "Choose optimiser loss, select the margin for hinge loss [1]")
flags.DEFINE_integer("negsamples", 0, "Number of negative samples [0]")
flags.DEFINE_float("ablation", 1, "Number of noise samples used per datapoint [1]")
FLAGS = flags.FLAGS


def main(_):
    logger.warn("creating model with flags \t {}".format(flags.FLAGS.__flags))


    vkge.modelA(embedding_size=FLAGS.embedding_size,distribution=FLAGS.distribution, epsilon=FLAGS.epsilon, no_batches=FLAGS.no_batches,dataset=FLAGS.dataset, negsamples=FLAGS.negsamples,
              lr=FLAGS.lr, file_name=FLAGS.file_name, alt_opt=FLAGS.alt_opt, projection=FLAGS.projection, score_func=FLAGS.score_func,ablation=FLAGS.ablation)


if __name__ == '__main__':
    tf.app.run()