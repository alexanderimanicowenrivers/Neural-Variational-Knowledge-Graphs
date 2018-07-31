# -*- coding: utf-8 -*-

import tensorflow as tf
import vkge

#
import logging

logger = logging.getLogger(__name__)

flags = tf.app.flags
flags.DEFINE_float("mean_c", 6, "Constant for mean embeddings initalisation [0.1]")
flags.DEFINE_float("init_sig", 6.0, "Initalised variance of variables [6.0]")
flags.DEFINE_float("epsilon", 1e-5, "Initalised epsilon of variables [1e-5]")
flags.DEFINE_float("mog_split", 0.5, "Split between spike and slab [6.0]")
flags.DEFINE_integer("embedding_size", 200, "The dimension of graph embeddings [50]")
flags.DEFINE_string("file_name", '/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/18_6_25', "file name for tensorboard file ['--']")
flags.DEFINE_integer("no_batches", 10, "Number of batches [10]")
flags.DEFINE_boolean("alt_updates", True, "Switch between alternating updates in training [True]")
flags.DEFINE_boolean("alt_cost", True, "Switch for compression cost to be used in training [True]")
flags.DEFINE_boolean("static_mean", False,
                     "Switch as to if mean is fixed at 0 or can train with random normal init [False]")
flags.DEFINE_string("dataset", 'wn18', "Alternate updates around each distribution[wn18]")
flags.DEFINE_boolean("Sigma_alt", True, "Alternate between two different standard dev calculations [True]")
flags.DEFINE_boolean("projection", False, "Alternate between using a projection on the means [False]")
flags.DEFINE_boolean("alt_opt", True, "Define the use of hinge loss or MLE True is MLE  [True]")
flags.DEFINE_string("score_func", 'DistMult', "Defines score function [dismult]")
flags.DEFINE_string("alt_test", 'none', "Defines the alternative test type, t1,t2,t3 [None]")
flags.DEFINE_float("lr", 0.1, "Choose optimiser loss, select the margin for hinge loss [1]")
flags.DEFINE_integer("negsamples", 0, "Number of negative samples [0]")
flags.DEFINE_float("samples_perdp", 1, "Number of noise samples used per datapoint [1]")
flags.DEFINE_float("p_threshold", 0.1, "Confidence threshold for experiments 2&3 [1]")
FLAGS = flags.FLAGS


def main(_):
    logger.warn("creating model with flags \t {}".format(flags.FLAGS.__flags))

    # vkge.VKGE(embedding_size=FLAGS.embedding_size, mean_c=FLAGS.mean_c,init_sig=FLAGS.init_sig, alt_cost=FLAGS.alt_cost, no_batches=FLAGS.no_batches,
    #           static_mean=FLAGS.static_mean, dataset=FLAGS.dataset, sigma_alt=FLAGS.Sigma_alt,
    #           lr=FLAGS.lr, file_name=FLAGS.file_name, alt_opt=FLAGS.alt_opt
    #           , projection=FLAGS.projection,  score_func=FLAGS.score_func)

    vkge.VKGE(embedding_size=FLAGS.embedding_size, mean_c=FLAGS.mean_c,epsilon=FLAGS.epsilon, alt_cost=FLAGS.alt_cost, no_batches=FLAGS.no_batches,
              static_mean=FLAGS.static_mean, dataset=FLAGS.dataset, sigma_alt=FLAGS.Sigma_alt,negsamples=FLAGS.negsamples,
              lr=FLAGS.lr, p_threshold=FLAGS.p_threshold,file_name=FLAGS.file_name, alt_opt=FLAGS.alt_opt, projection=FLAGS.projection,alt_test=FLAGS.alt_test,  score_func=FLAGS.score_func,alt_updates=FLAGS.alt_updates,nosamps=FLAGS.samples_perdp)


if __name__ == '__main__':
    tf.app.run()