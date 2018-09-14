# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
tfd = tf.contrib.distributions

def read_triples(path):
    triples = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


class IndexGenerator:
    def __init__(self):
        self.random_state = np.random.RandomState(0)

    def __call__(self, n_samples, candidate_indices):
        shuffled_indices = candidate_indices[self.random_state.permutation(len(candidate_indices))]
        rand_ints = shuffled_indices[np.arange(n_samples) % len(shuffled_indices)]
        return rand_ints

def distribution_scale(log_sigma_square):
        """
                        Returns the scale (std dev) from embeddings for tensorflow distributions MultivariateNormalDiag function
                """

        scale = tf.sqrt(tf.exp(log_sigma_square))

        return scale

def make_prior(code_size,distribution,alt_prior):
        """
                        Returns the prior on embeddings for tensorflow distributions

                        (i) MultivariateNormalDiag function

                        (ii) HypersphericalUniform

                        with alternative prior on gaussian

                        (1) Alt: N(0,1/code_size)
                        (2) N(0,1)
        """

        if distribution == 'normal':
            if alt_prior: #alternative prior 0,1/embeddings variance
                loc = tf.zeros(code_size)
                scale = tf.sqrt(tf.divide(tf.ones(code_size),code_size))

            else:
                loc = tf.zeros(code_size)
                scale = tf.ones(code_size)

            dist=tfd.MultivariateNormalDiag(loc, scale)

        elif distribution == 'vmf':

            dist=HypersphericalUniform(code_size - 1, dtype=tf.float32)

        else:
            raise NotImplemented

        return dist