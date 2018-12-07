# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd =  tfp.distributions

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


def make_latent_variables(meaninit,siginit,nb_variables,embedding_size,distribution,vtype):
    """
                    Returns the mean and scale embedding matrix
    """
    
    emd_mean_name=vtype+'_mean'
    emd_sig_name=vtype+'_sigma'

    if distribution == 'vmf':
        #almost initialise scale to 0
        scale_max = np.log(1e-8) 

    else:
        scale_max = np.log((1.0 / embedding_size * 1.0) + 1e-10)

    sigmax = np.round(scale_max, decimals=2)

    if meaninit=='ru':

        embedding_mean = tf.get_variable(emd_mean_name,
                                                     shape=[nb_variables + 1, embedding_size],
                                                     initializer=tf.random_uniform_initializer(
                                                         minval=-0.001,
                                                         maxval=0.001,
                                                         dtype=tf.float32))

    else:

        embedding_mean = tf.get_variable(emd_mean_name, shape=[nb_variables + 1, embedding_size],
                                                     initializer=tf.contrib.layers.xavier_initializer())

    if siginit=='ru' and distribution == 'normal':


        embedding_sigma = tf.get_variable(emd_sig_name,
                                                     shape=[nb_variables + 1, embedding_size],
                                                     initializer=tf.random_uniform_initializer(
                                                         minval=0, maxval=sigmax, dtype=tf.float32),
                                                     dtype=tf.float32)

    if (siginit != 'ru') and distribution == 'normal':

        embedding_sigma = tf.get_variable(emd_sig_name,
                                                      shape=[nb_variables + 1, embedding_size],
                                                      initializer=tf.random_uniform_initializer(
                                                          minval=sigmax, maxval=sigmax, dtype=tf.float32),
                                                      dtype=tf.float32)

    if distribution == 'vmf':

        embedding_sigma = tf.get_variable(emd_sig_name,
                                                      shape=[nb_variables + 1, 1],
                                                      initializer=tf.random_uniform_initializer(
                                                          minval=sigmax, maxval=sigmax, dtype=tf.float32),
                                                      dtype=tf.float32)

    if distribution not in ['vmf','normal']:
        raise NotImplemented

    return embedding_mean,embedding_sigma

def stats(values):
    """
                                Return mean and variance statistics
    """

    return ('{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4)))