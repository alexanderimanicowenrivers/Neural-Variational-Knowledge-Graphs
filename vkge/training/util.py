# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
import vkge.models as models
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


def get_latent_distributions(distribution,mu_s,mu_p,mu_o,log_sigma_sq_s,log_sigma_sq_p,log_sigma_sq_o):
    """
                    Returns tf distributions for the generative network 
    """

    if distribution == 'normal':

        # sample from mean and std of the normal distribution

        q_s = tfd.MultivariateNormalDiag(mu_s, distribution_scale(log_sigma_sq_s))
        q_p = tfd.MultivariateNormalDiag(mu_p, distribution_scale(log_sigma_sq_p))
        q_o = tfd.MultivariateNormalDiag(mu_o, distribution_scale(log_sigma_sq_o))



    elif distribution == 'vmf':

        # sample from mean and concentration of the von Mises-Fisher

        # '+1' used to prevent collapsing behaviors

        q_s = VonMisesFisher(mu_s, distribution_scale(log_sigma_sq_s) + 1)
        q_p = VonMisesFisher(mu_p, distribution_scale(log_sigma_sq_p) + 1)
        q_o = VonMisesFisher(mu_o, distribution_scale(log_sigma_sq_o) + 1)



    else:
        raise NotImplemented

    return q_s,q_p,q_o



def get_scoring_func(score_func,distribution,h_s,h_p,h_o,mu_s,mu_p,mu_o):
    """
                    Returns scoring function for training and testing  
    """

    if score_func == 'DistMult':

        model = models.BilinearDiagonalModel(subject_embeddings=h_s, predicate_embeddings=h_p,
                                             object_embeddings=h_o)

        if distribution == 'normal':

            model_test = models.BilinearDiagonalModel(subject_embeddings=mu_s, predicate_embeddings=mu_p,
                                                      object_embeddings=mu_o)

        elif distribution == 'vmf':

            model_test = models.BilinearDiagonalModel(subject_embeddings=mu_s,
                                                      predicate_embeddings=mu_p,
                                                      object_embeddings=mu_o)

    elif score_func == 'ComplEx':
        model = models.ComplexModel(subject_embeddings=h_s, predicate_embeddings=h_p,
                                    object_embeddings=h_o)

        if distribution == 'normal':

            model_test = models.ComplexModel(subject_embeddings=mu_s, predicate_embeddings=mu_p,
                                             object_embeddings=mu_o)

        elif distribution == 'vmf':

            model_test = models.ComplexModel(subject_embeddings=tf.nn.l2_normalize(mu_s, axis=-1),
                                             predicate_embeddings=tf.nn.l2_normalize(mu_p, axis=-1),
                                             object_embeddings=tf.nn.l2_normalize(mu_o, axis=-1))

    elif score_func == 'TransE':
        model = models.TranslatingModel(subject_embeddings=h_s, predicate_embeddings=h_p,
                                        object_embeddings=h_o)
        if distribution == 'normal':
            model_test = models.TranslatingModel(subject_embeddings=mu_s, predicate_embeddings=mu_p,
                                                 object_embeddings=mu_o)

        elif distribution == 'vmf':
            model_test = models.TranslatingModel(subject_embeddings=tf.nn.l2_normalize(mu_s, axis=-1),
                                                 predicate_embeddings=tf.nn.l2_normalize(mu_p, axis=-1),
                                                 object_embeddings=tf.nn.l2_normalize(mu_o, axis=-1))
    else:
        raise NotImplemented

    return model,model_test




def make_compression_cost(nb_batches):
    """
                    Returns compression cost coefficient vector
    """

    M = int(nb_batches)

    pi_s = np.log(2.0) * (M - 1)
    pi_e = np.log(2.0)

    pi_t = np.exp(np.linspace(pi_s, pi_e, M) - M * np.log(2.0))

    pi = (1 / np.sum(pi_t)) * pi_t  # normalise pi

    return pi


 def stats(values):
    """
                                Return mean and variance statistics
    """

    return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))