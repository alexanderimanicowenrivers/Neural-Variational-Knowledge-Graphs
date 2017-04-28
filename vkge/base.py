# -*- coding: utf-8 -*-

import tensorflow as tf
import vkge.models as models

import logging

logger = logging.getLogger(__name__)


class VKGE:
    def __init__(self,
                 nb_entities, entity_embedding_size,
                 nb_predicates, predicate_embedding_size,
                 optimizer):
        super().__init__()
        self.build_model(nb_entities, entity_embedding_size,
                         nb_predicates, predicate_embedding_size,
                         optimizer)

    @staticmethod
    def input_parameters(inputs, parameters_layer):
        parameters = tf.nn.embedding_lookup(parameters_layer, inputs)
        mu, log_sigma_square = tf.split(value=parameters, num_or_size_splits=2, axis=1)
        return mu, log_sigma_square

    @staticmethod
    def sample_embedding(mu, log_sigma_square):
        sigma = tf.sqrt(tf.exp(log_sigma_square))
        embedding_size = mu.get_shape()[1].value
        eps = tf.random_normal((1, embedding_size), 0, 1, dtype=tf.float32)
        return mu + sigma * eps

    def build_model(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, optimizer):
        self.s_inputs = tf.placeholder(tf.int32, shape=[None])
        self.p_inputs = tf.placeholder(tf.int32, shape=[None])
        self.o_inputs = tf.placeholder(tf.int32, shape=[None])
        self.y_inputs = tf.placeholder(tf.bool, shape=[None])

        self.build_encoder(nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size)
        self.build_decoder()

        # Kullback Leibler divergence
        self.e_objective = 0.0
        self.e_objective += 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_s - tf.square(self.mu_s) - tf.exp(self.log_sigma_sq_s))
        self.e_objective += 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_p - tf.square(self.mu_p) - tf.exp(self.log_sigma_sq_p))
        self.e_objective += 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_o - tf.square(self.mu_o) - tf.exp(self.log_sigma_sq_o))

        # Log likelihood
        self.g_objective = tf.reduce_sum(tf.log(tf.where(condition=self.y_inputs, x=self.p_x_i, y=1 - self.p_x_i) + 1e-4))
        self.elbo = tf.reduce_mean(self.g_objective - self.e_objective)

        self.training_step = optimizer.minimize(- self.elbo)

    def build_encoder(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size):
        logger.info('Building Inference Networks q(h_x | x) ..')
        with tf.variable_scope('encoder'):
            entity_parameters_layer = tf.get_variable('entities',
                                                      shape=[nb_entities + 1, entity_embedding_size * 2],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            predicate_parameters_layer = tf.get_variable('predicates',
                                                         shape=[nb_predicates + 1, predicate_embedding_size * 2],
                                                        initializer=tf.contrib.layers.xavier_initializer())

            self.mu_s, self.log_sigma_sq_s = VKGE.input_parameters(self.s_inputs, entity_parameters_layer)
            self.mu_p, self.log_sigma_sq_p = VKGE.input_parameters(self.p_inputs, predicate_parameters_layer)
            self.mu_o, self.log_sigma_sq_o = VKGE.input_parameters(self.o_inputs, entity_parameters_layer)

            self.h_s = VKGE.sample_embedding(self.mu_s, self.log_sigma_sq_s)
            self.h_p = VKGE.sample_embedding(self.mu_p, self.log_sigma_sq_p)
            self.h_o = VKGE.sample_embedding(self.mu_o, self.log_sigma_sq_o)

    def build_decoder(self):
        logger.info('Building Inference Network p(y|h) ..')
        with tf.variable_scope('decoder'):
            model = models.BilinearDiagonalModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p, object_embeddings=self.h_o)
            self.p_x_i = tf.sigmoid(model())

    def train(self, session, Xs, Xp, Xo, y):
        feed_dict = {
            self.s_inputs: Xs,
            self.p_inputs: Xp,
            self.o_inputs: Xo,
            self.y_inputs: y
        }
        _, elbo_value = session.run([self.training_step, self.elbo], feed_dict=feed_dict)
        return elbo_value
