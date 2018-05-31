# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf

from vkge.knowledgebase import Fact, KnowledgeBaseParser

import vkge.models as models
from vkge.training import constraints, corrupt, index
from vkge.training.util import make_batches

# import logging

# logger = logging.getLogger(__name__)


class VKGE:
    def __init__(self, triples, entity_embedding_size, predicate_embedding_size,lr=0.001,b1=0.9,b2=0.999,eps=1e-08):
        super().__init__()

        print('Parsing the facts in the Knowledge Base ..')

        # logger.info('Parsing the facts in the Knowledge Base ..')
        self.facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
        self.parser = KnowledgeBaseParser(self.facts)

        self.random_state = np.random.RandomState(0)
        nb_entities, nb_predicates = len(self.parser.entity_vocabulary), len(self.parser.predicate_vocabulary)

        optimizer=tf.train.AdamOptimizer(learning_rate=lr,beta1=b1,beta2=b2,epsilon=eps)
        self.build_model(nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size,optimizer)

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

    def build_model(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size,optimizer):
        self.s_inputs = tf.placeholder(tf.int32, shape=[None])
        self.p_inputs = tf.placeholder(tf.int32, shape=[None])
        self.o_inputs = tf.placeholder(tf.int32, shape=[None])
        self.y_inputs = tf.placeholder(tf.bool, shape=[None])

        self.build_encoder(nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size)
        self.build_decoder()

        # Kullback Leibler divergence
        self.e_objective = 0.0
        self.e_objective -= 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_s - tf.square(self.mu_s) - tf.exp(self.log_sigma_sq_s))
        self.e_objective -= 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_p - tf.square(self.mu_p) - tf.exp(self.log_sigma_sq_p))
        self.e_objective -= 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_o - tf.square(self.mu_o) - tf.exp(self.log_sigma_sq_o))

        # Log likelihood
        self.g_objective = -tf.reduce_sum(tf.log(tf.where(condition=self.y_inputs, x=self.p_x_i, y=1 - self.p_x_i) + 1e-4))

        self.elbo = tf.reduce_mean(self.g_objective + self.e_objective)

        self.training_step = optimizer.minimize(self.elbo)

    def build_encoder(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size):
        print('Building Inference Networks q(h_x | x) ..')

        # logger.info('Building Inference Networks q(h_x | x) ..')
        with tf.variable_scope('encoder'):
            self.entity_parameters_layer = tf.get_variable('entities',
                                                           shape=[nb_entities + 1, entity_embedding_size * 2],
                                                           initializer=tf.contrib.layers.xavier_initializer())
            self.predicate_parameters_layer = tf.get_variable('predicates',
                                                              shape=[nb_predicates + 1, predicate_embedding_size * 2],
                                                              initializer=tf.contrib.layers.xavier_initializer())

            self.mu_s, self.log_sigma_sq_s = VKGE.input_parameters(self.s_inputs, self.entity_parameters_layer)
            self.mu_p, self.log_sigma_sq_p = VKGE.input_parameters(self.p_inputs, self.predicate_parameters_layer)
            self.mu_o, self.log_sigma_sq_o = VKGE.input_parameters(self.o_inputs, self.entity_parameters_layer)

            self.h_s = VKGE.sample_embedding(self.mu_s, self.log_sigma_sq_s)
            self.h_p = VKGE.sample_embedding(self.mu_p, self.log_sigma_sq_p)
            self.h_o = VKGE.sample_embedding(self.mu_o, self.log_sigma_sq_o)

    def build_decoder(self):
        print('Building Inference Network p(y|h) ..')
        # logger.info('Building Inference Network p(y|h) ..')
        with tf.variable_scope('decoder'):
            model = models.BilinearDiagonalModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p, object_embeddings=self.h_o)
            self.p_x_i = tf.sigmoid(model())

    def train(self, session, nb_batches=10, nb_epochs=10, unit_cube=False):
        index_gen = index.GlorotIndexGenerator()
        neg_idxs = np.array(sorted(set(self.parser.entity_to_index.values())))

        subj_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs,
                                                 corr_obj=False)
        obj_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs,
                                                corr_obj=True)

        train_sequences = self.parser.facts_to_sequences(self.facts)

        Xs = np.array([s_idx for (_, [s_idx, _]) in train_sequences])
        Xp = np.array([p_idx for (p_idx, _) in train_sequences])
        Xo = np.array([o_idx for (_, [_, o_idx]) in train_sequences])

        assert Xs.shape == Xp.shape == Xo.shape

        nb_samples = Xs.shape[0]
        batch_size = math.ceil(nb_samples / nb_batches)
        # logger.info("Samples: {}, no. batches: {} -> batch size: {}".format(nb_samples, nb_batches, batch_size))
        print("Samples: {}, no. batches: {} -> batch size: {}".format(nb_samples, nb_batches, batch_size))

        # projection_steps = [constraints.unit_cube(self.entity_parameters_layer) if unit_cube
        #                     else constraints.unit_sphere(self.entity_parameters_layer, norm=1.0)]

        for epoch in range(1, nb_epochs + 1):
            order = self.random_state.permutation(nb_samples)
            Xs_shuf, Xp_shuf, Xo_shuf = Xs[order], Xp[order], Xo[order]

            Xs_sc, Xp_sc, Xo_sc = subj_corruptor(Xs_shuf, Xp_shuf, Xo_shuf)
            Xs_oc, Xp_oc, Xo_oc = obj_corruptor(Xs_shuf, Xp_shuf, Xo_shuf)

            batches = make_batches(nb_samples, batch_size)

            loss_values = []
            total_loss_value = 0

            nb_versions = 3

            for batch_start, batch_end in batches:
                curr_batch_size = batch_end - batch_start

                Xs_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xs_shuf.dtype)
                Xp_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xp_shuf.dtype)
                Xo_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xo_shuf.dtype)

                # Positive Example
                Xs_batch[0::nb_versions] = Xs_shuf[batch_start:batch_end]
                Xp_batch[0::nb_versions] = Xp_shuf[batch_start:batch_end]
                Xo_batch[0::nb_versions] = Xo_shuf[batch_start:batch_end]

                # Negative examples (corrupting subject)
                Xs_batch[1::nb_versions] = Xs_sc[batch_start:batch_end]
                Xp_batch[1::nb_versions] = Xp_sc[batch_start:batch_end]
                Xo_batch[1::nb_versions] = Xo_sc[batch_start:batch_end]

                # Negative examples (corrupting object)
                Xs_batch[2::nb_versions] = Xs_oc[batch_start:batch_end]
                Xp_batch[2::nb_versions] = Xp_oc[batch_start:batch_end]
                Xo_batch[2::nb_versions] = Xo_oc[batch_start:batch_end]

                y = np.zeros_like(Xp_batch)
                y[0::nb_versions] = 1

                loss_args = {
                    self.s_inputs: Xs_batch,
                    self.p_inputs: Xp_batch,
                    self.o_inputs: Xo_batch,
                    self.y_inputs: y
                }

                _, elbo_value = session.run([self.training_step, self.elbo], feed_dict=loss_args)

                loss_values += [elbo_value / (Xp_batch.shape[0] / nb_versions)]
                total_loss_value += elbo_value

                # for projection_step in projection_steps:
                #     session.run([projection_step])

            def stats(values):
                return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

            # logger.info('Epoch: {0}\tELBO: {1}'.format(epoch, stats(loss_values)))

            if epoch % 50 == 0:
                print('Epoch: {0}\tELBO: {1}'.format(epoch, stats(loss_values)))
