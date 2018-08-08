# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf
import os
from vkge.knowledgebase import Fact, KnowledgeBaseParser
from functools import reduce

import vkge.models as models
from vkge.training import constraints, corrupt, index
from vkge.training.util import make_batches
import vkge.io as io
from random import randint
# new

import logging
import sys
import collections

logger = logging.getLogger(__name__)


def read_triples(path):
    triples = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


def unit_cube_projection(var_matrix):
    unit_cube_projection = tf.minimum(1., tf.maximum(var_matrix, 0.))
    return tf.assign(var_matrix, unit_cube_projection)


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


class VKGE_tests:
    """
           model for testing the basic probabilistic aspects of the model, just using SGD optimiser  - !!working!! 91.34%Hits@10

            Achievies
        Initializes a Link Prediction Model.
        @param file_name: The TensorBoard file_name.
        @param opt_type: Determines the optimiser used
        @param embedding_size: The embedding_size for entities and predicates
        @param batch_s: The batch size
        @param lr: The learning rate
        @param b1: The beta 1 value for ADAM optimiser
        @param b2: The beta 2 value for ADAM optimiser
        @param eps: The epsilon value for ADAM optimiser
        @param GPUMode: Used for reduced print statements during architecture search.
        @param alt_cost: Determines the use of a compression cost KL or classical KL term
        @param train_mean: Determines whether the mean embeddings are trainable or fixed
        @param alt_updates: Determines if updates are done simultaneously or
                            separately for each e and g objective.
        @param sigma_alt: Determines between two standard deviation representation used
        @param tensorboard: Determines if Tensorboard events are logged
        @param projection: Determines if mean embeddings are projected
                                     network.

        @type file_name: str: '/home/workspace/acowenri/tboard'
        @type opt_type: str
        @type embedding_size: int
        @type batch_s: int
        @type lr: float
        @type b1: float
        @type b2: float
        @type eps: float
        @type GPUMode: bool
        @type alt_cost: bool
        @type train_mean: bool
        @type alt_updates: bool
        @type sigma_alt: bool
        @type tensorboard: bool
        @type projection: bool

            """

    def __init__(self, file_name, score_func='dismult', static_mean=False, embedding_size=50, no_batches=10, mean_c=0.1,
                 epsilon=1e-3,negsamples=0,
                 alt_cost=False, dataset='wn18', sigma_alt=True, lr=0.1, alt_opt=True, projection=True,alt_updates=False,nosamps=1,alt_test='none'):
        # super().__init__()

        self.alt_test=alt_test
        self.model='ptriple' #decides if we sample per triple or per entity/ predicate/ object
        self.sigma_alt = sigma_alt
        self.score_func=score_func
        self.alt_updates=alt_updates
        self.negsamples=negsamples
        self.alt_opt=alt_opt
        self.nosamps=int(nosamps)
        # sigma = tf.log(1 + tf.exp(log_sigma_square))
        self.no_confidence_samples=1000 #change to 1000
        sig_max = np.log((1.0/embedding_size*1.0))**2

        # sig_max = np.log(np.exp(1.0/embedding_size*1.0)-1)
        sig_min = sig_max

                # adjust for correct format for model input

        if self.score_func=='ComplEx':
            predicate_embedding_size = embedding_size*2
            entity_embedding_size = embedding_size*2


        else:
            predicate_embedding_size = embedding_size
            entity_embedding_size = embedding_size


        self.random_state = np.random.RandomState(0)
        tf.set_random_seed(0)
        self.static_mean = static_mean
        self.alt_cost = alt_cost
        self.projection = projection
        self.mean_c = mean_c

        # Dataset
        self.dataset_name = dataset

        logger.warn('Parsing the facts in the Knowledge Base for Dataset {}..'.format(self.dataset_name))

        train_triples = read_triples("data/{}/train.tsv".format(self.dataset_name))  # choose dataset
        valid_triples = read_triples("data/{}/dev.tsv".format(self.dataset_name))
        test_triples = read_triples("data/{}/test.tsv".format(self.dataset_name))
        self.nb_examples = len(train_triples)

        ##### for test time ######
        all_triples = train_triples + valid_triples + test_triples
        entity_set = {s for (s, p, o) in all_triples} | {o for (s, p, o) in all_triples}
        predicate_set = {p for (s, p, o) in all_triples}
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(predicate_set))}
        self.nb_entities, self.nb_predicates = len(entity_set), len(predicate_set)
        ############################
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # optimizer=tf.train.AdagradOptimizer(learning_rate=lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)

        # optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)

        self.var1_1 = randint(0, self.nb_entities - 1)
        self.var1_2 = randint(0, self.nb_predicates - 1)

        logger.warn("Entity Sample1 id is {} with name {}..".format(self.var1_1, list(entity_set)[self.var1_1]))

        logger.warn("Predicate Sample1 id is {} with name {}..".format(self.var1_2, list(predicate_set)[self.var1_2]))

        self.var1 = tf.Variable(initial_value=self.var1_1, trainable=False, dtype=tf.int32)
        self.var2 = tf.Variable(initial_value=self.var1_2, trainable=False, dtype=tf.int32)

        self.build_model(self.nb_entities, entity_embedding_size, self.nb_predicates, predicate_embedding_size,
                         optimizer, sig_max, sig_min)
        self.nb_epochs = 100


        self.train(nb_epochs=self.nb_epochs, test_triples=test_triples, valid_triples=valid_triples,entity_embedding_size=entity_embedding_size,
                   train_triples=train_triples, no_batches=int(no_batches)  , filename=str(file_name))

    @staticmethod
    def input_parameters(inputs, parameters_layer):
        """
                    Separates distribution parameters from embeddings
        """
        parameters = tf.nn.embedding_lookup(parameters_layer, inputs)
        mu, log_sigma_square = tf.split(value=parameters, num_or_size_splits=2, axis=1)
        return mu, log_sigma_square

    def _setup_summaries(self):

        self.variable_summaries(self.entity_embedding_sigma)

        self.variable_summaries(self.entity_embedding_mean)

        tf.summary.scalar("total e loss", self.e_objective)

        tf.summary.scalar("g loss", self.g_objective)

        tf.summary.scalar("total loss", self.elbo)


        # with tf.name_scope('Samples'):
        #
        #     with tf.name_scope('Entity1'):
        #         self.var1_1 = tf.nn.embedding_lookup(self.entity_embedding_mean, self.var1)
        #         self.var1_2 = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.var1)
        #
        #         with tf.name_scope('Entity1_Mean'):
        #             self.variable_summaries(self.var1_1)
        #
        #         with tf.name_scope('Entity1_logStd'):
        #             self.variable_summaries(tf.sqrt(tf.exp(self.var1_2)))
        #
        #     with tf.name_scope('Predicate1'):
        #         self.var2_1 = tf.nn.embedding_lookup(self.predicate_embedding_mean, self.var2)
        #         self.var2_2 = tf.nn.embedding_lookup(self.predicate_embedding_sigma, self.var2)
        #
        #         with tf.name_scope('Predicate1_Mean'):
        #             self.variable_summaries(self.var2_1)
        #
        #         with tf.name_scope('Predicate1_logStd'):
        #             self.variable_summaries(tf.sqrt(tf.exp(self.var2_2)))

    def _setup_training(self, loss, optimizer=tf.train.AdamOptimizer, l2=0.0, clip_op=None, clip=None):
        global_step = tf.train.get_global_step()
        if global_step is None:
            global_step = tf.train.create_global_step()

        gradients = optimizer.compute_gradients(loss=loss)
        tf.summary.scalar('gradients_l2', tf.add_n([tf.nn.l2_loss(grad[0]) for grad in gradients]),
                          collections=['summary_train'])

        if l2:
            loss += tf.add_n([tf.nn.l2_loss(v) for v in self.train_variables]) * l2
        if clip:
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                             for grad, var in gradients if grad is not None]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var)
                             for grad, var in gradients if grad is not None]

        train_op = optimizer.apply_gradients(gradients, global_step)

        variable_size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list()) if v.get_shape() else 1
        num_params = sum(variable_size(v) for v in self.train_variables)
        print("Number of parameters: {}".format(num_params))

        self.loss = loss
        self.training_step = train_op
        self.global_step = global_step
        return loss, train_op

    def sample_embedding(self, mu, log_sigma_square):
        """
                        Samples from embeddings
                """
        # if self.sigma_alt:
        #     sigma = tf.log(1 + tf.exp(log_sigma_square))
        # else:
        sigma = tf.sqrt(tf.exp(log_sigma_square))

        embedding_size = mu.get_shape()[1].value

        self.eps = tf.random_normal((1, embedding_size), 0, 1, dtype=tf.float32)

        return mu+sigma*self.eps

    def sample_embedding_ptriple(self, mu, log_sigma_square):
        """
                        Samples from embeddings
                """
        # if self.sigma_alt:
        #     sigma = tf.log(1 + tf.exp(log_sigma_square))
        # else:
        sigma = tf.sqrt(tf.exp(log_sigma_square))



        return mu+sigma*self.noise

    def build_model(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, optimizer,
                    sig_max, pred_sig):
        """
                        Constructs Model
        """
        self.noise = tf.placeholder(tf.float32, shape=[None,entity_embedding_size])
        self.bernoulli_elbo_sampling = tf.placeholder(tf.int32, shape=[None])

        self.no_samples = tf.placeholder(tf.int32)
        self.s_inputs = tf.placeholder(tf.int32, shape=[None])
        self.p_inputs = tf.placeholder(tf.int32, shape=[None])
        self.o_inputs = tf.placeholder(tf.int32, shape=[None])
        self.y_inputs = tf.placeholder(tf.bool, shape=[None])
        self.KL_discount = tf.placeholder(tf.float32)  # starts at 0.5
        self.epoch_d = tf.placeholder(tf.float32)  # starts at 0.5

        self.build_encoder(nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, sig_max,
                           pred_sig)
        self.build_decoder()

        # Kullback Leibler divergence   in one go
        self.e_objective = 0.0
        self.e_objective1 = 0.0
        self.e_objective2 = 0.0
        self.e_objective3 = 0.0

        if self.alt_opt: #ml
            self.g_objective = -tf.reduce_sum(
                tf.log(tf.where(condition=self.y_inputs, x=self.p_x_i, y=1 - self.p_x_i) + 1e-10))

        else: #else hinge margin of 1
            self.hinge_losses = tf.nn.relu(1 - self.scores * (2 * tf.cast(self.y_inputs,dtype=tf.float32) - 1))
            self.g_objective = tf.reduce_sum(self.hinge_losses)



        # ####################################  Weight uncertainity in NN's


        # self.mu_all = tf.concat(axis=0, values=[self.mu_s, self.mu_p, self.mu_o])
        # self.log_sigma_all = tf.concat(axis=0, values=[self.log_sigma_sq_s, self.log_sigma_sq_p, self.log_sigma_sq_o])
        # self.samples_all= tf.concat(axis=0, values=[self.h_s, self.h_p, self.h_o])
        # #
        # self.sigma_all= tf.log(1 + tf.exp(self.log_sigma_all ))
        # # self.dist = tf.contrib.distributions.Normal(self.mu_all, self.sigma_all)
        # # self.e_objective = self.dist.pdf(self.samples_all)
        # self.e_objective=tf.sqrt(1 / (2 * 3.14 * (self.sigma_all ** 2)))*tf.exp(-(self.samples_all - self.mu_all)**2 / (2 * (self.samples_all ** 2)))
        #
        # self.e_objective = tf.reduce_sum(tf.log(self.e_objective)) * self.KL_discount

        # ####################################  one KL

        #
        idx=self.bernoulli_elbo_sampling

        # self.mu_s_bs=tf.gather(self.mu_s,idx)
        # self.mu_o_bs=tf.gather(self.mu_o,idx)
        # self.mu_p_bs=tf.gather(self.mu_p,idx)
        #
        # self.log_sigma_sq_s_bs =tf.gather(self.log_sigma_sq_s,idx)
        # self.log_sigma_sq_o_bs =tf.gather(self.log_sigma_sq_o,idx)
        # self.log_sigma_sq_p_bs =tf.gather(self.log_sigma_sq_p,idx)
        #
        # self.mu_all=tf.concat(axis=0,values=[self.mu_s_bs,self.mu_o_bs,self.mu_p_bs])
        # self.log_sigma_all=tf.concat(axis=0,values=[self.log_sigma_sq_s_bs,self.log_sigma_sq_o_bs,self.log_sigma_sq_p_bs])
        # #
        self.mu_all = tf.concat(axis=0, values=[self.mu_s, self.mu_p, self.mu_o])
        self.log_sigma_all = tf.concat(axis=0, values=[self.log_sigma_sq_s, self.log_sigma_sq_p, self.log_sigma_sq_o])
        #

        self.e_objective-= 0.5 * tf.reduce_sum(
                         1. + self.log_sigma_all - tf.square(self.mu_all) - tf.exp(self.log_sigma_all))



        # ####################################  separately KL
        # self.e_objective1 -= 0.5 * tf.reduce_sum(
        #     1. + self.log_sigma_sq_s - tf.square(self.mu_s) - tf.exp(self.log_sigma_sq_s))
        # self.e_objective2 -= 0.5 * tf.reduce_sum(
        #     1. + self.log_sigma_sq_p - tf.square(self.mu_p) - tf.exp(self.log_sigma_sq_p))
        # self.e_objective3 -= 0.5 * tf.reduce_sum(
        #     1. + self.log_sigma_sq_o - tf.square(self.mu_o) - tf.exp(self.log_sigma_sq_o))  # Log likelihood
        #
        #
        # self.e_objective1 = self.e_objective1 * self.KL_discount
        # self.e_objective2 = self.e_objective1 * self.KL_discount
        # self.e_objective3 = self.e_objective1 * self.KL_discount
        #
        # self.e_objective = (self.e_objective1 + self.e_objective2 + self.e_objective3)

        # ####################################  separately KL

        self.training_stepe = optimizer.minimize(self.e_objective)
        self.training_stepg = optimizer.minimize(self.g_objective)



        self.elbo = self.g_objective + self.e_objective

        ##clip for robust learning as observed nans during training

        gradients = optimizer.compute_gradients(loss=self.elbo)

        if True:
            gradients = [(tf.clip_by_value(grad, -1, 1), var)
                         for grad, var in gradients if grad is not None]

        self.training_step = optimizer.apply_gradients(gradients)



        # self.training_step = optimizer.minimize(self.elbo)

        # self.train_variables=tf.trainable_variables()
        # self._setup_training(loss=self.elbo,optimizer=optimizer)
        # self._setup_summaries()
        # self._variables = tf.global_variables()
        self._saver = tf.train.Saver()

    def build_encoder(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, sig_max,
                      sig_min):
        """
                                Constructs Encoder
        """
        logger.warn('Building Inference Networks q(h_x | x) ..{}'.format(self.score_func))

        init1 = np.round((self.mean_c / np.sqrt(entity_embedding_size * 1.0)), decimals=2)
        init2 = np.round(sig_max,decimals=2)

        # experiment 1 parameters, initalises a sigma to 0.031
        # init2 = (np.log(0.05 ** 2))
        # init3 = (np.log(0.05 ** 2))

        with tf.variable_scope('Encoder'):

            with tf.variable_scope('entity'):
                with tf.variable_scope('mu'):

                    self.entity_embedding_mean = tf.get_variable('entities', shape=[nb_entities + 1, entity_embedding_size],
                                                                 initializer=tf.contrib.layers.xavier_initializer())


                with tf.variable_scope('sigma'):

                    self.entity_embedding_sigma = tf.get_variable('entities_sigma',
                                                              shape=[nb_entities + 1, entity_embedding_size],
                                                              initializer=tf.random_uniform_initializer(
                                                                  minval=init2, maxval=init2, dtype=tf.float32),
                                                              dtype=tf.float32)


                self.mu_s = tf.nn.embedding_lookup(self.entity_embedding_mean, self.s_inputs)
                self.log_sigma_sq_s = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.s_inputs)

                self.mu_o = tf.nn.embedding_lookup(self.entity_embedding_mean, self.o_inputs)
                self.log_sigma_sq_o = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.o_inputs)



            with tf.variable_scope('predicate'):
                with tf.variable_scope('sigma'):

                    self.predicate_embedding_sigma = tf.get_variable('predicate_sigma',
                                                             shape=[nb_predicates + 1,
                                                                    predicate_embedding_size],
                                                             initializer=tf.random_uniform_initializer(
                                                                 minval=init2, maxval=init2, dtype=tf.float32),
                                                             dtype=tf.float32)
                with tf.variable_scope('mu'):

                    self.predicate_embedding_mean = tf.get_variable('predicates',
                                                                shape=[nb_predicates + 1, predicate_embedding_size],
                                                                    initializer=tf.contrib.layers.xavier_initializer())

                self.mu_p = tf.nn.embedding_lookup(self.predicate_embedding_mean, self.p_inputs)
                self.log_sigma_sq_p = tf.nn.embedding_lookup(self.predicate_embedding_sigma, self.p_inputs)

            with tf.variable_scope('Decoder'):

                # self.h_s = self.sample_embedding_ptriple(self.mu_s, self.log_sigma_sq_s)
                # self.h_p = self.sample_embedding_ptriple(self.mu_p, self.log_sigma_sq_p)
                # self.h_o = self.sample_embedding_ptriple(self.mu_o, self.log_sigma_sq_o)
                #
                # else:

                self.h_s = self.sample_embedding(self.mu_s, self.log_sigma_sq_s)
                self.h_p = self.sample_embedding(self.mu_p, self.log_sigma_sq_p)
                self.h_o = self.sample_embedding(self.mu_o, self.log_sigma_sq_o)




    def build_decoder(self):
        """
                                Constructs Decoder
        """
        logger.warn('Building Inference Network p(y|h) for {} score function.'.format(self.score_func))
        # w9=['TransE','DistMult','RESCAL','ComplEx']

        with tf.variable_scope('Inference'):

            if self.score_func=='DistMult':

                model = models.BilinearDiagonalModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                                     object_embeddings=self.h_o)

                model_test = models.BilinearDiagonalModel(subject_embeddings=self.mu_s, predicate_embeddings=self.mu_p,
                                                           object_embeddings=self.mu_o)
            elif self.score_func=='ComplEx':
                model = models.ComplexModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                                     object_embeddings=self.h_o)

                model_test = models.ComplexModel(subject_embeddings=self.mu_s, predicate_embeddings=self.mu_p,
                                                          object_embeddings=self.mu_o)

            elif self.score_func=='TransE':
                model = models.TranslatingModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                                     object_embeddings=self.h_o)
                model_test = models.TranslatingModel(subject_embeddings=self.mu_s, predicate_embeddings=self.mu_p,
                                                          object_embeddings=self.mu_o)


            self.scores = model()
            self.scores_test = model_test()
            self.p_x_i = tf.sigmoid(self.scores)
            self.p_x_i_test = tf.sigmoid(self.scores_test)




    def variable_summaries(self, var):
        """Summaries of a Tensor"""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)



    def stats(self, values):
        """
                                Return mean and variance statistics
        """
        return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

    def train(self, test_triples, valid_triples, train_triples, entity_embedding_size,no_batches, session=0, nb_epochs=500, unit_cube=True,
              filename='/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/'):
        """

                                Train Model

        """


        earl_stop = 0

        all_triples = train_triples + valid_triples + test_triples

        len_valid=len(valid_triples)
        len_test=len(test_triples)

        index_gen = index.GlorotIndexGenerator()

        Xs = np.array([self.entity_to_idx[s] for (s, p, o) in train_triples], dtype=np.int32)
        Xp = np.array([self.predicate_to_idx[p] for (s, p, o) in train_triples], dtype=np.int32)
        Xo = np.array([self.entity_to_idx[o] for (s, p, o) in train_triples], dtype=np.int32)

        assert Xs.shape == Xp.shape == Xo.shape

        nb_samples = Xs.shape[0]
        nb_batches = no_batches
        batch_size = math.ceil(nb_samples / nb_batches)

        self.batch_size=batch_size

        batches = make_batches(self.nb_examples, batch_size)

        # self.negsamples = int(5)

        self.negsamples = int(2.0*len(all_triples)*(self.nb_entities-1)/(nb_batches*1.0))

        logger.warn("Number of negative samples per batch is {}, \t batch size is {} \t number of positive triples {}".format(self.negsamples,self.negsamples+batch_size,len(all_triples)))

        nb_versions = int(self.negsamples + 1)  # neg samples + original

        projection_steps = [constraints.unit_sphere(self.entity_embedding_mean, norm=1.0)]

        max_hits_at_k = 0
        ####### COMPRESSION COST PARAMETERS

        M = int(nb_batches)

        pi_s = np.log(2.0) * (M - 1)
        pi_e = np.log(2.0)

        pi_t = np.exp(np.linspace(pi_s, pi_e, M) - M * np.log(2.0))

        pi = (1 / np.sum(pi_t)) * pi_t  # normalise pi
        #####################

        ##
        # Train
        ##

        init_op = tf.global_variables_initializer()
        with tf.Session() as session:
            self._saver.restore(session,'/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/nuclcs_nvkg_v1.w1=1000_w2=0.001_w3=200_w4=kinship_w5=False_w6=0.001_w7=DistMult_w8=20_w9=none_epoch_20.ckpt')


            # train_writer = tf.summary.FileWriter(filename, session.graph)

            experiments=collections.defaultdict(list)

            for epoch in range(0,1):

                #
                #
                # counter = 0
                #
                # kl_inc_val = 1.0
                #
                # order = self.random_state.permutation(nb_samples)
                # Xs_shuf, Xp_shuf, Xo_shuf = Xs[order], Xp[order], Xo[order]
                #
                # loss_values = []
                # total_loss_value = 0
                #
                # for batch_no, (batch_start, batch_end) in enumerate(batches):
                #
                #     curr_batch_size = batch_end - batch_start
                #
                #     Xs_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xs_shuf.dtype)
                #     Xp_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xp_shuf.dtype)
                #     Xo_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xo_shuf.dtype)
                #
                #     Xs_batch[0::nb_versions] = Xs_shuf[batch_start:batch_end]
                #     Xp_batch[0::nb_versions] = Xp_shuf[batch_start:batch_end]
                #     Xo_batch[0::nb_versions] = Xo_shuf[batch_start:batch_end]
                #
                #     for q in range((int(self.negsamples/2))): # Xs_batch[1::nb_versions] needs to be corrupted
                #         Xs_batch[q::nb_versions] = index_gen(curr_batch_size, np.arange(self.nb_entities))
                #         Xp_batch[q::nb_versions] = Xp_shuf[batch_start:batch_end]
                #         Xo_batch[q::nb_versions] = Xo_shuf[batch_start:batch_end]
                #
                #     for q in range((int(self.negsamples/2))): # Xs_batch[1::nb_versions] needs to be corrupted
                #         Xs_batch[q::nb_versions] = Xs_shuf[batch_start:batch_end]
                #         Xp_batch[q::nb_versions] = Xp_shuf[batch_start:batch_end]
                #         Xo_batch[q::nb_versions] = index_gen(curr_batch_size, np.arange(self.nb_entities))
                #
                #
                #     vec_neglabels=[int(1)]+([int(0)]*(int(nb_versions-1)))
                #
                #     #
                #     # loss_args = {
                #     #     self.KL_discount: pi[counter],
                #     #     self.s_inputs: Xs_batch,
                #     #     self.p_inputs: Xp_batch,
                #     #     self.o_inputs: Xo_batch,
                #     #     self.y_inputs: np.array(vec_neglabels * curr_batch_size),
                #     #     self.epoch_d: kl_inc_val
                #     # }
                #     noise=session.run(tf.random_normal((nb_versions*curr_batch_size, entity_embedding_size), 0, 1, dtype=tf.float32))
                #
                #     loss_args = {
                #         self.no_samples:1, #number of samples for precision test
                #         self.KL_discount: 1.0,
                #         self.s_inputs: Xs_batch,
                #         self.p_inputs: Xp_batch,
                #         self.o_inputs: Xo_batch,
                #         self.y_inputs: np.array(vec_neglabels * curr_batch_size),
                #         self.epoch_d: 1.0
                #         ,self.bernoulli_elbo_sampling: np.arange(int(self.batch_size+10))
                #         # ,self.noise:noise
                #     }
                #
                #     # merge = tf.summary.merge_all()  # for TB
                #
                #     _, elbo_value = session.run([ self.training_step, self.elbo],
                #                                          feed_dict=loss_args)
                #
                #         # summary, _, elbo_value = session.run([merge, self.training_step, self.elbo],
                #         #                                      feed_dict=loss_args)
                #
                #         # h_s = session.run(self.h_s2,
                #         #                                  feed_dict=loss_args)
                #         # logger.warn('Shape : {0}'.format(h_s.shape))
                #
                #     # tensorboard
                #
                #     # train_writer.add_summary(summary, tf.train.global_step(session, self.global_step))
                #
                #
                #     # logger.warn('mu s: {0}\t \t log sig s: {1} \t \t h s {2}'.format(a1,a2,a3 ))
                #
                #
                #     loss_values += [elbo_value / (Xp_batch.shape[0] )]
                #     total_loss_value += elbo_value
                #
                #     counter += 1
                #
                # # if (self.projection == True):  # project means
                # #     for projection_step in projection_steps:
                # #         session.run([projection_step])
                #
                #


                # logger.warn('Epoch: {0}\t Negative ELBO: {1}'.format(epoch, self.stats(loss_values)))


                if True:
                    # self._saver.save(session, filename+'_epoch_'+str(epoch)+'.ckpt')


                    eval_name = 'valid'
                    eval_triples = valid_triples
                    ranks_subj, ranks_obj = [], []
                    filtered_ranks_subj, filtered_ranks_obj = [], []

                    for _i, (s, p, o) in enumerate(eval_triples):
                        s_idx, p_idx, o_idx = self.entity_to_idx[s], self.predicate_to_idx[p], \
                                              self.entity_to_idx[o]

                        Xs_v = np.full(shape=(self.nb_entities,), fill_value=s_idx, dtype=np.int32)
                        Xp_v = np.full(shape=(self.nb_entities,), fill_value=p_idx, dtype=np.int32)
                        Xo_v = np.full(shape=(self.nb_entities,), fill_value=o_idx, dtype=np.int32)

                        feed_dict_corrupt_subj = {self.s_inputs: np.arange(self.nb_entities),
                                                  self.p_inputs: Xp_v,
                                                  self.o_inputs: Xo_v}
                        feed_dict_corrupt_obj = {self.s_inputs: Xs_v, self.p_inputs: Xp_v,
                                                 self.o_inputs: np.arange(self.nb_entities)}

                        # scores of (1, p, o), (2, p, o), .., (N, p, o)
                        scores_subj = session.run(self.scores_test, feed_dict=feed_dict_corrupt_subj)

                        # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
                        scores_obj = session.run(self.scores_test, feed_dict=feed_dict_corrupt_obj)

                        ranks_subj += [1 + np.sum(scores_subj > scores_subj[s_idx])]
                        ranks_obj += [1 + np.sum(scores_obj > scores_obj[o_idx])]

                        filtered_scores_subj = scores_subj.copy()
                        filtered_scores_obj = scores_obj.copy()

                        rm_idx_s = [self.entity_to_idx[fs] for (fs, fp, fo) in all_triples if
                                    fs != s and fp == p and fo == o]
                        rm_idx_o = [self.entity_to_idx[fo] for (fs, fp, fo) in all_triples if
                                    fs == s and fp == p and fo != o]

                        filtered_scores_subj[rm_idx_s] = - np.inf
                        filtered_scores_obj[rm_idx_o] = - np.inf

                        filtered_ranks_subj += [1 + np.sum(filtered_scores_subj > filtered_scores_subj[s_idx])]
                        filtered_ranks_obj += [1 + np.sum(filtered_scores_obj > filtered_scores_obj[o_idx])]

                    filtered_ranks = filtered_ranks_subj + filtered_ranks_obj
                    ranks = ranks_subj + ranks_obj

                    for setting_name, setting_ranks in [('Raw', ranks), ('Filtered', filtered_ranks)]:
                        mean_rank = np.mean(setting_ranks)
                        logger.warn('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))
                        for k in [1, 3, 5, 10]:
                            hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                            logger.warn('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))
            ##
            # Test
            ##

            # logger.warn('PRINTING TOP 20 ROWS FROM SAMPLE ENTITY MEAN AND VAR ')
            #
            # samp1_mu, samp1_sig = session.run([self.var1_1, self.var1_2],feed_dict={})
            #
            # logger.warn('Sample Mean \t {} \t Sample Var \t {}'.format(samp1_mu[:20],samp1_sig[:20]))

                    logger.warn('Beginning test phase')


                    eval_name = 'test'
                    eval_triples = test_triples
                    ranks_subj, ranks_obj = [], []
                    filtered_ranks_subj, filtered_ranks_obj = [], []

                    for _i, (s, p, o) in enumerate(eval_triples):

                        #corrupts both a subject and object

                        s_idx, p_idx, o_idx = self.entity_to_idx[s], self.predicate_to_idx[p], self.entity_to_idx[o]

                        Xs_v = np.full(shape=(self.nb_entities,), fill_value=s_idx, dtype=np.int32)
                        Xp_v = np.full(shape=(self.nb_entities,), fill_value=p_idx, dtype=np.int32)
                        Xo_v = np.full(shape=(self.nb_entities,), fill_value=o_idx, dtype=np.int32)

                        feed_dict_corrupt_subj = {self.s_inputs: np.arange(self.nb_entities), self.p_inputs: Xp_v,
                                                  self.o_inputs: Xo_v}
                        feed_dict_corrupt_obj = {self.s_inputs: Xs_v, self.p_inputs: Xp_v,
                                                 self.o_inputs: np.arange(self.nb_entities)}

                        # scores of (1, p, o), (2, p, o), .., (N, p, o)



                        scores_subj = session.run(self.scores_test, feed_dict=feed_dict_corrupt_subj)

                            # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
                        scores_obj = session.run(self.scores_test, feed_dict=feed_dict_corrupt_obj)



                        #########################
                        # Calculate score confidence
                        #########################


                        if self.alt_test in ['test1']: #CORRECTION of scores for confidence TEST1

                            scores_subj=0
                            scores_obj=0

                            for samp_no in range((self.no_confidence_samples)):

                                scores_subj+= session.run(self.scores, feed_dict=feed_dict_corrupt_subj)

                                # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
                                scores_obj += session.run(self.scores, feed_dict=feed_dict_corrupt_obj)

                        if self.alt_test in ['none','test1']:
                            ranks_subj += [1 + np.sum(scores_subj > scores_subj[s_idx])]
                            ranks_obj += [1 + np.sum(scores_obj > scores_obj[o_idx])]

                            hts = [1, 3, 5, 10]

                        else:
                            hts = [1]


                        if self.alt_test in ['test2']: #CORRECTION of scores for confidence TEST1

                            confidence_subj=np.zeros(self.nb_entities)

                            confidence_obj=np.zeros(self.nb_entities)

                            for samp_no in range((self.no_confidence_samples)):

                                scores_subj = session.run(self.p_x_i, feed_dict=feed_dict_corrupt_subj)

                                confidence_subj+= np.divide(scores_subj,self.no_confidence_samples)

                                # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
                                scores_obj = session.run(self.p_x_i, feed_dict=feed_dict_corrupt_obj)

                                confidence_obj+=((scores_obj)/self.no_confidence_samples*1.0)

                        elif self.alt_test in ['test2_bline']: #creates random confidence levels between 0 and 1 for baseline test2
                            confidence_subj=np.random.random_sample(self.nb_entities,)
                            confidence_obj=np.random.random_sample(self.nb_entities,)
                        #########################
                        # Calculate new scores wrs to confidence
                        #########################


                        if self.alt_test in ['test2','test2_bline']: #multiply by binary threshold on variance

                            scores_subj = scores_subj

                            scores_obj = scores_obj

                            if (confidence_subj[s_idx] > self.p_threshold): #need to index subj and obj here
                                ranks_subj += [1 + np.sum(scores_subj > scores_subj[s_idx])]
                            if (confidence_obj[o_idx] > self.p_threshold):
                                ranks_obj += [1 + np.sum(scores_obj > scores_obj[o_idx])]



                        filtered_scores_subj = scores_subj.copy()
                        filtered_scores_obj = scores_obj.copy()

                        rm_idx_s = [self.entity_to_idx[fs] for (fs, fp, fo) in all_triples if
                                    fs != s and fp == p and fo == o]
                        rm_idx_o = [self.entity_to_idx[fo] for (fs, fp, fo) in all_triples if
                                    fs == s and fp == p and fo != o]

                        filtered_scores_subj[rm_idx_s] = - np.inf
                        filtered_scores_obj[rm_idx_o] = - np.inf


                        if self.alt_test in ['none','test1']:  # multiply by probability

                            filtered_ranks_subj += [1 + np.sum(filtered_scores_subj > filtered_scores_subj[s_idx])]
                            filtered_ranks_obj += [1 + np.sum(filtered_scores_obj > filtered_scores_obj[o_idx])]

                        elif self.alt_test in ['test2','test2_bline']:  # multiply by binary threshold on variance

                            if (confidence_subj[s_idx] > self.p_threshold):
                                filtered_ranks_subj += [1 + np.sum(filtered_scores_subj > filtered_scores_subj[s_idx])]
                            if (confidence_obj[o_idx] > self.p_threshold):
                                filtered_ranks_obj += [1 + np.sum(filtered_scores_obj > filtered_scores_obj[o_idx])]
                        if [1 + np.sum(filtered_scores_subj > filtered_scores_subj[s_idx])] == [1]:
                            logger.warn(
                                "\t \t filtered_scores_subj  rank 1 idx is {},{},{} \t \t".format(s_idx, o_idx, p_idx))

                        if [1 + np.sum(filtered_scores_obj > filtered_scores_obj[o_idx])] == [1]:
                            logger.warn(
                                "\t \t filtered_scores_obj rank 1 idx is {},{},{} \t \t".format(s_idx, o_idx, p_idx))

                    filtered_ranks = filtered_ranks_subj + filtered_ranks_obj
                    ranks = ranks_subj + ranks_obj
                    logger.warn("\t \t Number of samples in valid phase {} \t \t".format(len(filtered_ranks_obj)))
                    for setting_name, setting_ranks in [('Raw', ranks), ('Filtered', filtered_ranks)]:
                        mean_rank = np.mean(setting_ranks)
                        logger.warn('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))
                        for k in hts:
                            hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                            logger.warn('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))




                            # if setting_name=='Filtered' and self.alt_test=='test1':
                            #     experiments[k].append(hits_at_k)
                            #     logger.warn('[{}] {} Hits@{} List: {}'.format(eval_name, setting_name, k, experiments[k]))

                    # e1, e2, p1, p2 = session.run(
                    #     [self.entity_embedding_mean, self.entity_embedding_sigma, self.predicate_embedding_mean,
                    #      self.predicate_embedding_sigma], feed_dict={})
                    # np.save(
                    #     "/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/embeddings/entity_embeddings",
                    #     e1)
                    # np.save(
                    #     "/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/embeddings/entity_embedding_sigma",
                    #     e2)
                    # np.save(
                    #     "/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/embeddings/predicate_embedding_mean",
                    #     p1)
                    # np.save(
                    #     "/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/embeddings/predicate_embedding_sigma",
                    #     p2)






                    # entity_embeddings,entity_embedding_sigma=session.run([self.entity_embedding_mean,self.entity_embedding_sigma],feed_dict={})
                    # np.savetxt(filename+"/entity_embeddings.tsv", entity_embeddings, delimiter="\t")
                    # np.savetxt(filename+"/entity_embedding_sigma.tsv", entity_embedding_sigma, delimiter="\t")
                #
                # if (epoch % 50) == 0:
                #
                #     eval_name = 'valid'
                #     eval_triples = valid_triples
                #     ranks_subj, ranks_obj = [], []
                #     filtered_ranks_subj, filtered_ranks_obj = [], []
                #
                #     for _i, (s, p, o) in enumerate(eval_triples):
                #         s_idx, p_idx, o_idx = self.entity_to_idx[s], self.predicate_to_idx[p], \
                #                               self.entity_to_idx[o]
                #
                #         Xs_v = np.full(shape=(self.nb_entities,), fill_value=s_idx, dtype=np.int32)
                #         Xp_v = np.full(shape=(self.nb_entities,), fill_value=p_idx, dtype=np.int32)
                #         Xo_v = np.full(shape=(self.nb_entities,), fill_value=o_idx, dtype=np.int32)
                #
                #         feed_dict_corrupt_subj = {self.s_inputs: np.arange(self.nb_entities),
                #                                   self.p_inputs: Xp_v,
                #                                   self.o_inputs: Xo_v}
                #         feed_dict_corrupt_obj = {self.s_inputs: Xs_v, self.p_inputs: Xp_v,
                #                                  self.o_inputs: np.arange(self.nb_entities)}
                #
                #         # scores of (1, p, o), (2, p, o), .., (N, p, o)
                #         scores_subj = session.run(self.scores_test, feed_dict=feed_dict_corrupt_subj)
                #
                #         # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
                #         scores_obj = session.run(self.scores_test, feed_dict=feed_dict_corrupt_obj)
                #
                #         ranks_subj += [1 + np.sum(scores_subj > scores_subj[s_idx])]
                #         ranks_obj += [1 + np.sum(scores_obj > scores_obj[o_idx])]
                #
                #         filtered_scores_subj = scores_subj.copy()
                #         filtered_scores_obj = scores_obj.copy()
                #
                #         rm_idx_s = [self.entity_to_idx[fs] for (fs, fp, fo) in all_triples if
                #                     fs != s and fp == p and fo == o]
                #         rm_idx_o = [self.entity_to_idx[fo] for (fs, fp, fo) in all_triples if
                #                     fs == s and fp == p and fo != o]
                #
                #         filtered_scores_subj[rm_idx_s] = - np.inf
                #         filtered_scores_obj[rm_idx_o] = - np.inf
                #
                #         filtered_ranks_subj += [1 + np.sum(filtered_scores_subj > filtered_scores_subj[s_idx])]
                #         filtered_ranks_obj += [1 + np.sum(filtered_scores_obj > filtered_scores_obj[o_idx])]
                #
                #     filtered_ranks = filtered_ranks_subj + filtered_ranks_obj
                #     ranks = ranks_subj + ranks_obj
                #
                #     for setting_name, setting_ranks in [('Raw', ranks), ('Filtered', filtered_ranks)]:
                #         mean_rank = np.mean(setting_ranks)
                #         logger.warn('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))
                #         for k in [1, 3, 5, 10]:
                #             hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                #             logger.warn('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))
                #
                #     ##
                #     # Test
                #     ##
                #
                #     # logger.warn('PRINTING TOP 20 ROWS FROM SAMPLE ENTITY MEAN AND VAR ')
                #     #
                #     # samp1_mu, samp1_sig = session.run([self.var1_1, self.var1_2],feed_dict={})
                #     #
                #     # logger.warn('Sample Mean \t {} \t Sample Var \t {}'.format(samp1_mu[:20],samp1_sig[:20]))
                #
                #     logger.warn('Beginning test phase')
                #
                #     eval_name = 'test'
                #     eval_triples = test_triples
                #     ranks_subj, ranks_obj = [], []
                #     filtered_ranks_subj, filtered_ranks_obj = [], []
                #
                #     for _i, (s, p, o) in enumerate(eval_triples):
                #         s_idx, p_idx, o_idx = self.entity_to_idx[s], self.predicate_to_idx[p], \
                #                               self.entity_to_idx[o]
                #
                #         Xs_v = np.full(shape=(self.nb_entities,), fill_value=s_idx, dtype=np.int32)
                #         Xp_v = np.full(shape=(self.nb_entities,), fill_value=p_idx, dtype=np.int32)
                #         Xo_v = np.full(shape=(self.nb_entities,), fill_value=o_idx, dtype=np.int32)
                #
                #         feed_dict_corrupt_subj = {self.s_inputs: np.arange(self.nb_entities),
                #                                   self.p_inputs: Xp_v,
                #                                   self.o_inputs: Xo_v}
                #         feed_dict_corrupt_obj = {self.s_inputs: Xs_v, self.p_inputs: Xp_v,
                #                                  self.o_inputs: np.arange(self.nb_entities)}
                #
                #         # scores of (1, p, o), (2, p, o), .., (N, p, o)
                #         scores_subj = session.run(self.scores_test, feed_dict=feed_dict_corrupt_subj)
                #
                #         # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
                #         scores_obj = session.run(self.scores_test, feed_dict=feed_dict_corrupt_obj)
                #
                #         ranks_subj += [1 + np.sum(scores_subj > scores_subj[s_idx])]
                #         ranks_obj += [1 + np.sum(scores_obj > scores_obj[o_idx])]
                #
                #         filtered_scores_subj = scores_subj.copy()
                #         filtered_scores_obj = scores_obj.copy()
                #
                #         rm_idx_s = [self.entity_to_idx[fs] for (fs, fp, fo) in all_triples if
                #                     fs != s and fp == p and fo == o]
                #         rm_idx_o = [self.entity_to_idx[fo] for (fs, fp, fo) in all_triples if
                #                     fs == s and fp == p and fo != o]
                #
                #         filtered_scores_subj[rm_idx_s] = - np.inf
                #         filtered_scores_obj[rm_idx_o] = - np.inf
                #
                #         filtered_ranks_subj += [1 + np.sum(filtered_scores_subj > filtered_scores_subj[s_idx])]
                #         filtered_ranks_obj += [1 + np.sum(filtered_scores_obj > filtered_scores_obj[o_idx])]
                #
                #     filtered_ranks = filtered_ranks_subj + filtered_ranks_obj
                #     ranks = ranks_subj + ranks_obj
                #
                #     for setting_name, setting_ranks in [('Raw', ranks), ('Filtered', filtered_ranks)]:
                #         mean_rank = np.mean(setting_ranks)
                #         logger.warn('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))
                #         for k in [1, 3, 5, 10]:
                #             hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                #             logger.warn('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))
