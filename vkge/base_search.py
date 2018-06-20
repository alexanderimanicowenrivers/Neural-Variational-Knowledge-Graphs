# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf

from vkge.knowledgebase import Fact, KnowledgeBaseParser

import vkge.models as models
from vkge.training import constraints, corrupt, index
from vkge.training.util import make_batches
import vkge.io as io


# new

import logging
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


class VKGE2:
    """
           model for testing the basic probabilistic aspects of the model, just using SGD optimiser  - !!working!! 91.34% Hits@10 saved

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
    def __init__(self,decay_kl, file_name,embedding_size=5,batch_s=14145, lr=0.001, b1=0.9, b2=0.999, eps=1e-08, GPUMode=False, ent_sig=6.0,
                 alt_cost=False,static_pred=False,static_mean=False,alt_updates=True,sigma_alt=True,opt_type='ml',tensorboard=True,projection=True,opt='adam'):
        super().__init__()

        self.sigma_alt=sigma_alt

        if ent_sig==-1:
            ent_sigma=ent_sig
        else:
            if sigma_alt:
                ent_sigma=tf.log(tf.exp(ent_sig)-1)
            else:
                ent_sigma = (np.log(ent_sig**2)) #old sigma

        pred_sigma = ent_sigma #adjust for correct format for model input
        predicate_embedding_size = embedding_size
        entity_embedding_size = embedding_size
        self.random_state = np.random.RandomState(0)
        tf.set_random_seed(0)


        self.GPUMode = GPUMode
        self.alt_cost = alt_cost
        self.static_mean=static_mean
        self.alt_updates=alt_updates
        self.tensorboard=tensorboard
        self.projection=projection
        logger.warn('This model is probabilistic ..')

        logger.warn('Parsing the facts in the Knowledge Base ..')

        # Dataset
        dataset_name = 'wn18'

        train_triples = read_triples("data2/{}/train.tsv".format(dataset_name))  # choose dataset
        test_triples = read_triples("data2/{}/dev.tsv".format(dataset_name))
        self.nb_examples = len(train_triples)


        ##### for test time ######
        all_triples = train_triples + test_triples
        entity_set = {s for (s, p, o) in all_triples} | {o for (s, p, o) in all_triples}
        predicate_set = {p for (s, p, o) in all_triples}
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(predicate_set))}
        self.nb_entities, self.nb_predicates = len(entity_set), len(predicate_set)
        ############################

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)        # optimizer=tf.train.AdagradOptimizer(learning_rate=0.1)
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.1)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.01, epsilon=1e-5)

        self.build_model(self.nb_entities, entity_embedding_size, self.nb_predicates, predicate_embedding_size,
                         optimizer,
                         ent_sigma, pred_sigma)

        self.train(nb_epochs=1000, test_triples=test_triples, train_triples=train_triples,batch_size=batch_s,filename=file_name)

    @staticmethod
    def input_parameters(inputs, parameters_layer):
        """
                    Separates distribution parameters from embeddings
        """
        parameters = tf.nn.embedding_lookup(parameters_layer, inputs)
        mu, log_sigma_square = tf.split(value=parameters, num_or_size_splits=2, axis=1)
        return mu, log_sigma_square

    @staticmethod
    def sample_embedding(mu, log_sigma_square):
        """
                Samples from embeddings
        """
        sigma = tf.sqrt(tf.exp(log_sigma_square))
        embedding_size = mu.get_shape()[1].value
        eps = tf.random_normal((1, embedding_size), 0, 1, dtype=tf.float32)
        return mu + sigma * eps

    def build_model(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, optimizer,
                    ent_sig, pred_sig):
        """
                        Constructs Model
        """
        self.s_inputs = tf.placeholder(tf.int32, shape=[None])
        self.p_inputs = tf.placeholder(tf.int32, shape=[None])
        self.o_inputs = tf.placeholder(tf.int32, shape=[None])
        self.y_inputs = tf.placeholder(tf.bool, shape=[None])

        self.KL_discount = tf.placeholder(tf.float32)  # starts at 0.5

        self.build_encoder(nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, ent_sig,
                           pred_sig)
        self.build_decoder()

        # Kullback Leibler divergence
        # self.e_objective -= 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_s - tf.square(self.mu_s) - tf.exp(self.log_sigma_sq_s))
        # self.e_objective -= 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_p - tf.square(self.mu_p) - tf.exp(self.log_sigma_sq_p))
        # self.e_objective -= 0.5 * tf.reduce_sum(1. + self.log_sigma_sq_o - tf.square(self.mu_o) - tf.exp(self.log_sigma_sq_o))

        # Log likelihood
        # self.g_objective = -tf.reduce_sum(tf.log(tf.gather(self.p_x_i, self.y_inputs) + 1e-10))

        self.hinge_losses = tf.nn.relu(5 - self.scores * (2 * tf.cast(self.y_inputs,dtype=tf.float32) - 1))
        self.g_objective = tf.reduce_sum(self.hinge_losses)

        # self.g_objective = -tf.reduce_sum(tf.log(tf.where(condition=self.y_inputs, x=self.p_x_i, y=1 - self.p_x_i) + 1e-10))

        self.elbo = self.g_objective

        self.training_step = optimizer.minimize(self.elbo)

        if self.tensorboard:

            _ = tf.summary.scalar("total e loss", self.e_objective)
            _ = tf.summary.scalar("g loss", self.g_objective)

            _ = tf.summary.scalar("total loss", self.elbo)



    def build_encoder(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, ent_sig,
                      pred_sig):
        """
                                Constructs Encoder
        """
        logger.warn('Building Inference Networks q(h_x | x) ..')

        init1=np.round((6.0/np.sqrt(entity_embedding_size*1.0)), decimals=2)
        init2=(np.log(0.05**2))
        init3=(np.log(0.05**2))
        logger.warn('init is {} ..'.format(init1))

        with tf.variable_scope('encoder'):
            self.entity_embedding_mean = tf.get_variable('entities',
                                                           shape=[nb_entities + 1, entity_embedding_size ],
                                                           initializer=tf.random_uniform_initializer(minval=-init1,maxval=init1,dtype=tf.float32))
            self.predicate_embedding_mean = tf.get_variable('predicates',
                                                              shape=[nb_predicates + 1, predicate_embedding_size],
                                                            initializer=tf.random_uniform_initializer(minval=-init1,
                                                                                                      maxval=init1,dtype=tf.float32))
            self.entity_embedding_sigma = tf.get_variable('entities_sigma',
                                                          shape=[nb_entities + 1, entity_embedding_size],
                                                          initializer=tf.random_uniform_initializer(minval=init2,maxval=init3,dtype=tf.float32)            , dtype=tf.float32,
                                                          trainable=True)

            self.predicate_embedding_sigma = tf.get_variable('predicate_sigma',
                                                             shape=[nb_predicates + 1, predicate_embedding_size],
                                                          initializer=tf.random_uniform_initializer(minval=init2,maxval=init3,dtype=tf.float32), dtype=tf.float32,
                                                          trainable=True)

            self.mu_s = tf.nn.embedding_lookup(self.entity_embedding_mean, self.s_inputs)
            self.log_sigma_sq_s = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.s_inputs)
            self.h_s = self.sample_embedding(self.mu_s, self.log_sigma_sq_s)

            self.mu_o = tf.nn.embedding_lookup(self.entity_embedding_mean, self.o_inputs)
            self.log_sigma_sq_o = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.o_inputs)
            self.h_o = self.sample_embedding(self.mu_o, self.log_sigma_sq_o)

            self.mu_p = tf.nn.embedding_lookup(self.predicate_embedding_mean, self.p_inputs)
            self.log_sigma_sq_p = tf.nn.embedding_lookup(self.predicate_embedding_sigma, self.p_inputs)
            self.h_p = self.sample_embedding(self.mu_p, self.log_sigma_sq_p)



    def build_decoder(self):
        """
                                Constructs Decoder
        """
        logger.warn('Building Inference Network p(y|h) ..')

        with tf.variable_scope('decoder'):
            model = models.BilinearDiagonalModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                                 object_embeddings=self.h_o)
            self.scores = model()
            self.p_x_i = self.scores


    def stats(self,values):
        """
                                Return mean and variance statistics
        """
        return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

    def train(self, test_triples, train_triples, batch_size, session=0, nb_epochs=1000,unit_cube=True,filename='/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/'):
        """
                                Train Model
        """

        nb_versions = 3

        all_triples=train_triples+test_triples

        index_gen = index.GlorotIndexGenerator()

        Xs = np.array([self.entity_to_idx[s] for (s, p, o) in train_triples], dtype=np.int32)
        Xp = np.array([self.predicate_to_idx[p] for (s, p, o) in train_triples], dtype=np.int32)
        Xo = np.array([self.entity_to_idx[o] for (s, p, o) in train_triples], dtype=np.int32)

        assert Xs.shape == Xp.shape == Xo.shape


        nb_samples = Xs.shape[0]
        nb_batches= math.ceil(nb_samples / batch_size)
        batch_size = math.ceil(nb_samples / nb_batches)

        batches = make_batches(self.nb_examples, batch_size)

        # logger.warn("Samples: {}, no. batches: {} -> batch size: {}".format(nb_samples, nb_batches, batch_size))
        logger.warn("Samples: {}, no. batches: {} -> batch size: {}".format(nb_samples, nb_batches, batch_size))

        projection_steps = [constraints.unit_cube(self.entity_embedding_mean) if unit_cube
                            else constraints.unit_sphere(self.entity_embedding_mean, norm=1.0)]

        minloss = 10000
        maxhits=0
        maxepoch=0
        minepoch = 0

        ####### COMPRESSION COST PARAMETERS

        M = int(nb_batches + 1)

        pi_s = np.log(2.0)*M
        pi_e = np.log(2.0)

        pi = np.exp(np.linspace(pi_s, pi_e, M)-M*np.log(2.0))


        #####################

        ##
        # Train
        ##

        init_op = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init_op)

            if self.tensorboard:
                train_writer = tf.summary.FileWriter(filename, session.graph)

            for epoch in range(1, nb_epochs + 1):
                counter = 0

                order = self.random_state.permutation(nb_samples)
                Xs_shuf, Xp_shuf, Xo_shuf = Xs[order], Xp[order], Xo[order]


                loss_values = []
                total_loss_value = 0


                for batch_no, (batch_start, batch_end) in enumerate(batches):

                    curr_batch_size = batch_end - batch_start

                    Xs_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xs_shuf.dtype)
                    Xp_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xp_shuf.dtype)
                    Xo_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xo_shuf.dtype)

                    Xs_batch[0::nb_versions] = Xs_shuf[batch_start:batch_end]
                    Xp_batch[0::nb_versions] = Xp_shuf[batch_start:batch_end]
                    Xo_batch[0::nb_versions] = Xo_shuf[batch_start:batch_end]

                    # Xs_batch[1::nb_versions] needs to be corrupted
                    Xs_batch[1::nb_versions] = index_gen(curr_batch_size, np.arange(self.nb_entities))
                    Xp_batch[1::nb_versions] = Xp_shuf[batch_start:batch_end]
                    Xo_batch[1::nb_versions] = Xo_shuf[batch_start:batch_end]

                    # Xo_batch[2::nb_versions] needs to be corrupted
                    Xs_batch[2::nb_versions] = Xs_shuf[batch_start:batch_end]
                    Xp_batch[2::nb_versions] = Xp_shuf[batch_start:batch_end]
                    Xo_batch[2::nb_versions] = index_gen(curr_batch_size, np.arange(self.nb_entities))

                    # y = np.zeros_like(Xp_batch)
                    # y[0::nb_versions] = 1

                    loss_args = {
                            self.KL_discount: (1.0/nb_batches),
                            self.s_inputs: Xs_batch,
                            self.p_inputs: Xp_batch,
                            self.o_inputs: Xo_batch,
                            self.y_inputs: np.array([1.0, 0.0, 0.0] * curr_batch_size)
                        }




                    a1,a2,a3,_, elbo_value = session.run([self.mu_s,self.log_sigma_sq_s,self.h_s,self.training_step, self.elbo],
                                                                 feed_dict=loss_args)

                    # logger.warn('mu s: {0}\t \t log sig s: {1} \t \t h s {2}'.format(a1,a2,a3 ))

                    if counter % 2 == 0:
                        if self.tensorboard:
                            train_writer.add_summary(summary, counter)  # tensorboard

                    loss_values += [elbo_value / (Xp_batch.shape[0] / nb_versions)]
                    total_loss_value += elbo_value

                    counter += 1

                    for projection_step in projection_steps:
                        session.run([projection_step])


                if (round(np.mean(loss_values), 4) < minloss):
                    minloss = round(np.mean(loss_values), 4)
                    minepoch = epoch
                else:

                    if (round(np.mean(loss_values), 4) < minloss):
                        minloss = round(np.mean(loss_values), 4)
                        minepoch = epoch

                logger.warn('Epoch: {0}\tELBO: {1}'.format(epoch, self.stats(loss_values)))

                ##
                # Test
                ##

                if (epoch % 500) == 0:
                    eval_name='valid'
                    eval_triples = test_triples
                    ranks_subj, ranks_obj = [], []
                    filtered_ranks_subj, filtered_ranks_obj = [], []

                    for _i, (s, p, o) in enumerate(eval_triples):
                        s_idx, p_idx, o_idx = self.entity_to_idx[s], self.predicate_to_idx[p], self.entity_to_idx[o]

                        Xs_v = np.full(shape=(self.nb_entities,), fill_value=s_idx, dtype=np.int32)
                        Xp_v = np.full(shape=(self.nb_entities,), fill_value=p_idx, dtype=np.int32)
                        Xo_v = np.full(shape=(self.nb_entities,), fill_value=o_idx, dtype=np.int32)

                        feed_dict_corrupt_subj = {self.s_inputs: np.arange(self.nb_entities), self.p_inputs: Xp_v,
                                                  self.o_inputs: Xo_v}
                        feed_dict_corrupt_obj = {self.s_inputs: Xs_v, self.p_inputs: Xp_v,
                                                 self.o_inputs: np.arange(self.nb_entities)}

                        # scores of (1, p, o), (2, p, o), .., (N, p, o)
                        scores_subj = session.run(self.scores, feed_dict=feed_dict_corrupt_subj)

                        # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
                        scores_obj = session.run(self.scores, feed_dict=feed_dict_corrupt_obj)

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


                    if hits_at_k>maxhits:
                        maxhits=hits_at_k
                        maxepoch=epoch

            logger.warn("The minimum loss achieved is {0} \t at epoch {1}".format(minloss, minepoch))
            logger.warn("The maximum Hits@10 value: {0} \t at epoch {1}".format(maxhits, maxepoch))