# import math
import numpy as np
import tensorflow as tf
import os
from vkge.knowledgebase import Fact, KnowledgeBaseParser
from functools import reduce
import math
import vkge.models as models
from vkge.training import constraints, corrupt, index
from vkge.training.util import make_batches
import vkge.io as io
from random import randint
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


class VKGE_simple:
    """
           model for testing the non probabilistic aspects of the model

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
    def __init__(self, score_func='dismult', static_mean=False, embedding_size=50, no_batches=10, mean_c=0.1,
                 init_sig=6.0,
                 alt_cost=False, dataset='wn18', sigma_alt=True, lr=0.1, alt_opt=True):
        super().__init__()

        self.sigma_alt = sigma_alt
        self.score_func=score_func

        if init_sig == -1:
            sig_max = init_sig
            sig_min = init_sig
        else:
            if sigma_alt:

                sig_max = tf.log(tf.exp(init_sig) - 1)
                sig_min = sig_max
            else:
                sig_max = (tf.log(init_sig ** 2))  # old sigma
                sig_min = sig_max

                # adjust for correct format for model input

        if self.score_func=='ComplEx':
            predicate_embedding_size = embedding_size*2
            entity_embedding_size = embedding_size*2

        elif self.score_func == 'RESCAL':
            predicate_embedding_size = embedding_size * embedding_size
            entity_embedding_size = embedding_size

        else:
            predicate_embedding_size = embedding_size
            entity_embedding_size = embedding_size


        self.random_state = np.random.RandomState(0)
        tf.set_random_seed(0)
        self.static_mean = static_mean
        self.alt_cost = alt_cost
        self.mean_c = mean_c

        # Dataset
        self.dataset_name = dataset

        logger.warn('Parsing the facts in the Knowledge Base for Dataset {}..'.format(self.dataset_name))

        train_triples = read_triples("data2/{}/train.tsv".format(self.dataset_name))  # choose dataset
        valid_triples = read_triples("data2/{}/dev.tsv".format(self.dataset_name))
        test_triples = read_triples("data2/{}/test.tsv".format(self.dataset_name))
        self.nb_examples = len(train_triples)

        ##### for test time ######
        all_triples = train_triples + valid_triples + test_triples
        entity_set = {s for (s, p, o) in all_triples} | {o for (s, p, o) in all_triples}
        predicate_set = {p for (s, p, o) in all_triples}
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(predicate_set))}
        self.nb_entities, self.nb_predicates = len(entity_set), len(predicate_set)
        ############################
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)        # optimizer=tf.train.AdagradOptimizer(learning_rate=0.1)
        if alt_opt:
            optimizer = tf.train.AdagradOptimizer(learning_rate=lr)  # original KG
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-05)




        self.build_model(self.nb_entities, entity_embedding_size, self.nb_predicates, predicate_embedding_size,
                         optimizer, sig_max, sig_min)
        self.nb_epochs = 1000


        self.train(nb_epochs=self.nb_epochs, test_triples=test_triples, valid_triples=valid_triples,
                   train_triples=train_triples, no_batches=no_batches)

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


        self.build_encoder(nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, ent_sig,
                           pred_sig)
        self.build_decoder()


        self.g_objective = -tf.reduce_sum(tf.log(tf.where(condition=self.y_inputs, x=self.p_x_i, y=1 - self.p_x_i) + 1e-10))

        self.elbo = self.g_objective

        self.training_step = optimizer.minimize(self.elbo)

    def build_encoder(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, sig_max,
                      sig_min):
        """
                                Constructs Encoder
        """
        logger.warn('Building Inference Networks q(h_x | x) ..')

        init1 = np.round((self.mean_c / np.sqrt(entity_embedding_size * 1.0)), decimals=2)

        with tf.variable_scope('encoder'):
            self.entity_embedding_mean = tf.get_variable('entities',
                                                           shape=[nb_entities + 1, entity_embedding_size ],
                                                         initializer=tf.random_uniform_initializer(minval=-init1,
                                                                                                   maxval=init1,
                                                                                                   dtype=tf.float32))
            self.predicate_parameters_layer = tf.get_variable('predicates',
                                                              shape=[nb_predicates + 1, predicate_embedding_size],
                                                              initializer=tf.random_uniform_initializer(minval=-init1,
                                                                                                        maxval=init1,
                                                                                                        dtype=tf.float32))

            self.h_s = tf.nn.embedding_lookup(self.entity_embedding_mean,self.s_inputs)
            self.h_p = tf.nn.embedding_lookup(self.predicate_parameters_layer,self.p_inputs)
            self.h_o = tf.nn.embedding_lookup(self.entity_embedding_mean,self.o_inputs)

            # self.h_s = VKGE.sample_embedding(self.mu_s, self.log_sigma_sq_s)
            # self.h_p = VKGE.sample_embedding(self.mu_p, self.log_sigma_sq_p)
            # self.h_o = VKGE.sample_embedding(self.mu_o, self.log_sigma_sq_o)

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

            elif self.score_func=='ComplEx':
                model = models.ComplexModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                                     object_embeddings=self.h_o)

            elif self.score_func=='TransE':
                model = models.TranslatingModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                                     object_embeddings=self.h_o)

            elif self.score_func=='RESCAL':
                model = models.BilinearModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                                     object_embeddings=self.h_o)


            self.scores = model()
            self.scores_test = model()
            self.p_x_i = tf.sigmoid(self.scores)


    def stats(self,values):
        """
                                Return mean and variance statistics
        """
        return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

    def train(self, test_triples, valid_triples, train_triples, no_batches, session=0, nb_epochs=1000,
              unit_cube=True):
        """
                                Train Model
        """

        # nb_versions = 3
        nb_versions = 1

        earl_stop = 0

        all_triples = train_triples + valid_triples + test_triples

        index_gen = index.GlorotIndexGenerator()

        Xs = np.array([self.entity_to_idx[s] for (s, p, o) in train_triples], dtype=np.int32)
        Xp = np.array([self.predicate_to_idx[p] for (s, p, o) in train_triples], dtype=np.int32)
        Xo = np.array([self.entity_to_idx[o] for (s, p, o) in train_triples], dtype=np.int32)

        assert Xs.shape == Xp.shape == Xo.shape

        nb_samples = Xs.shape[0]
        nb_batches = no_batches
        batch_size = math.ceil(nb_samples / nb_batches)

        batches = make_batches(self.nb_examples, batch_size)

        # projection_steps = [constraints.unit_sphere(self.entity_embedding_mean, norm=5.0)]

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
            session.run(init_op)



            for epoch in range(1, nb_epochs + 1):

                if earl_stop == 1:
                    break

                counter = 0

                kl_inc_val = 1.0

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

                    # # Xs_batch[1::nb_versions] needs to be corrupted
                    # Xs_batch[1::nb_versions] = index_gen(curr_batch_size, np.arange(self.nb_entities))
                    # Xp_batch[1::nb_versions] = Xp_shuf[batch_start:batch_end]
                    # Xo_batch[1::nb_versions] = Xo_shuf[batch_start:batch_end]
                    #
                    # # Xo_batch[2::nb_versions] needs to be corrupted
                    # Xs_batch[2::nb_versions] = Xs_shuf[batch_start:batch_end]
                    # Xp_batch[2::nb_versions] = Xp_shuf[batch_start:batch_end]
                    # Xo_batch[2::nb_versions] = index_gen(curr_batch_size, np.arange(self.nb_entities))

                    loss_args = {
                        self.s_inputs: Xs_batch,
                        self.p_inputs: Xp_batch,
                        self.o_inputs: Xo_batch,
                        self.y_inputs: np.array([1.0] * curr_batch_size)
                        # self.y_inputs: np.array([1.0, 0.0, 0.0] * curr_batch_size)
                    }


                    _, elbo_value = session.run([self.training_step, self.elbo],
                                                         feed_dict=loss_args)

                    # tensorboard

                    # train_writer.add_summary(summary, tf.train.global_step(session, self.global_step))

                    # logger.warn('mu s: {0}\t \t log sig s: {1} \t \t h s {2}'.format(a1,a2,a3 ))


                    loss_values += [elbo_value / (Xp_batch.shape[0] / nb_versions)]
                    total_loss_value += elbo_value

                    counter += 1

                # if (self.projection == True):  # project means
                #     for projection_step in projection_steps:
                #         session.run([projection_step])




                logger.warn('Epoch: {0}\tELBO: {1}'.format(epoch, self.stats(loss_values)))

                ##
                # Early Stopping
                ##

                if (epoch % 50) == 0:

                    eval_name = 'valid'
                    eval_triples = valid_triples
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

                    if hits_at_k > max_hits_at_k:
                        max_hits_at_k = hits_at_k
                    else:
                        earl_stop = 1
                        logger.warn('Early Stopping with valid HITS@10 {}'.format(max_hits_at_k))

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
                    # save embeddings

                    # entity_embeddings,entity_embedding_sigma=session.run([self.entity_embedding_mean,self.entity_embedding_sigma],feed_dict={})
                    # np.savetxt(filename+"/entity_embeddings.tsv", entity_embeddings, delimiter="\t")
                    # np.savetxt(filename+"/entity_embedding_sigma.tsv", entity_embedding_sigma, delimiter="\t")
