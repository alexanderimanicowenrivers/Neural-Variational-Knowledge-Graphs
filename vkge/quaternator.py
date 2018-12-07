# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf
from vkge.training import constraints, index
from vkge.training import util as util
import vkge.models as models
import logging
import tensorflow_probability as tfp
import sys

tfd = tfp.distributions
logger = logging.getLogger(__name__)


class Quaternator:
    """
           Latent Fact Model

        Initializes and trains Link Prediction Model.

        @param file_name: The TensorBoard file_name.
        @param embedding_size: The embedding_size for entities and predicates
        @param no_batches: The number of batches per epoch
        @param lr: The learning rate
        @param eps: The epsilon value for ADAM optimiser
        @param distribution: The prior distribution we assume generates the data
        @param dataset: The dataset which the model is trained on
        @param projection: Determines if variance embeddings constrained to sum to unit variance.
        @param alt_prior: Determines whether a normal or specific Gaussian prior is used.

        @type file_name: str: '/home/workspace/acowenri/tboard'
        @type embedding_size: int
        @type no_batches: int
        @type lr: float
        @type eps: float
        @type distribution: str
        @type dataset: str
        @type projection: bool
        @type alt_prior: bool

            """

    def __init__(self, file_name, score_func='DistMult', embedding_size=200, no_batches=10, distribution='normal',
                 epsilon=1e-3, negsamples=5, dataset='wn18', lr=0.01, projection=False):

        self.nb_epochs = 500

        seed = np.random.randint(100, size=1)[0]
        self.random_state = np.random.RandomState(seed)
        tf.set_random_seed(seed)
        logger.warning("\n \n Using Random Seed {} \n \n".format(seed))

        self.score_func = score_func
        self.negsamples = int(negsamples)
        self.distribution = distribution
        self.projection = projection
        # optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

        ##### dataset  ######
        self.dataset_name = dataset
        logger.warning('Parsing the facts in the Knowledge Base for Dataset {}..'.format(self.dataset_name))
        train_triples = util.read_triples("data/{}/train.tsv".format(self.dataset_name))  # choose dataset
        valid_triples = util.read_triples("data/{}/dev.tsv".format(self.dataset_name))
        test_triples = util.read_triples("data/{}/test.tsv".format(self.dataset_name))
        self.nb_examples = len(train_triples)
        all_triples = train_triples + valid_triples + test_triples
        entity_set = {s for (s, p, o) in all_triples} | {o for (s, p, o) in all_triples}
        predicate_set = {p for (s, p, o) in all_triples}
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(predicate_set))}
        self.nb_entities, self.nb_predicates = len(entity_set), len(predicate_set)

        ############## placeholders
        self.s_inputs = tf.placeholder(tf.int32, shape=[None])
        self.p_inputs = tf.placeholder(tf.int32, shape=[None])
        self.o_inputs = tf.placeholder(tf.int32, shape=[None])
        self.y_inputs = tf.placeholder(tf.bool, shape=[None])
        ############## Model

        self.build_model(self.nb_entities, self.nb_predicates, embedding_size, optimizer)

        #### Q normaliser

        rel_r, rel_x, rel_y, rel_z = tf.split(self.q_predicate_embedding, 4, axis=1)
        ent_r, ent_x, ent_y, ent_z = tf.split(self.q_entity_embedding, 4, axis=1)

        self.normaliser_r = tf.sqrt(
            tf.square(rel_r) + tf.square(rel_x) + tf.square(rel_y) + tf.square(rel_z) )
        self.normaliser_e = tf.sqrt(
            tf.square(ent_r) + tf.square(ent_x) + tf.square(ent_y) + tf.square(ent_z) )

        rel_r = tf.div(rel_r, self.normaliser_r)
        rel_x = tf.div(rel_x, self.normaliser_r)
        rel_y = tf.div(rel_y, self.normaliser_r)
        rel_z = tf.div(rel_z, self.normaliser_r)

        q_predicate_embedding = tf.concat([rel_r, rel_x, rel_y, rel_z], axis=1)  # double check this

        ent_r = tf.div(ent_r, self.normaliser_e)
        ent_x = tf.div(ent_x, self.normaliser_e)
        ent_y = tf.div(ent_y, self.normaliser_e)
        ent_z = tf.div(ent_z, self.normaliser_e)

        q_entity_embedding = tf.concat([ent_r, ent_x, ent_y, ent_z], axis=1)  # double check this

        self.projection_steps = [tf.assign(self.q_entity_embedding, q_entity_embedding),
                                 tf.assign(self.q_predicate_embedding, q_predicate_embedding)]

        self.train(nb_epochs=self.nb_epochs, test_triples=test_triples, valid_triples=valid_triples,
                   embedding_size=embedding_size,
                   train_triples=train_triples, no_batches=int(no_batches), filename=str(file_name))

    def _setup_training(self, loss, optimizer=tf.train.AdamOptimizer,clip_op=False,reg=True):
        global_step = tf.train.get_global_step()
        if global_step is None:
            global_step = tf.train.create_global_step()

        if reg:
            self.loss += tf.add_n([tf.nn.l2_loss(v) for v in self.train_variables]) * 0.01

        gradients = optimizer.compute_gradients(loss=loss)

        if clip_op:
            gradients = [(tf.clip_by_value(grad, -1, 1), var)
                         for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(gradients, global_step)

        self.loss = loss
        self.training_step = train_op

        return loss, train_op

    def build_encoder(self, nb_entities, nb_predicates, embedding_size):
        """
                                Constructs encoder
        """
        logger.warning('Building Inference Networks q(h_x | x) ..{}'.format(self.score_func))


        pi=np.pi

        with tf.variable_scope('encoder'):
            with tf.variable_scope('entity'):

                self.phi_init = tf.get_variable('phi_init',
                                                shape=[nb_entities + 1, embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer(),trainable=False)

                self.theta_init = tf.get_variable('theta_init', shape=[nb_entities + 1, embedding_size],
                                                             initializer=tf.random_uniform_initializer(minval=-pi,
                                                                                                       maxval=pi,
                                                                                                       dtype=tf.float32),trainable=False)

                self.r_init = tf.get_variable('r_init', shape=[nb_entities + 1, embedding_size],
                                              initializer=tf.random_uniform_initializer(minval=0,
                                                                                        maxval=0,
                                                                                        dtype=tf.float32),trainable=False)

                self.x_init=tf.get_variable('x_init', shape=[nb_entities + 1, embedding_size],
                                                             initializer=tf.random_uniform_initializer(minval=0,
                                                                                                       maxval=1,
                                                                                                       dtype=tf.float32),trainable=False)

                self.y_init=tf.get_variable('y_init', shape=[nb_entities + 1, embedding_size],
                                                             initializer=tf.random_uniform_initializer(minval=0,
                                                                                                       maxval=1,
                                                                                                       dtype=tf.float32),trainable=False)

                self.z_init= tf.get_variable('z_init', shape=[nb_entities + 1, embedding_size],
                                                             initializer=tf.random_uniform_initializer(minval=0,
                                                                                                       maxval=1,
                                                                                                       dtype=tf.float32),trainable=False)

                self.normaliser=tf.sqrt(tf.square(self.x_init)+tf.square(self.y_init)+tf.square(self.z_init))

                self.q_i=tf.div(self.x_init,self.normaliser)
                self.q_j=tf.div(self.y_init,self.normaliser)
                self.q_k=tf.div(self.z_init,self.normaliser)

                self.w_r=self.phi_init*tf.cos(self.theta_init)
                self.w_i=self.phi_init*self.q_i*tf.sin(self.theta_init)
                self.w_j=self.phi_init*self.q_j*tf.sin(self.theta_init)
                self.w_k=self.phi_init*self.q_k*tf.sin(self.theta_init)

                self.we=tf.concat([self.w_r,self.w_i,self.w_j,self.w_k],axis=1) #double check this

                self.q_entity_embedding= tf.Variable(name='entities',initial_value=self.we)

            with tf.variable_scope('predicate'):
                self.phi_init_r = tf.get_variable('phi_init_r',
                                                shape=[nb_predicates + 1, embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer(), trainable=False)

                self.theta_init_r = tf.get_variable('theta_init_r', shape=[nb_predicates + 1, embedding_size],
                                                  initializer=tf.random_uniform_initializer(minval=-pi,
                                                                                            maxval=pi,
                                                                                            dtype=tf.float32),
                                                  trainable=False)

                self.r_init_r = tf.get_variable('r_init_r', shape=[nb_predicates + 1, embedding_size],
                                              initializer=tf.random_uniform_initializer(minval=0,
                                                                                        maxval=0,
                                                                                        dtype=tf.float32),
                                              trainable=False)

                self.x_init_r= tf.get_variable('x_init_r', shape=[nb_predicates + 1, embedding_size],
                                              initializer=tf.random_uniform_initializer(minval=0,
                                                                                        maxval=1,
                                                                                        dtype=tf.float32),
                                              trainable=False)

                self.y_init_r= tf.get_variable('y_init_r', shape=[nb_predicates + 1, embedding_size],
                                              initializer=tf.random_uniform_initializer(minval=0,
                                                                                        maxval=1,
                                                                                        dtype=tf.float32),
                                              trainable=False)

                self.z_init_r= tf.get_variable('z_init_r', shape=[nb_predicates + 1, embedding_size],
                                              initializer=tf.random_uniform_initializer(minval=0,
                                                                                        maxval=1,
                                                                                        dtype=tf.float32),
                                              trainable=False)

                self.normaliser_init_r= tf.sqrt(
                    tf.square(self.x_init_r) + tf.square(self.y_init_r) + tf.square(self.z_init_r))

                self.q_i_r= tf.div(self.x_init_r, self.normaliser_init_r)
                self.q_j_r= tf.div(self.y_init_r, self.normaliser_init_r)
                self.q_k_r= tf.div(self.z_init_r, self.normaliser_init_r)

                self.w_r_r= self.phi_init_r* tf.cos(self.theta_init_r)
                self.w_i_r= self.phi_init_r* self.q_i_r* tf.sin(self.theta_init_r)
                self.w_j_r= self.phi_init_r* self.q_j_r* tf.sin(self.theta_init_r)
                self.w_k_r= self.phi_init_r* self.q_k_r* tf.sin(self.theta_init_r)

                self.wp = tf.concat([self.w_r_r, self.w_i_r, self.w_j_r, self.w_k_r], axis=1)  # double check this
                self.q_predicate_embedding = tf.Variable(name='predicates',initial_value=self.wp)


            self.h_s = tf.nn.embedding_lookup(self.q_entity_embedding, self.s_inputs)
            self.h_p = tf.nn.embedding_lookup(self.q_predicate_embedding, self.p_inputs)
            self.h_o = tf.nn.embedding_lookup(self.q_entity_embedding, self.o_inputs)


    def build_decoder(self):
        """
                                Constructs Decoder
        """

        with tf.variable_scope('Inference'):
            if self.score_func == 'Quaternator':
                logger.warning('Building Inference Network p(y|h) for Quaternator score function.')
                self.scores = models.Quaternator(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                           object_embeddings=self.h_o)

            else:
                raise NotImplementedError("Need to use another scoring function")

            self.p_x_i = tf.sigmoid(self.scores)



    def build_model(self, nb_entities, nb_predicates, embedding_size, optimizer):
        """
                        Construct full computation graph for Latent Fact Model
        """

        self.build_encoder(nb_entities, nb_predicates, embedding_size)
        self.build_decoder()

        # self.hinge_losses = tf.nn.relu(1 - self.scores * (2 * tf.cast(self.y_inputs, dtype=tf.float32) - 1))
        # self.loss = tf.reduce_sum(self.hinge_losses)

        self.loss = -tf.reduce_sum(tf.log(tf.where(condition=self.y_inputs, x=self.p_x_i, y=1 - self.p_x_i) + 1e-10))
        self.train_variables = tf.trainable_variables()

        self._setup_training(loss=self.loss, optimizer=optimizer)

    def train(self, test_triples, valid_triples, train_triples, embedding_size, no_batches, nb_epochs=500,
              filename='/home/'):
        """

                                Train and test model

        """



        all_triples = train_triples + valid_triples + test_triples
        util.index_gen = index.GlorotIndexGenerator()

        Xs = np.array([self.entity_to_idx[s] for (s, p, o) in train_triples], dtype=np.int32)
        Xp = np.array([self.predicate_to_idx[p] for (s, p, o) in train_triples], dtype=np.int32)
        Xo = np.array([self.entity_to_idx[o] for (s, p, o) in train_triples], dtype=np.int32)

        assert Xs.shape == Xp.shape == Xo.shape

        nb_samples = Xs.shape[0]
        nb_batches = no_batches
        batch_size = math.ceil(nb_samples / nb_batches)
        self.batch_size = batch_size
        batches = util.make_batches(self.nb_examples, batch_size)
        nb_versions = int(self.negsamples + 1)  # neg samples + original
        neg_subs = math.ceil(int(self.negsamples / 2))

        init_op = tf.global_variables_initializer()


        with tf.Session() as session:
            session.run(init_op)

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

                    for q in range((neg_subs)):  # Xs_batch[1::nb_versions] of subject needs to be corrupted
                        Xs_batch[(q + 1)::nb_versions] = util.index_gen(curr_batch_size, np.arange(self.nb_entities))
                        Xp_batch[(q + 1)::nb_versions] = Xp_shuf[batch_start:batch_end]
                        Xo_batch[(q + 1)::nb_versions] = Xo_shuf[batch_start:batch_end]

                    for q2 in range(neg_subs,
                                    (self.negsamples - neg_subs)):  # Xs_batch[1::nb_versions] of object needs to be corrupted
                        Xs_batch[(q2 + 1)::nb_versions] = Xs_shuf[batch_start:batch_end]
                        Xp_batch[(q2 + 1)::nb_versions] = Xp_shuf[batch_start:batch_end]
                        Xo_batch[(q2 + 1)::nb_versions] = util.index_gen(curr_batch_size, np.arange(self.nb_entities))

                    vec_neglabels = [int(1)] + ([int(0)] * (int(self.negsamples)))
                

                    loss_args = {
                        self.s_inputs: Xs_batch,
                        self.p_inputs: Xp_batch,
                        self.o_inputs: Xo_batch,
                        self.y_inputs: np.array(vec_neglabels * curr_batch_size)
                    }

                    # _, loss = session.run([self.training_step, self.loss],
                    #                             feed_dict=loss_args)

                    _, loss = session.run([self.training_step, self.loss],
                                          feed_dict=loss_args)


                    loss_values += [loss / (Xp_batch.shape[0] / nb_versions)]
                    total_loss_value += loss

                    counter += 1

                    # if self.projection:
                    if False:
                        for projection_step in self.projection_steps: #normalisation
                            session.run([projection_step])

                logger.warning('Epoch: {0}\t Loss: {1}'.format(epoch, util.stats(loss_values)))

                ##
                # Test
                ##

                if (epoch % 50) == 0:

                    for eval_name, eval_triples in [('valid', valid_triples), ('test', test_triples)]:

                        ranks_subj, ranks_obj = [], []
                        filtered_ranks_subj, filtered_ranks_obj = [], []

                        for _i, (s, p, o) in enumerate(eval_triples):
                            s_idx, p_idx, o_idx = self.entity_to_idx[s], self.predicate_to_idx[p], \
                                                  self.entity_to_idx[o]

                            #####

                            Xs_v = np.full(shape=(self.nb_entities,), fill_value=s_idx, dtype=np.int32)
                            Xp_v = np.full(shape=(self.nb_entities,), fill_value=p_idx, dtype=np.int32)
                            Xo_v = np.full(shape=(self.nb_entities,), fill_value=o_idx, dtype=np.int32)

                            feed_dict_corrupt_subj = {self.s_inputs: np.arange(self.nb_entities),
                                                      self.p_inputs: Xp_v,
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
                            logger.warning('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))

                            for k in [1, 3, 5, 10]:
                                hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                                logger.warning('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))
