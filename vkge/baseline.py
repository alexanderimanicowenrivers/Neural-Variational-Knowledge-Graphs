# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf
from vkge.training import constraints, index
from vkge.training import util as util
import vkge.models as models
import logging
import tensorflow_probability as tfp

tfd = tfp.distributions
logger = logging.getLogger(__name__)


class baseline:
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

    def __init__(self, file_name, score_func='ComplEx', embedding_size=200, no_batches=10, distribution='normal',
                 epsilon=1e-3, negsamples=5, dataset='wn18', lr=0.001, projection=False):

        self.nb_epochs = 500

        seed = np.random.randint(100, size=1)[0]
        self.random_state = np.random.RandomState(seed)
        tf.set_random_seed(seed)
        logger.warning("\n \n Using Random Seed {} \n \n".format(seed))

        self.score_func = score_func
        self.negsamples = int(negsamples)
        self.distribution = distribution
        self.projection = projection
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)

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



        ##############
        self.build_model(self.nb_entities, self.nb_predicates, embedding_size, optimizer)

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

        with tf.variable_scope('encoder'):
            self.entity_embedding= tf.get_variable('entities',
                                                         shape=[nb_entities + 1, embedding_size],
                                                         initializer=tf.contrib.layers.xavier_initializer())
            self.predicate_embedding = tf.get_variable('predicates',
                                                              shape=[nb_predicates + 1, embedding_size],
                                                              initializer=tf.contrib.layers.xavier_initializer())

            self.h_s = tf.nn.embedding_lookup(self.entity_embedding, self.s_inputs)
            self.h_p = tf.nn.embedding_lookup(self.predicate_embedding, self.p_inputs)
            self.h_o = tf.nn.embedding_lookup(self.entity_embedding, self.o_inputs)

    def build_decoder(self):
        """
                                Constructs Decoder
        """

        with tf.variable_scope('Inference'):
            if self.score_func == 'DistMult':
                logger.warning('Building Inference Network p(y|h) for DistMult score function.')
                model = models.Quaternator(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                           object_embeddings=self.h_o)

            else:
                logger.warning('Building Inference Network p(y|h) for ComplEx score function.')
                model = models.ComplexModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
                                            object_embeddings=self.h_o)


            self.scores = model()
            self.p_x_i = tf.sigmoid(self.scores)



    def build_model(self, nb_entities, nb_predicates, embedding_size, optimizer):
        """
                        Construct full computation graph for Latent Fact Model
        """
        self.s_inputs = tf.placeholder(tf.int32, shape=[None])
        self.p_inputs = tf.placeholder(tf.int32, shape=[None])
        self.o_inputs = tf.placeholder(tf.int32, shape=[None])
        self.y_inputs = tf.placeholder(tf.bool, shape=[None])

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

        projection_steps = [constraints.unit_sphere_logvar(self.predicate_embedding, norm=1.0),
                            constraints.unit_sphere_logvar(self.entity_embedding, norm=1.0)]
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

                    for q in range((neg_subs)):  # Xs_batch[1::nb_versions] needs to be corrupted
                        Xs_batch[(q + 1)::nb_versions] = util.index_gen(curr_batch_size, np.arange(self.nb_entities))
                        Xp_batch[(q + 1)::nb_versions] = Xp_shuf[batch_start:batch_end]
                        Xo_batch[(q + 1)::nb_versions] = Xo_shuf[batch_start:batch_end]

                    for q2 in range(neg_subs,
                                    (self.negsamples - neg_subs)):  # Xs_batch[1::nb_versions] needs to be corrupted
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

                    _, loss = session.run([self.training_step, self.loss],
                                                feed_dict=loss_args)

                    loss_values += [loss / (Xp_batch.shape[0] / nb_versions)]
                    total_loss_value += loss

                    counter += 1

                    if self.projection:

                        for projection_step in projection_steps:
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
