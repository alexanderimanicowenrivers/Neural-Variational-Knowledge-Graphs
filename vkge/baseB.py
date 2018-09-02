# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf
import vkge.models as models
from vkge.training import constraints, corrupt, index
from vkge.training.util import make_batches
import logging
logger = logging.getLogger(__name__)
tfd = tf.contrib.distributions

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


class modelB:
    """
           model for testing the basic probabilistic aspects of the model

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

    def __init__(self, file_name, score_func='DistMult', static_mean=False, embedding_size=50, no_batches=10, mean_c=0.1,
                 epsilon=1e-3,negsamples=0,
                 alt_cost=False, dataset='wn18', sigma_alt=True, lr=0.1, alt_opt=True, projection=True,alt_updates=False,nosamps=1,alt_test='none'):

        seed=np.random.randint(100,size=1)[0]

        self.random_state = np.random.RandomState(seed)
        tf.set_random_seed(seed)

        logger.warn("\n \n Using Random Seed {} \n \n".format(seed))

        self.alt_test=alt_test
        self.sigma_alt = sigma_alt
        self.score_func=score_func
        self.alt_updates=alt_updates
        self.negsamples=1

        self.alt_opt=alt_opt
        self.abltaion_num=negsamples

        if self.score_func=='ComplEx':
            predicate_embedding_size = embedding_size*2
            entity_embedding_size = embedding_size*2
            var_max = np.log((1.0/embedding_size*2.0)+1e-10)

        else:
            predicate_embedding_size = embedding_size
            entity_embedding_size = embedding_size
            var_max = np.log((1.0/embedding_size*1.0)+1e-10)

        var_min = var_max


        self.static_mean = static_mean
        self.alt_cost = alt_cost

        if self.abltaion_num == 6:
            self.projection = False
        else:
            self.projection = projection

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
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)

        self.build_model(self.nb_entities, entity_embedding_size, self.nb_predicates, predicate_embedding_size,
                         optimizer, var_max, var_min)
        self.nb_epochs = 500


        self.train(nb_epochs=self.nb_epochs, test_triples=test_triples, valid_triples=valid_triples,entity_embedding_size=entity_embedding_size,
                   train_triples=train_triples, no_batches=int(no_batches)  , filename=str(file_name))

    def distribution_scale(self, log_sigma_square):
        """
                        Returns the scale (std dev) from embeddings for tensorflow distributions MultivariateNormalDiag function
                """

        scale = tf.sqrt(tf.exp(log_sigma_square))

        return scale

    def make_prior(self,code_size):

        """
                        Returns the prior on embeddings for tensorflow distributions MultivariateNormalDiag function

                        (1) Alt: N(0,1/code_size)
                        (2) N(0,1)
                """

        if self.alt_opt: #alternative prior 0,1/embeddings variance
            loc = tf.zeros(code_size)
            scale = tf.sqrt(tf.divide(tf.ones(code_size),code_size))

        else:
            loc = tf.zeros(code_size)
            scale = tf.ones(code_size)

        return tfd.MultivariateNormalDiag(loc, scale)

    def build_encoder(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, var_max,
                      var_min):
        """
                                Constructs encoder
        """
        logger.warn('Building Inference Networks q(h_x | x) ..{}'.format(self.score_func))

        init2 = np.round(var_max,decimals=2)

        with tf.variable_scope('Encoder'):

            with tf.variable_scope('entity'):
                with tf.variable_scope('mu'):

                    if self.abltaion_num==5:

                        self.entity_embedding_mean = tf.get_variable('entities',
                                                                     shape=[nb_entities + 1, entity_embedding_size],
                                                                     initializer=tf.random_uniform_initializer(
                                                                         minval=-0.001,
                                                                         maxval=0.001,
                                                                         dtype=tf.float32))

                    else:

                        self.entity_embedding_mean = tf.get_variable('entities', shape=[nb_entities + 1, entity_embedding_size],
                                                                 initializer=tf.contrib.layers.xavier_initializer())


                with tf.variable_scope('sigma'):

                    if self.abltaion_num==4:

                        self.entity_embedding_sigma = tf.get_variable('entities_sigma',
                                                                      shape=[nb_entities + 1, entity_embedding_size],
                                                                      initializer=tf.random_uniform_initializer(
                                                                          minval=0, maxval=init2, dtype=tf.float32),
                                                                      dtype=tf.float32)


                    else:

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
                with tf.variable_scope('mu'):
                    if self.abltaion_num == 5:

                        self.predicate_embedding_mean = tf.get_variable('predicates',
                                                                        shape=[nb_predicates + 1, predicate_embedding_size],
                                                                        initializer=tf.random_uniform_initializer(
                                                                            minval=-0.001,
                                                                            maxval=0.001,
                                                                            dtype=tf.float32))


                    else:

                         self.predicate_embedding_mean = tf.get_variable('predicates',
                                                                shape=[nb_predicates + 1, predicate_embedding_size],
                                                                    initializer=tf.contrib.layers.xavier_initializer())


                with tf.variable_scope('sigma'):

                    if self.abltaion_num==4:


                        self.predicate_embedding_sigma = tf.get_variable('predicate_sigma',
                                                             shape=[nb_predicates + 1,
                                                                    predicate_embedding_size],
                                                             initializer=tf.random_uniform_initializer(
                                                                 minval=0, maxval=init2, dtype=tf.float32),
                                                             dtype=tf.float32)

                    else:

                        self.predicate_embedding_sigma = tf.get_variable('predicate_sigma',
                                                                         shape=[nb_predicates + 1,
                                                                                predicate_embedding_size],
                                                                         initializer=tf.random_uniform_initializer(
                                                                             minval=init2, maxval=init2,
                                                                             dtype=tf.float32),
                                                                         dtype=tf.float32)


                self.mu_p = tf.nn.embedding_lookup(self.predicate_embedding_mean, self.p_inputs)
                self.log_sigma_sq_p = tf.nn.embedding_lookup(self.predicate_embedding_sigma, self.p_inputs)



    def build_decoder(self):
        """
                                Constructs Decoder
        """
        logger.warn('Building Inference Network p(y|h) for {} score function.'.format(self.score_func))

        with tf.variable_scope('Inference'):

            self.h_s = tfd.MultivariateNormalDiag(self.mu_s, self.distribution_scale(self.log_sigma_sq_s)).sample()
            self.h_p = tfd.MultivariateNormalDiag(self.mu_p, self.distribution_scale(self.log_sigma_sq_p)).sample()
            self.h_o = tfd.MultivariateNormalDiag(self.mu_o, self.distribution_scale(self.log_sigma_sq_o)).sample()

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

    def _setup_training(self, loss, optimizer=tf.train.AdamOptimizer, l2=0.0, clip_op=None, clip=False):
        global_step = tf.train.get_global_step()
        if global_step is None:
            global_step = tf.train.create_global_step()

        gradients = optimizer.compute_gradients(loss=loss)
        train_op = optimizer.apply_gradients(gradients, global_step)

        self.loss = loss
        self.training_step = train_op

        return loss, train_op

    def build_model(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, optimizer,
                    var_max, pred_sig):
        """
                        Construct full computation graph
        """
        self.noise = tf.placeholder(tf.float32, shape=[None,entity_embedding_size])
        self.idx_pos = tf.placeholder(tf.int32, shape=[None])
        self.idx_neg = tf.placeholder(tf.int32, shape=[None])

        self.no_samples = tf.placeholder(tf.int32)
        self.s_inputs = tf.placeholder(tf.int32, shape=[None])
        self.p_inputs = tf.placeholder(tf.int32, shape=[None])
        self.o_inputs = tf.placeholder(tf.int32, shape=[None])
        self.y_inputs = tf.placeholder(tf.bool, shape=[None])
        self.KL_discount = tf.placeholder(tf.float32)
        self.BernoulliSRescale = tf.placeholder(tf.float32)  #

        self.build_encoder(nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, var_max,
                           pred_sig)
        self.build_decoder()

        self.y_pos = tf.gather(self.y_inputs, self.idx_pos)
        self.y_neg = tf.gather(self.y_inputs, self.idx_neg)

        self.p_x_i_pos = tf.gather(self.p_x_i, self.idx_pos)
        self.p_x_i_neg = tf.gather(self.p_x_i, self.idx_neg)

        # Negative log likelihood

        self.g_objective_p = -tf.reduce_sum(
            tf.log(tf.where(condition=self.y_pos, x=self.p_x_i_pos, y=1 - self.p_x_i_pos) + 1e-10))
        self.g_objective_n = -tf.reduce_sum((
            tf.log(tf.where(condition=self.y_neg, x=self.p_x_i_neg, y=1 - self.p_x_i_neg) + 1e-10)))


        #KL

        self.mu_s_ps = tf.gather(self.mu_s, self.idx_pos, axis=0)
        self.mu_o_ps = tf.gather(self.mu_o, self.idx_pos, axis=0)
        self.mu_p_ps = tf.gather(self.mu_p, self.idx_pos, axis=0)
        #
        self.log_sigma_sq_s_ps = tf.gather(self.log_sigma_sq_s, self.idx_pos, axis=0)
        self.log_sigma_sq_o_ps = tf.gather(self.log_sigma_sq_o, self.idx_pos, axis=0)
        self.log_sigma_sq_p_ps = tf.gather(self.log_sigma_sq_p, self.idx_pos, axis=0)

        self.mu_all_ps = tf.concat(axis=0, values=[self.mu_s_ps, self.mu_o_ps, self.mu_p_ps])
        self.log_sigma_ps = tf.concat(axis=0,
                                      values=[self.log_sigma_sq_s_ps, self.log_sigma_sq_o_ps, self.log_sigma_sq_p_ps])
        #

        # negative samples

        self.mu_s_ns = tf.gather(self.mu_s, self.idx_neg, axis=0)
        self.mu_o_ns = tf.gather(self.mu_o, self.idx_neg, axis=0)
        self.mu_p_ns = tf.gather(self.mu_p, self.idx_neg, axis=0)
        #
        self.log_sigma_sq_s_ns = tf.gather(self.log_sigma_sq_s, self.idx_neg, axis=0)
        self.log_sigma_sq_o_ns = tf.gather(self.log_sigma_sq_o, self.idx_neg, axis=0)
        self.log_sigma_sq_p_ns = tf.gather(self.log_sigma_sq_p, self.idx_neg, axis=0)

        self.mu_all_ns = tf.concat(axis=0, values=[self.mu_s_ns, self.mu_o_ns, self.mu_p_ns])
        self.log_sigma_ns = tf.concat(axis=0,
                                      values=[self.log_sigma_sq_s_ns, self.log_sigma_sq_o_ns, self.log_sigma_sq_p_ns])
        #

        prior = self.make_prior(code_size=entity_embedding_size)
        pos_posterior=tfd.MultivariateNormalDiag(self.mu_all_ps, self.distribution_scale(self.log_sigma_ps))
        neg_posterior=tfd.MultivariateNormalDiag(self.mu_all_ns, self.distribution_scale(self.log_sigma_ns))

        self.e_objective1=tf.reduce_sum(tfd.kl_divergence(pos_posterior, prior))
        self.e_objective2=tf.reduce_sum(tfd.kl_divergence(neg_posterior, prior))

        self.nelbo = (self.g_objective_p+self.e_objective1) + (self.g_objective_n+self.e_objective2* self.KL_discount)*self.BernoulliSRescale  #if reduce sum

        # Negative ELBO


        self._setup_training(loss=self.nelbo,optimizer=optimizer)

    def stats(self, values):
        """
                                Return mean and variance statistics
        """
        return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))

    def train(self, test_triples, valid_triples, train_triples, entity_embedding_size,no_batches, session=0, nb_epochs=500, unit_cube=True,
              filename='/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/'):
        """

                                Train and test model

        """

        all_triples = train_triples + valid_triples + test_triples

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

        self.negsamples = int(self.negsamples)

        nb_versions = int(self.negsamples + 1)  # neg samples + original
        projection_steps = [constraints.unit_sphere_logvar(self.predicate_embedding_sigma, norm=1.0),constraints.unit_sphere_logvar(self.entity_embedding_sigma, norm=1.0)]

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

            neg_subs = math.ceil(int(self.negsamples / 2))

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

                    for q in range((neg_subs)): # Xs_batch[1::nb_versions] needs to be corrupted
                        Xs_batch[(q+1)::nb_versions] = index_gen(curr_batch_size, np.arange(self.nb_entities))
                        Xp_batch[(q+1)::nb_versions] = Xp_shuf[batch_start:batch_end]
                        Xo_batch[(q+1)::nb_versions] = Xo_shuf[batch_start:batch_end]

                    for q2 in range(neg_subs,(self.negsamples-neg_subs)): # Xs_batch[1::nb_versions] needs to be corrupted
                        Xs_batch[(q2+1)::nb_versions] = Xs_shuf[batch_start:batch_end]
                        Xp_batch[(q2+1)::nb_versions] = Xp_shuf[batch_start:batch_end]
                        Xo_batch[(q2+1)::nb_versions] = index_gen(curr_batch_size, np.arange(self.nb_entities))

                    vec_neglabels=[int(1)]+([int(0)]*(int(self.negsamples)))

                    BS=(2.0*(self.nb_entities-1)/self.negsamples)

                    if (epoch >= (nb_epochs/10)) or (self.abltaion_num == 2 ):
                        kl_linwarmup = (pi[counter])

                        if self.abltaion_num == 1:
                            kl_linwarmup = (1.0 / nb_batches)

                    else:
                        kl_linwarmup = 0.0



                    if self.abltaion_num==3:
                        BS=1.0


                    loss_args = {
                        self.KL_discount: kl_linwarmup,
                        self.s_inputs: Xs_batch,
                        self.p_inputs: Xp_batch,
                        self.o_inputs: Xo_batch,
                        self.y_inputs: np.array(vec_neglabels * curr_batch_size)
                        ,self.BernoulliSRescale: BS
                        , self.idx_pos: np.arange(curr_batch_size),
                        self.idx_neg: np.arange(curr_batch_size, curr_batch_size * nb_versions)
                    }

                    _, elbo_value = session.run([ self.training_step, self.nelbo],
                                                             feed_dict=loss_args)

                    loss_values += [elbo_value / (Xp_batch.shape[0] / nb_versions)]
                    total_loss_value += elbo_value

                    counter += 1

                    if ((self.projection) and (epoch<=nb_epochs)): #so you do not project before evaluation

                        for projection_step in projection_steps:
                            session.run([projection_step])

                logger.warn('Epoch: {0}\t Negative ELBO: {1}'.format(epoch, self.stats(loss_values)))

                ##
                # Test
                ##

                if (epoch % 50) == 0:

                    for eval_name,eval_triples in [('valid',valid_triples),('test',test_triples)]:

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
