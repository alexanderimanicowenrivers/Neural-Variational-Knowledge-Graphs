# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf
from vkge.training import constraints, index
from vkge.training import util as util 
import logging
from hyperspherical_vae.distributions import VonMisesFisher
import tensorflow_probability as tfp
tfd =  tfp.distributions
logger = logging.getLogger(__name__)

class modelA:
    """
           Model A (latents generated independently) 

            
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
                 epsilon=1e-3,negsamples=5, dataset='wn18', lr=0.001, alt_prior=False, projection=False):


        
        self.nb_epochs = 500

        seed=np.random.randint(100,size=1)[0]
        self.random_state = np.random.RandomState(seed)
        tf.set_random_seed(seed)
        logger.warning("\n \n Using Random Seed {} \n \n".format(seed))

        self.score_func=score_func
        self.negsamples=int(negsamples)
        self.distribution=distribution
        self.alt_prior=alt_prior
        self.projection=projection
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
        ############################
        
        self.build_modelA(self.nb_entities, self.nb_predicates, embedding_size,optimizer)

        self.train(nb_epochs=self.nb_epochs, test_triples=test_triples, valid_triples=valid_triples,embedding_size=embedding_size,
                   train_triples=train_triples, no_batches=int(no_batches)  , filename=str(file_name))
        
    def _setup_training(self, loss, optimizer=tf.train.AdamOptimizer):
        global_step = tf.train.get_global_step()
        if global_step is None:
            global_step = tf.train.create_global_step()

        gradients = optimizer.compute_gradients(loss=loss)
        train_op = optimizer.apply_gradients(gradients, global_step)

        self.loss = loss
        self.training_step = train_op

        return loss, train_op

    def build_encoder(self, nb_entities, nb_predicates, embedding_size):
        """
                                Constructs encoder
        """
        logger.warning('Building Inference Networks q(h_x | x) ..{}'.format(self.score_func))

        with tf.variable_scope('Encoder'):

            self.entity_embedding_mean,self.entity_embedding_sigma=util.make_latent_variables(meaninit='xavier', siginit='constant', nb_variables=nb_entities, embedding_size=embedding_size, distribution=self.distribution,vtype='entities')
            self.predicate_embedding_mean,self.predicate_embedding_sigma=util.make_latent_variables(meaninit='xavier', siginit='constant', nb_variables=nb_predicates, embedding_size=embedding_size, distribution=self.distribution,vtype='predicates')

            self.mu_s = tf.nn.embedding_lookup(self.entity_embedding_mean, self.s_inputs)
            self.log_sigma_sq_s = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.s_inputs)
            self.mu_o = tf.nn.embedding_lookup(self.entity_embedding_mean, self.o_inputs)
            self.log_sigma_sq_o = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.o_inputs)
            self.mu_p = tf.nn.embedding_lookup(self.predicate_embedding_mean, self.p_inputs)
            self.log_sigma_sq_p = tf.nn.embedding_lookup(self.predicate_embedding_sigma, self.p_inputs)

    def build_decoder(self):
        """
                                Constructs Decoder
        """
        logger.warning('Building Inference Network p(y|h) for {} score function.'.format(self.score_func))

        with tf.variable_scope('Inference'):

            self.q_s, self.q_p, self.q_o=util.get_latent_distributions(self.distribution, self.mu_s, self.mu_p,self.mu_o, self.log_sigma_sq_s, self.log_sigma_sq_p, self.log_sigma_sq_o)

            self.h_s = self.q_s.sample()
            self.h_p = self.q_p.sample()
            self.h_o = self.q_o.sample()

            model, model_test=util.get_scoring_func(self.score_func, self.distribution, self.h_s, self.h_p, self.h_o, self.mu_s, self.mu_p, self.mu_o)

            self.scores = model()
            self.scores_test = model_test()
            self.p_x_i = tf.sigmoid(self.scores)
            self.p_x_i_test = tf.sigmoid(self.scores_test)

    def build_modelA(self, nb_entities, nb_predicates, embedding_size, optimizer):
        """
                        Construct full computation graph for Model A
        """

        ############## placeholders

        self.noise = tf.placeholder(tf.float32, shape=[None, embedding_size])
        self.idx_pos = tf.placeholder(tf.int32, shape=[None])
        self.idx_neg = tf.placeholder(tf.int32, shape=[None])

        self.no_samples = tf.placeholder(tf.int32)
        self.s_inputs = tf.placeholder(tf.int32, shape=[None])
        self.p_inputs = tf.placeholder(tf.int32, shape=[None])
        self.o_inputs = tf.placeholder(tf.int32, shape=[None])
        self.y_inputs = tf.placeholder(tf.bool, shape=[None])
        self.KL_discount = tf.placeholder(tf.float32)
        self.ELBOBS = tf.placeholder(tf.float32)

        ##############

        self.build_encoder(nb_entities, nb_predicates, embedding_size)
        self.build_decoder()

        ############## ##############
        ##############  #Loss
        ############## ##############

        self.y_pos = tf.gather(self.y_inputs, self.idx_pos)
        self.y_neg = tf.gather(self.y_inputs, self.idx_neg)

        self.p_x_i_pos = tf.gather(self.p_x_i, self.idx_pos)
        self.p_x_i_neg = tf.gather(self.p_x_i, self.idx_neg)

        # Negative reconstruction loss

        self.reconstruction_loss_p = -tf.reduce_sum(
            tf.log(tf.where(condition=self.y_pos, x=self.p_x_i_pos, y=1 - self.p_x_i_pos) + 1e-10))
        self.reconstruction_loss_n = -tf.reduce_sum((
            tf.log(tf.where(condition=self.y_neg, x=self.p_x_i_neg, y=1 - self.p_x_i_neg) + 1e-10)))
        self.nreconstruction_loss = self.reconstruction_loss_p + self.reconstruction_loss_n * self.ELBOBS  # if reduce sum

        prior = util.make_prior(code_size=embedding_size, distribution=self.distribution, alt_prior=self.alt_prior)

        if self.distribution == 'normal':
            # KL divergence between normal approximate posterior and prior

            entity_posterior = tfd.MultivariateNormalDiag(self.entity_embedding_mean,
                                                          util.distribution_scale(self.entity_embedding_sigma))
            predicate_posterior = tfd.MultivariateNormalDiag(self.predicate_embedding_mean,
                                                             util.distribution_scale(self.predicate_embedding_sigma))
            self.kl1 = tf.reduce_sum(tfd.kl_divergence(entity_posterior, prior))
            self.kl2 = tf.reduce_sum(tfd.kl_divergence(predicate_posterior, prior))

        elif self.distribution == 'vmf':
            # KL divergence between vMF approximate posterior and uniform hyper-spherical prior

            entity_posterior = VonMisesFisher(self.entity_embedding_mean,
                                              util.distribution_scale(self.entity_embedding_sigma) + 1)
            predicate_posterior = VonMisesFisher(self.predicate_embedding_mean,
                                                 util.distribution_scale(self.predicate_embedding_sigma) + 1)
            kl1 = entity_posterior.kl_divergence(prior)
            kl2 = predicate_posterior.kl_divergence(prior)
            self.kl1 = tf.reduce_sum(kl1)
            self.kl2 = tf.reduce_sum(kl2)

        else:
            raise NotImplemented

        self.nkl = (self.kl1 + self.kl2) * self.KL_discount

        # Negative ELBO

        self.nelbo = self.nkl + self.nreconstruction_loss

        self._setup_training(loss=self.nelbo, optimizer=optimizer)

    def train(self, test_triples, valid_triples, train_triples, embedding_size,no_batches, nb_epochs=500,
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
        self.batch_size=batch_size
        batches = util.make_batches(self.nb_examples, batch_size)
        nb_versions = int(self.negsamples + 1)  # neg samples + original
        neg_subs = math.ceil(int(self.negsamples / 2))
        pi=util.make_compression_cost(nb_batches)

        init_op = tf.global_variables_initializer()
        projection_steps = [constraints.unit_sphere_logvar(self.predicate_embedding_sigma, norm=1.0),constraints.unit_sphere_logvar(self.entity_embedding_sigma, norm=1.0)]

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

                    for q in range((neg_subs)): # Xs_batch[1::nb_versions] needs to be corrupted
                        Xs_batch[(q+1)::nb_versions] = util.index_gen(curr_batch_size, np.arange(self.nb_entities))
                        Xp_batch[(q+1)::nb_versions] = Xp_shuf[batch_start:batch_end]
                        Xo_batch[(q+1)::nb_versions] = Xo_shuf[batch_start:batch_end]

                    for q2 in range(neg_subs,(self.negsamples-neg_subs)): # Xs_batch[1::nb_versions] needs to be corrupted
                        Xs_batch[(q2+1)::nb_versions] = Xs_shuf[batch_start:batch_end]
                        Xp_batch[(q2+1)::nb_versions] = Xp_shuf[batch_start:batch_end]
                        Xo_batch[(q2+1)::nb_versions] = util.index_gen(curr_batch_size, np.arange(self.nb_entities))

                    vec_neglabels=[int(1)]+([int(0)]*(int(self.negsamples)))

                    BS=(2.0*(self.nb_entities-1)/self.negsamples)

                    if (epoch >= (nb_epochs/10)): #linear warmup

                        kl_linwarmup = (pi[counter])
                        #kl_linwarmup = (1.0 / nb_batches)

                    else:
                        kl_linwarmup = 0.0



                    if False: #turn Bernoulli Sample Off
                        BS=1.0


                    loss_args = {
                        self.KL_discount: kl_linwarmup,
                        self.s_inputs: Xs_batch,
                        self.p_inputs: Xp_batch,
                        self.o_inputs: Xo_batch,
                        self.y_inputs: np.array(vec_neglabels * curr_batch_size)
                        ,self.ELBOBS: BS
                        , self.idx_pos: np.arange(curr_batch_size),
                        self.idx_neg: np.arange(curr_batch_size, curr_batch_size * nb_versions)
                    }

                    _, elbo_value = session.run([ self.training_step, self.nelbo],
                                                             feed_dict=loss_args)

                    loss_values += [elbo_value / (Xp_batch.shape[0] / nb_versions)]
                    total_loss_value += elbo_value

                    counter += 1

                    if (self.projection):

                        for projection_step in projection_steps:
                            session.run([projection_step])

                logger.warning('Epoch: {0}\t Negative ELBO: {1}'.format(epoch, util.stats(loss_values)))

                ##
                # Test
                ##

                if (epoch % 100) == 0:

                    for eval_name,eval_triples in [('valid',valid_triples),('test',test_triples)]:

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
                            logger.warning('[{}] {} Mean Rank: {}'.format(eval_name, setting_name, mean_rank))

                            for k in [1, 3, 5, 10]:
                                hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
                                logger.warning('[{}] {} Hits@{}: {}'.format(eval_name, setting_name, k, hits_at_k))
