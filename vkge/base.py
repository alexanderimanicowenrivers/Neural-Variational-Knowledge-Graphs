# -*- coding: utf-8 -*-

import math
import numpy as np
import tensorflow as tf
from random import randint

from vkge.knowledgebase import Fact, KnowledgeBaseParser

import vkge.models as models
from vkge.training import constraints, corrupt, index
from vkge.training.util import make_batches
import vkge.io as io


# new

import logging
logger = logging.getLogger(__name__)


class VKGE:
    """
        composition of literature for architecture search


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
        @type opt_type: str: ['adam','rms']
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
                ent_sigma = (np.log(ent_sig)**2) #old sigma

        pred_sigma = ent_sigma #adjust for correct format for model input
        predicate_embedding_size = embedding_size
        entity_embedding_size = embedding_size

        triples = io.read_triples("data/wn18/wordnet-mlj12-train.txt")  # choose dataset
        test_triples = io.read_triples("data/wn18/wordnet-mlj12-test.txt")
        self.decay_kl=decay_kl
        self.static_pred=static_pred
        self.random_state = np.random.RandomState(0)
        self.GPUMode = GPUMode
        self.alt_cost = alt_cost
        self.nb_examples = len(triples)
        self.static_mean= static_mean
        self.alt_updates=alt_updates
        self.tensorboard=tensorboard
        self.projection=projection
        self.opt_type=opt_type
        logger.warn('Parsing the facts in the Knowledge Base ..')

        # logger.warn('Parsing the facts in the Knowledge Base ..')
        self.facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]

        self.test_facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in test_triples]

        self.test_parser = KnowledgeBaseParser(self.test_facts)
        self.parser = KnowledgeBaseParser(self.facts)

        ##### for test time ######
        all_triples = triples + test_triples
        entity_set = {s for (s, p, o) in all_triples} | {o for (s, p, o) in all_triples}
        predicate_set = {p for (s, p, o) in all_triples}
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(predicate_set))}
        self.nb_entities, self.nb_predicates = len(entity_set), len(predicate_set)

        ###########
        # generate 4 entity and 2 predicate variables to keep track of in Tensorboard

        self.var1 = randint(0, self.nb_entities-1)
        self.var2 = randint(0, self.nb_entities-1)
        self.var3 = randint(0, self.nb_entities-1)
        self.var4 = randint(0, self.nb_entities-1)
        self.var5 = randint(0, self.nb_predicates-1)
        self.var6 = randint(0, self.nb_predicates-1)

        ######### Print sample ID's
        logger.warn("Entity Sample1 is {} ..".format(self.var1))
        logger.warn("Entity Sample2 is {} ..".format(self.var2))
        logger.warn("Entity Sample3 is {} ..".format(self.var3))
        logger.warn("Entity Sample4 is {} ..".format(self.var4))

        logger.warn("Predicate Sample1 is {} ..".format(self.var5))
        logger.warn("Predicate Sample1 is {} ..".format(self.var6))

        ############################

        # if opt_type == 'rms':
        #     optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=eps)
        # elif opt_type == 'adam':

        if opt=='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2, epsilon=eps)
        else:
            optimizer=tf.train.AdagradOptimizer(learning_rate=lr) #original KG
        # else:

        self.build_model(self.nb_entities, entity_embedding_size, self.nb_predicates, predicate_embedding_size,
                         optimizer,
                         ent_sigma, pred_sigma)
        self.nb_epochs=1500
        self.decaykl= np.linspace(0, 1, self.nb_epochs)

        self.train(nb_epochs=self.nb_epochs, test_triples=test_triples, all_triples=all_triples,batch_size=batch_s,filename=file_name)

    @staticmethod
    def input_parameters(inputs, parameters_layer):
        """
                    Separates distribution parameters from embeddings
        """
        parameters = tf.nn.embedding_lookup(parameters_layer, inputs)
        mu, log_sigma_square = tf.split(value=parameters, num_or_size_splits=2, axis=1)
        return mu, log_sigma_square

    def sample_embedding(self,mu, log_sigma_square):
        """
                Samples from embeddings
        """

        if self.sigma_alt :
            sigma = tf.log(1+tf.exp(log_sigma_square))
        else:
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

        self.KL_discount = tf.placeholder(tf.float32)  # starts at 0.5

        # Kullback Leibler divergence

        self.e_objective = 0.0
        self.e_objective1 = 0.0
        self.e_objective2 = 0.0
        self.e_objective3 = 0.0

        self.e_objective1 -= 0.5 * tf.reduce_sum(
            1. + self.log_sigma_sq_s - tf.square(self.mu_s) - tf.exp(self.log_sigma_sq_s))
        self.e_objective2 -= 0.5 * tf.reduce_sum(
            1. + self.log_sigma_sq_p - tf.square(self.mu_p) - tf.exp(self.log_sigma_sq_p))
        self.e_objective3 -= 0.5 * tf.reduce_sum(
            1. + self.log_sigma_sq_o - tf.square(self.mu_o) - tf.exp(self.log_sigma_sq_o))

        # Adjust through linear/non linear KL loss

        if self.decay_kl:
            self.e_objective1 = self.e_objective1 * self.KL_discount * self.decaykl
            self.e_objective2 = self.e_objective1 * self.KL_discount * self.decaykl
            self.e_objective3 = self.e_objective1 * self.KL_discount * self.decaykl

        else:
            self.e_objective1 = self.e_objective1 * self.KL_discount
            self.e_objective2 = self.e_objective1 * self.KL_discount
            self.e_objective3 = self.e_objective1 * self.KL_discount

        # Total Loss

        self.e_objective = self.e_objective1+self.e_objective2+self.e_objective3
        # Log likelihood
        if self.opt_type=='ml':
            self.g_objective = -tf.reduce_sum(
                tf.log(tf.where(condition=self.y_inputs, x=self.p_x_i, y=1 - self.p_x_i) + 1e-4))
        else:
            self.hinge_losses = tf.nn.relu(5 - self.scores * (2 * tf.cast(self.y_inputs,dtype=tf.float32) - 1))
            self.g_objective = tf.reduce_sum(self.hinge_losses)

        self.elbo = self.g_objective + self.e_objective



        self.training_step = optimizer.minimize(self.elbo)

        self.training_step1 = optimizer.minimize(self.e_objective1)
        self.training_step2 = optimizer.minimize(self.e_objective2)
        self.training_step3 = optimizer.minimize(self.e_objective3)
        self.training_step4 = optimizer.minimize(self.g_objective)


        if self.tensorboard:

            _ = tf.summary.scalar("total e loss", self.e_objective)
            _ = tf.summary.scalar("total e subject loss", self.e_objective1)
            _ = tf.summary.scalar("total e predicate loss", self.e_objective2)
            _ = tf.summary.scalar("total e object loss", self.e_objective3)
            _ = tf.summary.scalar("g loss", self.g_objective)

            _ = tf.summary.scalar("total loss", self.elbo)

    def variable_summaries(self,var):
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

    def build_encoder(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, ent_sig,
                      pred_sig):
        """
                                Constructs Encoder
        """
        logger.warn('Building Inference Networks q(h_x | x) ..')

        ## Later try sigma init to normal distribution also

        # logger.warn('Building Inference Networks q(h_x | x) ..')

        with tf.variable_scope("encoder"):

            if self.static_mean:

                self.entity_embedding_mean = tf.get_variable('entities_mean',
                                                         shape=[nb_entities + 1, entity_embedding_size],
                                                         initializer=tf.zeros_initializer(), dtype=tf.float32,trainable=False)

                self.predicate_embedding_mean = tf.get_variable('predicate_mean',
                                                                shape=[nb_predicates + 1, predicate_embedding_size],
                                                                initializer=tf.zeros_initializer(), dtype=tf.float32,
                                                                trainable=False)



            else:

                self.entity_embedding_mean = tf.get_variable('entities_mean',
                                                             shape=[nb_entities + 1, entity_embedding_size],
                                                             initializer=tf.initializers.random_normal(), dtype=tf.float32,trainable=True)


                self.predicate_embedding_mean = tf.get_variable('predicate_mean',
                                                                shape=[nb_predicates + 1, predicate_embedding_size],
                                                                initializer=tf.initializers.random_normal(), dtype=tf.float32,
                                                                trainable=True)

            if pred_sig==-1: #flag for initialising the sigmas to random normal distributions


                logger.warn("pred_sig == -1")
                logger.warn("tf.initializers.random_normal() for entity and predicate mean")

                self.entity_embedding_sigma = tf.get_variable('entities_sigma',
                                                             shape=[nb_entities + 1, entity_embedding_size],
                                                             initializer=tf.initializers.random_normal(), dtype=tf.float32,trainable=True)

                self.predicate_embedding_sigma = tf.get_variable('predicate_sigma',
                                                                shape=[nb_predicates + 1, predicate_embedding_size],
                                                                initializer=tf.initializers.random_normal(), dtype=tf.float32,trainable=True)

            else:

                self.entity_embedding_sigm = tf.get_variable(shape=[nb_entities + 1, entity_embedding_size],
                                                             initializer=tf.ones_initializer(), dtype=tf.float32,name='entities_sigm')

                self.entity_embedding_sigma = tf.Variable(self.entity_embedding_sigm.initialized_value() * ent_sig,
                                                          dtype=tf.float32,name='entities_sigma')

                self.predicate_embedding_sigm = tf.get_variable(name='predicate_sigm',shape=[nb_predicates + 1, predicate_embedding_size],
                                                                initializer=tf.ones_initializer(), dtype=tf.float32)

                self.predicate_embedding_sigma = tf.Variable(
                    self.predicate_embedding_sigm.initialized_value() * pred_sig, dtype=tf.float32,name='predicate_sigma')




            self.mu_s = tf.nn.embedding_lookup(self.entity_embedding_mean, self.s_inputs)
            self.log_sigma_sq_s = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.s_inputs)
            self.h_s = self.sample_embedding(self.mu_s, self.log_sigma_sq_s)

            self.mu_o = tf.nn.embedding_lookup(self.entity_embedding_mean, self.o_inputs)
            self.log_sigma_sq_o = tf.nn.embedding_lookup(self.entity_embedding_sigma, self.o_inputs)
            self.h_o = self.sample_embedding(self.mu_o, self.log_sigma_sq_o)

            if self.static_pred:
                self.h_p = tf.nn.embedding_lookup(self.predicate_embedding_mean, self.p_inputs)

            else:
                self.mu_p = tf.nn.embedding_lookup(self.predicate_embedding_mean, self.p_inputs)
                self.log_sigma_sq_p = tf.nn.embedding_lookup(self.predicate_embedding_sigma, self.p_inputs)
                self.h_p = self.sample_embedding(self.mu_p, self.log_sigma_sq_p)

            if self.tensorboard:


                var1_1= tf.nn.embedding_lookup(self.entity_embedding_mean, self.var1)
                var1_2= tf.nn.embedding_lookup(self.entity_embedding_sigma, self.var1)

                with tf.name_scope('Entity1_Mean'):

                    self.variable_summaries(var1_1)

                with tf.name_scope('Entity1_Std'):

                    self.variable_summaries(var1_2)

                var2_1= tf.nn.embedding_lookup(self.entity_embedding_mean, self.var2)
                var2_2= tf.nn.embedding_lookup(self.entity_embedding_sigma, self.var2)

                with tf.name_scope('Entity2_Mean'):

                    self.variable_summaries(var2_1)

                with tf.name_scope('Entity2_Std'):

                    self.variable_summaries(var2_2)

                var3_1= tf.nn.embedding_lookup(self.entity_embedding_mean, self.var3)
                var3_2= tf.nn.embedding_lookup(self.entity_embedding_sigma, self.var3)

                with tf.name_scope('Entity3_Mean'):

                    self.variable_summaries(var3_1)

                with tf.name_scope('Entity3_Std'):

                    self.variable_summaries(var3_2)

                var4_1= tf.nn.embedding_lookup(self.entity_embedding_mean, self.var4)
                var4_2= tf.nn.embedding_lookup(self.entity_embedding_sigma, self.var4)

                with tf.name_scope('Entity4_Mean'):

                    self.variable_summaries(var4_1)

                with tf.name_scope('Entity4_Std'):

                    self.variable_summaries(var4_2)

                var5_1= tf.nn.embedding_lookup(self.predicate_embedding_mean, self.var5)
                var5_2= tf.nn.embedding_lookup(self.predicate_embedding_sigma, self.var5)

                with tf.name_scope('Predicate1_Mean'):

                    self.variable_summaries(var5_1)

                with tf.name_scope('Predicate1_Std'):

                    self.variable_summaries(var5_2)

                var6_1= tf.nn.embedding_lookup(self.predicate_embedding_mean, self.var6)
                var6_2= tf.nn.embedding_lookup(self.predicate_embedding_sigma, self.var6)

                with tf.name_scope('Predicate2_Mean'):

                    self.variable_summaries(var6_1)

                with tf.name_scope('Predicate2_Std'):

                    self.variable_summaries(var6_2)



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

    def train(self, test_triples, all_triples, batch_size, session=0, nb_epochs=1000,unit_cube=False,filename='/home/acowenri/workspace/Neural-Variational-Knowledge-Graphs/logs/'):
        """
                                Train Model
        """
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
        # batch_size = math.ceil(nb_samples / nb_batches)
        nb_batches= math.ceil(nb_samples / batch_size)
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

                    if self.alt_cost:  # if compression cost

                        loss_args = {
                            self.KL_discount: pi[counter],
                            self.s_inputs: Xs_batch,
                            self.p_inputs: Xp_batch,
                            self.o_inputs: Xo_batch,
                            self.y_inputs: np.array([1.0, 0.0, 0.0] * curr_batch_size)
                        }

                    else:

                        loss_args = {
                            self.KL_discount: (1.0/nb_batches),
                            self.s_inputs: Xs_batch,
                            self.p_inputs: Xp_batch,
                            self.o_inputs: Xo_batch,
                            self.y_inputs: np.array([1.0, 0.0, 0.0] * curr_batch_size)
                        }

                    if self.tensorboard:
                        merge = tf.summary.merge_all()

                    if self.alt_updates:

                        _, elbo_value1 = session.run([self.training_step1, self.e_objective1], feed_dict=loss_args)
                        _, elbo_value2 = session.run([self.training_step2, self.e_objective2], feed_dict=loss_args)
                        _, elbo_value3 = session.run([self.training_step3, self.e_objective3], feed_dict=loss_args)

                        if self.tensorboard:

                            summary,_, g_value, elbo_value = session.run([merge,self.training_step4, self.g_objective,self.elbo], feed_dict=loss_args)

                        else:

                            _, g_value, elbo_value = session.run([self.training_step4, self.g_objective,self.elbo], feed_dict=loss_args)


                    else:


                        if self.tensorboard:
                            summary,_, elbo_value = session.run([merge,self.training_step, self.elbo], feed_dict=loss_args)


                        else:

                             _, elbo_value = session.run([self.training_step, self.elbo],
                                                                 feed_dict=loss_args)

                    if counter % 2 == 0:
                        if self.tensorboard:
                            train_writer.add_summary(summary, counter)  # tensorboard

                    loss_values += [elbo_value / (Xp_batch.shape[0] / nb_versions)]
                    total_loss_value += elbo_value

                    counter += 1

                    if (self.projection==True) and (self.static_mean==False): #project means
                        for projection_step in projection_steps:
                            session.run([projection_step])


                if (round(np.mean(loss_values), 4) < minloss):
                    minloss = round(np.mean(loss_values), 4)
                    minepoch = epoch
                else:

                    if (round(np.mean(loss_values), 4) < minloss):
                        minloss = round(np.mean(loss_values), 4)
                        minepoch = epoch

                ##
                # Test
                ##

                if (epoch % 100)==0:

                    for eval_name in ['valid']:

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

                logger.warn('Epoch: {0}\tELBO: {1}'.format(epoch, self.stats(loss_values)))

            logger.warn("The minimum loss achieved is {0} \t at epoch {1}".format(minloss, minepoch))
            logger.warn("The maximum Filtered Hits@10 value: {0} \t at epoch {1}".format(maxhits, maxepoch))

# class MoGVKGE:
#     def __init__(self, embedding_size=5,batch_s=14145, lr=0.001, b1=0.9, b2=0.999, eps=1e-08, GPUMode=False, sigma1_e=6.0,sigma1_p=6.0,sigma2_e=1.0,sigma2_p=1.0,mix=6.0,
#                  alt_cost=True,train_mean=True):
#         super().__init__()
#
#         predicate_embedding_size = embedding_size
#         entity_embedding_size = embedding_size
#
#         triples = io.read_triples("data/wn18/wordnet-mlj12-train.txt")  # choose dataset
#         test_triples = io.read_triples("data/wn18/wordnet-mlj12-test.txt")
#
#         self.random_state = np.random.RandomState(0)
#         self.GPUMode = GPUMode
#         self.alt_cost = alt_cost
#         self.nb_examples = len(triples)
#
#         if not (self.GPUMode):
#             logger.warn('Parsing the facts in the Knowledge Base ..')
#
#         # logger.warn('Parsing the facts in the Knowledge Base ..')
#         self.facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
#
#         self.test_facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in test_triples]
#
#         self.test_parser = KnowledgeBaseParser(self.test_facts)
#         self.parser = KnowledgeBaseParser(self.facts)
#
#         ##### for test time ######
#         all_triples = triples + test_triples
#         entity_set = {s for (s, p, o) in all_triples} | {o for (s, p, o) in all_triples}
#         predicate_set = {p for (s, p, o) in all_triples}
#         self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entity_set))}
#         self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(predicate_set))}
#         self.nb_entities, self.nb_predicates = len(entity_set), len(predicate_set)
#         ############################
#
#         optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2, epsilon=eps)
#         self.build_model(self.nb_entities, entity_embedding_size, self.nb_predicates, predicate_embedding_size,
#                          optimizer,sigma1_e, sigma1_p, sigma2_e, sigma2_p, mix,train_mean)
#
#         self.train(nb_epochs=1000, test_triples=test_triples, all_triples=all_triples,batch_size=batch_s)
#
#     @staticmethod
#     def input_parameters(inputs, parameters_layer):
#         parameters = tf.nn.embedding_lookup(parameters_layer, inputs)
#         mu, log_sigma_square = tf.split(value=parameters, num_or_size_splits=2, axis=1)
#         return mu, log_sigma_square
#
#     @staticmethod
#     def sample_embedding(mu, log_sigma_square):
#         sigma = tf.sqrt(tf.exp(log_sigma_square))
#         embedding_size = mu.get_shape()[1].value
#         eps = tf.random_normal((1, embedding_size), 0, 1, dtype=tf.float32)
#         return mu + sigma * eps
#
#     def build_model(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size, optimizer,
#                     sigma1_e, sigma1_p, sigma2_e, sigma2_p, mix,train_mean):
#         self.s_inputs = tf.placeholder(tf.int32, shape=[None])
#         self.p_inputs = tf.placeholder(tf.int32, shape=[None])
#         self.o_inputs = tf.placeholder(tf.int32, shape=[None])
#         self.y_inputs = tf.placeholder(tf.bool, shape=[None])
#
#
#         self.build_encoder(nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size,
#                            sigma1_e, sigma1_p, sigma2_e, sigma2_p, mix,train_mean)
#         self.build_decoder()
#
#         self.KL_discount = tf.placeholder(tf.float32)  # starts at 0.5
#
#         # Kullback Leibler divergence spike
#         self.e_objective = 0.0
#         self.e_objective -= 0.5 * tf.reduce_sum(
#             1. + self.log_sigma_sq_s1 - tf.square(self.mu_s) - tf.exp(self.log_sigma_sq_s1))
#         self.e_objective -= 0.5 * tf.reduce_sum(
#             1. + self.log_sigma_sq_p1 - tf.square(self.mu_p) - tf.exp(self.log_sigma_sq_p1))
#         self.e_objective -= 0.5 * tf.reduce_sum(
#             1. + self.log_sigma_sq_o1 - tf.square(self.mu_o) - tf.exp(self.log_sigma_sq_o1))
#         ## KL Slab
#
#         self.e_objective -= 0.5 * tf.reduce_sum(
#             1. + self.log_sigma_sq_s2 - tf.square(self.mu_s) - tf.exp(self.log_sigma_sq_s2))
#         self.e_objective -= 0.5 * tf.reduce_sum(
#             1. + self.log_sigma_sq_p2 - tf.square(self.mu_p) - tf.exp(self.log_sigma_sq_p2))
#         self.e_objective -= 0.5 * tf.reduce_sum(
#             1. + self.log_sigma_sq_o2 - tf.square(self.mu_o) - tf.exp(self.log_sigma_sq_o2))
#
#         ##compression cost
#         self.e_objective = (self.e_objective * self.KL_discount)
#         # Log likelihood
#         self.g_objective = -tf.reduce_sum(
#             tf.log(tf.where(condition=self.y_inputs, x=self.p_x_i, y=1 - self.p_x_i) + 1e-4))
#
#         self.elbo = self.g_objective + self.e_objective
#
#         self.training_step = optimizer.minimize(self.elbo)
#
#     def build_encoder(self, nb_entities, entity_embedding_size, nb_predicates, predicate_embedding_size,
#                       sigma1_e, sigma1_p, sigma2_e, sigma2_p, mix,train_mean):
#         if not (self.GPUMode):
#             logger.warn('Building Inference Networks q(h_x | x) ..')
#
#         # logger.warn('Building Inference Networks q(h_x | x) ..')
#
#         with tf.variable_scope("encoder"):
#             self.entity_embedding_mean = tf.get_variable('entities_mean',
#                                                          shape=[nb_entities + 1, entity_embedding_size],
#                                                          initializer=tf.zeros_initializer(), dtype=tf.float32,trainable=train_mean)
#             self.entity_embedding_sigm1 = tf.get_variable('entities_sigma1',
#                                                          shape=[nb_entities + 1, entity_embedding_size],
#                                                          initializer=tf.ones_initializer(), dtype=tf.float32)
#
#             self.entity_embedding_sigma1 = tf.Variable(self.entity_embedding_sigm1.initialized_value() * sigma1_e,
#                                                       dtype=tf.float32)
#
#             self.entity_embedding_sigm2 = tf.get_variable('entities_sigma2',
#                                                          shape=[nb_entities + 1, entity_embedding_size],
#                                                          initializer=tf.ones_initializer(), dtype=tf.float32)
#
#             self.entity_embedding_sigma2 = tf.Variable(self.entity_embedding_sigm2.initialized_value() * sigma2_e,
#                                                       dtype=tf.float32)
#
#             self.mu_s = tf.nn.embedding_lookup(self.entity_embedding_mean, self.s_inputs)
#             self.log_sigma_sq_s1 = tf.nn.embedding_lookup(self.entity_embedding_sigma1, self.s_inputs)
#             self.log_sigma_sq_s2 = tf.nn.embedding_lookup(self.entity_embedding_sigma2, self.s_inputs)
#
#             self.h_s1 = VKGE.sample_embedding(self.mu_s, self.log_sigma_sq_s1)
#             self.h_s2 = VKGE.sample_embedding(self.mu_s, self.log_sigma_sq_s2)
#
#             self.h_s = (self.h_s1*mix +(1-mix)*self.h_s2) #MoG
#
#             self.mu_o = tf.nn.embedding_lookup(self.entity_embedding_mean, self.o_inputs)
#             self.log_sigma_sq_o1 = tf.nn.embedding_lookup(self.entity_embedding_sigma1, self.o_inputs)
#             self.log_sigma_sq_o2 = tf.nn.embedding_lookup(self.entity_embedding_sigma2, self.o_inputs)
#
#             self.h_o1 = VKGE.sample_embedding(self.mu_o, self.log_sigma_sq_o1)
#             self.h_o2 = VKGE.sample_embedding(self.mu_o, self.log_sigma_sq_o2)
#
#             self.h_o = (self.h_o1*mix +(1-mix)*self.h_o2) #MoG
#
#             self.predicate_embedding_mean = tf.get_variable('predicate_mean',
#                                                             shape=[nb_predicates + 1, predicate_embedding_size],
#                                                             initializer=tf.zeros_initializer(), dtype=tf.float32,trainable=train_mean)
#             self.predicate_embedding_sigm1 = tf.get_variable('predicate_sigma',
#                                                             shape=[nb_predicates + 1, predicate_embedding_size],
#                                                             initializer=tf.ones_initializer(), dtype=tf.float32)
#
#             self.predicate_embedding_sigma1 = tf.Variable(self.predicate_embedding_sigm1.initialized_value() * pred_sig,
#                                                          dtype=tf.float32)
#             self.predicate_embedding_sigm2 = tf.get_variable('predicate_sigma',
#                                                              shape=[nb_predicates + 1, predicate_embedding_size],
#                                                              initializer=tf.ones_initializer(), dtype=tf.float32)
#
#             self.predicate_embedding_sigma2 = tf.Variable(self.predicate_embedding_sigm1.initialized_value() * pred_sig,dtype=tf.float32)
#
#             self.mu_p = tf.nn.embedding_lookup(self.predicate_embedding_mean, self.p_inputs)
#             self.log_sigma_sq_p1 = tf.nn.embedding_lookup(self.predicate_embedding_sigma1, self.p_inputs)
#             self.log_sigma_sq_p2 = tf.nn.embedding_lookup(self.predicate_embedding_sigma2, self.p_inputs)
#
#             self.h_p1 = VKGE.sample_embedding(self.mu_p, self.log_sigma_sq_p1)
#             self.h_p2 = VKGE.sample_embedding(self.mu_p, self.log_sigma_sq_p2)
#             self.h_p = (self.h_p1*mix +(1-mix)*self.h_p2) #MoG
#
#
#
#     def build_decoder(self):
#         if not (self.GPUMode):
#             logger.warn('Building Inference Network p(y|h) ..')
#         # logger.warn('Building Inference Network p(y|h) ..')
#         with tf.variable_scope('decoder'):
#             model = models.BilinearDiagonalModel(subject_embeddings=self.h_s, predicate_embeddings=self.h_p,
#                                                  object_embeddings=self.h_o)
#             self.scores = model()
#             self.p_x_i = tf.sigmoid(self.scores)
#
#     def train(self, test_triples, all_triples, batch_size, session=0, nb_epochs=1000):
#         index_gen = index.GlorotIndexGenerator()
#         neg_idxs = np.array(sorted(set(self.parser.entity_to_index.values())))
#
#         subj_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs,
#                                                  corr_obj=False)
#         obj_corruptor = corrupt.SimpleCorruptor(index_generator=index_gen, candidate_indices=neg_idxs,
#                                                 corr_obj=True)
#
#         train_sequences = self.parser.facts_to_sequences(self.facts)
#
#         Xs = np.array([s_idx for (_, [s_idx, _]) in train_sequences])
#         Xp = np.array([p_idx for (p_idx, _) in train_sequences])
#         Xo = np.array([o_idx for (_, [_, o_idx]) in train_sequences])
#
#         assert Xs.shape == Xp.shape == Xo.shape
#
#         nb_samples = Xs.shape[0]
#         # batch_size = math.ceil(nb_samples / nb_batches)
#         nb_batches= math.ceil(nb_samples / batch_size)
#         # logger.warn("Samples: {}, no. batches: {} -> batch size: {}".format(nb_samples, nb_batches, batch_size))
#         if not (self.GPUMode):
#             logger.warn("Samples: {}, no. batches: {} -> batch size: {}".format(nb_samples, nb_batches, batch_size))
#
#         # projection_steps = [constraints.unit_cube(self.entity_parameters_layer) if unit_cube
#         #                     else constraints.unit_sphere(self.entity_parameters_layer, norm=1.0)]
#         minloss = 10000
#
#         minepoch = 0
#
#         ####### COMPRESSION COST PARAMETERS
#
#         M = int(np.ceil((self.nb_examples * nb_epochs / batch_size)) + 1)
#
#         pi_s = np.log(2.0) * M
#         pi_e = np.log(2.0)
#
#         pi = np.exp(np.linspace(pi_s, pi_e, M) - M * np.log(2.0))
#
#         counter = 0
#
#         #####################
#
#         init_op = tf.global_variables_initializer()
#         with tf.Session() as session:
#             session.run(init_op)
#             for epoch in range(1, nb_epochs + 1):
#                 order = self.random_state.permutation(nb_samples)
#                 Xs_shuf, Xp_shuf, Xo_shuf = Xs[order], Xp[order], Xo[order]
#
#                 Xs_sc, Xp_sc, Xo_sc = subj_corruptor(Xs_shuf, Xp_shuf, Xo_shuf)
#                 Xs_oc, Xp_oc, Xo_oc = obj_corruptor(Xs_shuf, Xp_shuf, Xo_shuf)
#
#                 batches = make_batches(nb_samples, batch_size)
#
#                 loss_values = []
#                 total_loss_value = 0
#
#                 nb_versions = 3
#
#                 for batch_start, batch_end in batches:
#                     curr_batch_size = batch_end - batch_start
#
#                     Xs_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xs_shuf.dtype)
#                     Xp_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xp_shuf.dtype)
#                     Xo_batch = np.zeros((curr_batch_size * nb_versions), dtype=Xo_shuf.dtype)
#
#                     # Positive Example
#                     Xs_batch[0::nb_versions] = Xs_shuf[batch_start:batch_end]
#                     Xp_batch[0::nb_versions] = Xp_shuf[batch_start:batch_end]
#                     Xo_batch[0::nb_versions] = Xo_shuf[batch_start:batch_end]
#
#                     # Negative examples (corrupting subject)
#                     Xs_batch[1::nb_versions] = Xs_sc[batch_start:batch_end]
#                     Xp_batch[1::nb_versions] = Xp_sc[batch_start:batch_end]
#                     Xo_batch[1::nb_versions] = Xo_sc[batch_start:batch_end]
#
#                     # Negative examples (corrupting object)
#                     Xs_batch[2::nb_versions] = Xs_oc[batch_start:batch_end]
#                     Xp_batch[2::nb_versions] = Xp_oc[batch_start:batch_end]
#                     Xo_batch[2::nb_versions] = Xo_oc[batch_start:batch_end]
#
#                     y = np.zeros_like(Xp_batch)
#                     y[0::nb_versions] = 1
#
#                     if self.alt_cost:  # if compression cost
#
#                         loss_args = {
#                             self.KL_discount: pi[counter],
#                             self.s_inputs: Xs_batch,
#                             self.p_inputs: Xp_batch,
#                             self.o_inputs: Xo_batch,
#                             self.y_inputs: y
#                         }
#
#                     else:
#
#                         loss_args = {
#                             self.KL_discount: 1.0,
#                             self.s_inputs: Xs_batch,
#                             self.p_inputs: Xp_batch,
#                             self.o_inputs: Xo_batch,
#                             self.y_inputs: y
#                         }
#
#                     _, elbo_value = session.run([self.training_step, self.elbo], feed_dict=loss_args)
#
#                     loss_values += [elbo_value / (Xp_batch.shape[0] / nb_versions)]
#                     total_loss_value += elbo_value
#
#                     counter += 1
#
#                     # for projection_step in projection_steps:
#                     #     session.run([projection_step])
#
#                 def stats(values):
#                     return '{0:.4f} ± {1:.4f}'.format(round(np.mean(values), 4), round(np.std(values), 4))
#
#                 # logger.warn('Epoch: {0}\tELBO: {1}'.format(epoch, stats(loss_values)))
#                 if self.GPUMode:
#                     if (round(np.mean(loss_values), 4) < minloss):
#                         minloss = round(np.mean(loss_values), 4)
#                         minepoch = epoch
#                 else:
#
#                     if (round(np.mean(loss_values), 4) < minloss):
#                         minloss = round(np.mean(loss_values), 4)
#                         minepoch = epoch
#
#                     logger.warn('Epoch: {0}\tVLB: {1}'.format(epoch, stats(loss_values)))
#
#                 if (epoch % 200)==0:
#
#                     for eval_type in ['valid']:
#
#                         eval_triples = test_triples
#                         ranks_subj, ranks_obj = [], []
#                         filtered_ranks_subj, filtered_ranks_obj = [], []
#
#                         for _i, (s, p, o) in enumerate(eval_triples):
#                             s_idx, p_idx, o_idx = self.entity_to_idx[s], self.predicate_to_idx[p], self.entity_to_idx[o]
#
#                             Xs_v = np.full(shape=(self.nb_entities,), fill_value=s_idx, dtype=np.int32)
#                             Xp_v = np.full(shape=(self.nb_entities,), fill_value=p_idx, dtype=np.int32)
#                             Xo_v = np.full(shape=(self.nb_entities,), fill_value=o_idx, dtype=np.int32)
#
#                             feed_dict_corrupt_subj = {self.s_inputs: np.arange(self.nb_entities), self.p_inputs: Xp_v,
#                                                       self.o_inputs: Xo_v}
#                             feed_dict_corrupt_obj = {self.s_inputs: Xs_v, self.p_inputs: Xp_v,
#                                                      self.o_inputs: np.arange(self.nb_entities)}
#
#                             # scores of (1, p, o), (2, p, o), .., (N, p, o)
#                             scores_subj = session.run(self.scores, feed_dict=feed_dict_corrupt_subj)
#
#                             # scores of (s, p, 1), (s, p, 2), .., (s, p, N)
#                             scores_obj = session.run(self.scores, feed_dict=feed_dict_corrupt_obj)
#
#                             ranks_subj += [1 + np.sum(scores_subj > scores_subj[s_idx])]
#                             ranks_obj += [1 + np.sum(scores_obj > scores_obj[o_idx])]
#
#                             filtered_scores_subj = scores_subj.copy()
#                             filtered_scores_obj = scores_obj.copy()
#
#                             rm_idx_s = [self.entity_to_idx[fs] for (fs, fp, fo) in all_triples if
#                                         fs != s and fp == p and fo == o]
#                             rm_idx_o = [self.entity_to_idx[fo] for (fs, fp, fo) in all_triples if
#                                         fs == s and fp == p and fo != o]
#
#                             filtered_scores_subj[rm_idx_s] = - np.inf
#                             filtered_scores_obj[rm_idx_o] = - np.inf
#
#                             filtered_ranks_subj += [1 + np.sum(filtered_scores_subj > filtered_scores_subj[s_idx])]
#                             filtered_ranks_obj += [1 + np.sum(filtered_scores_obj > filtered_scores_obj[o_idx])]
#
#                         ranks = ranks_subj + ranks_obj
#                         filtered_ranks = filtered_ranks_subj + filtered_ranks_obj
#
#                         for setting_name, setting_ranks in [('Filtered', filtered_ranks)]:
#                             mean_rank = np.mean(setting_ranks)
#                             k = 10
#                             hits_at_k = np.mean(np.asarray(setting_ranks) <= k) * 100
#                     t1, t2 = mean_rank, hits_at_k
#                     logger.warn('Hits@10 value: {0} %'.format(t2))
#
#             logger.warn("The minimum loss achieved is {0} \t at epoch {1}".format(minloss, minepoch))
