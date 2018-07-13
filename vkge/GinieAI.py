# -*- coding: utf-8 -*-

import tensorflow as tf
from random import choice, shuffle
import numpy as np
from numpy import array
import os
from vkge.knowledgebase import Fact, KnowledgeBaseParser

import logging

logger = logging.getLogger(__name__)


class GinieAI:
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
        @param GPUMode: Used for reduced logger.warn statements during architecture search.
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

    def __init__(self,no_clusters=10):
        super().__init__()

        ## AUTOENCODER


        n_nodes_inpl = 18800  # encoder
        n_nodes_hl1 = 32  # encoder
        n_nodes_hl2 = 32  # decoder
        n_nodes_outl = 18800  # decoder

        self.hidden_1_layer_vals = {
            'weights': tf.Variable(tf.random_normal([n_nodes_inpl, n_nodes_hl1])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
        # second hidden layer has 32*32 weights and 32 biases
        hidden_2_layer_vals = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
        # second hidden layer has 32*784 weights and 784 biases
        output_layer_vals = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_outl])),
            'biases': tf.Variable(tf.random_normal([n_nodes_outl]))}

        # image with shape 784 goes in
        self.input_layer = tf.placeholder('float', [None, 18800])
        # multiply output of self.input_layer wth a weight matrix and add biases
        self.layer_1 = tf.nn.sigmoid(
            tf.add(tf.matmul(self.input_layer, self.hidden_1_layer_vals['weights']),
                   self.hidden_1_layer_vals['biases']))
        # multiply output of self.layer_1 wth a weight matrix and add biases
        layer_2 = tf.nn.sigmoid(
            tf.add(tf.matmul(self.layer_1, hidden_2_layer_vals['weights']),
                   hidden_2_layer_vals['biases']))
        # multiply output of layer_2 wth a weight matrix and add biases
        self.output_layer= tf.matmul(layer_2, output_layer_vals['weights']) +output_layer_vals['biases']

        # self.output_true shall have the original image for error calculations
        self.output_true = tf.placeholder('float', [None, n_nodes_inpl])
        # define our cost function
        self.meansq = tf.reduce_mean(tf.square(self.output_layer- self.output_true))
        # define our optimizer
        learn_rate = 0.01  # how fast the model should learn

        self.optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.meansq)

        # KMEANS

        logger.warn('Begin training')

        # self.train()


        logger.warn('Begin clustering of hidden representation')


        matrix=np.load('/home/acowenri/clause_compact.npy')

        logger.warn("Data loaded has size {}".format(matrix.shape))


        for no_clusts in [no_clusters]:

            centroids, assignments, mean_dist=self.TFKMeansCluster(matrix,no_clusts)
            logger.warn('number of clusters {}, mean distance {}'.format(no_clusts,mean_dist))

            np.save('/home/acowenri/centroids'+str(no_clusts),centroids)
            np.save('/home/acowenri/assignments'+str(no_clusts),assignments)



    def TFKMeansCluster(self,vectors, noofclusters):
        """
        K-Means Clustering using TensorFlow.
        'vectors' should be a n*k 2-D NumPy array, where n is the number
        of vectors of dimensionality k.
        'noofclusters' should be an integer.
        """

        noofclusters = int(noofclusters)
        assert noofclusters < len(vectors)

        # Find out the dimensionality
        dim = len(vectors[0])

        # Will help select random centroids from among the available vectors
        vector_indices = list(range(len(vectors)))
        shuffle(vector_indices)

        # GRAPH OF COMPUTATION
        # We initialize a new graph and set it as the default during each run
        # of this algorithm. This ensures that as this function is called
        # multiple times, the default graph doesn't keep getting crowded with
        # unused ops and Variables from previous function calls.




        graph = tf.Graph()

        with graph.as_default():

            # SESSION OF COMPUTATION

            sess = tf.Session()



            ##CONSTRUCTING THE ELEMENTS OF COMPUTATION

            ##First lets ensure we have a Variable vector for each centroid,
            ##initialized to one of the vectors from the available data points
            centroids = [tf.Variable((vectors[vector_indices[i]]),dtype=tf.float32)
                         for i in range(noofclusters)]
            ##These nodes will assign the centroid Variables the appropriate
            ##values
            centroid_value = tf.placeholder('float32',[dim])
            cent_assigns = []
            for centroid in centroids:
                cent_assigns.append(tf.assign(centroid, centroid_value))

            ##Variables for cluster assignments of individual vectors(initialized
            ##to 0 at first)
            assignments = [tf.Variable(0) for i in range(len(vectors))]
            ##These nodes will assign an assignment Variable the appropriate
            ##value
            assignment_value = tf.placeholder("int32")
            cluster_assigns = []
            for assignment in assignments:
                cluster_assigns.append(tf.assign(assignment,
                                                 assignment_value))

            ##Now lets construct the node that will compute the mean
            # The placeholder for the input
            mean_input = tf.placeholder("float32", [None, dim])
            # The Node/op takes the input and computes a mean along the 0th
            # dimension, i.e. the list of input vectors
            mean_op = tf.reduce_mean(mean_input, 0)

            ##Node for computing Euclidean distances
            # Placeholders for input
            v1 = tf.placeholder("float32", [dim])
            v2 = tf.placeholder("float32", [dim])
            euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(
                v1, v2), 2)))

            ##This node will figure out which cluster to assign a vector to,
            ##based on Euclidean distances of the vector from the centroids.
            # Placeholder for input
            centroid_distances = tf.placeholder("float32", [noofclusters])
            cluster_assignment = tf.argmin(centroid_distances, 0)

            ##INITIALIZING STATE VARIABLES

            ##This will help initialization of all Variables defined with respect
            ##to the graph. The Variable-initializer should be defined after
            ##all the Variables have been constructed, so that each of them
            ##will be included in the initialization.
            init_op = tf.global_variables_initializer()

            # Initialize all variables
            sess.run(init_op)

            ##CLUSTERING ITERATIONS
            logger.warn(len(vectors), "number of vectors to cluster")

            # Now perform the Expectation-Maximization steps of K-Means clustering
            # iterations. To keep things simple, we will only do a set number of
            # iterations, instead of using a Stopping Criterion.
            # noofiterations = 100
            noofiterations = 1

            for iteration_n in range(noofiterations):

                ##EXPECTATION STEP
                ##Based on the centroid locations till last iteration, compute
                ##the _expected_ centroid assignments.
                # Iterate over each vector
                for vector_n in range(len(vectors)):

                    vect = vectors[vector_n]
                    # Compute Euclidean distance between this vector and each
                    # centroid. Remember that this list cannot be named
                    # 'centroid_distances', since that is the input to the
                    # cluster assignment node.
                    distances = [sess.run(euclid_dist, feed_dict={
                        v1: vect, v2: sess.run(centroid)})
                                 for centroid in centroids]
                    # Now use the cluster assignment node, with the distances
                    # as the input
                    assignment = sess.run(cluster_assignment, feed_dict={
                        centroid_distances: distances})
                    # Now assign the value to the appropriate state variable
                    sess.run(cluster_assigns[vector_n], feed_dict={
                        assignment_value: assignment})


                mean_dist = np.mean(distances)

                logger.warn('Epoch', iteration_n, '/', 'loss:', mean_dist)

                ##MAXIMIZATION STEP
                # Based on the expected state computed from the Expectation Step,
                # compute the locations of the centroids so as to maximize the
                # overall objective of minimizing within-cluster Sum-of-Squares
                for cluster_n in range(noofclusters):
                    # Collect all the vectors assigned to this cluster
                    assigned_vects = [vectors[i] for i in range(len(vectors))
                                      if sess.run(assignments[i]) == cluster_n]
                    # Compute new centroid location
                    new_location = sess.run(mean_op, feed_dict={
                        mean_input: array(assigned_vects)})
                    # Assign value to appropriate variable
                    sess.run(cent_assigns[cluster_n], feed_dict={
                        centroid_value: new_location})






            # Return centroids and assignments
            centroids = sess.run(centroids)
            assignments = sess.run(assignments)
            return centroids, assignments,mean_dist

    def train(self):
        """

                                Train Model

        """

        all_clauses = np.load('/home/acowenri/clauses2vec.npy')
        logger.warn("Data loaded has size {}".format(all_clauses.shape))
        # initialising stuff and starting the session
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            # defining batch size, number of epochs and learning rate
            batch_size = 1129  # how many images to use together for training
            hm_epochs = 100  # how many times to go through the entire dataset
            tot_images = all_clauses.shape[0]  # total number of images
            # running the model for a 1000 epochs taking 100 images in batches
            # total improvement is logger.warned out after each epoch
            for epoch in range(hm_epochs):
                epoch_loss = 0  # initializing error as 0
                for i in range(int(tot_images / batch_size)):
                    epoch_x = all_clauses[i * batch_size: (i + 1) * batch_size,:]
                    # logger.warn(epoch_x.shape)
                    _, c = sess.run([self.optimizer , self.meansq],
                                    feed_dict={self.input_layer: epoch_x,
                                               self.output_true: epoch_x})

                    epoch_loss += c
                logger.warn('Epoch', epoch, '/', hm_epochs, 'loss:', epoch_loss)

            clause_compact = sess.run(self.layer_1,
                            feed_dict={self.input_layer: all_clauses,
                                       self.output_true: all_clauses})

            np.save('/home/acowenri/clause_compact',clause_compact)
