#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf
import vkge.models as models

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))

nb_entities = 100
nb_predicates = 5

entity_embedding_size = 10
predicate_embedding_size = 10


def sample_embedding(inputs, parameters_layer):
    """
    :param inputs: [batch_size] tf.int32 tensor
    :param parameters_layer: [nb_entities, embedding_size * 2] tf.float32 tensor
    :return: [batch_size, embedding_size] tf.float32 tensor
    """
    # [batch_size, embedding_size * 2] tf.float32 tensor
    parameters = tf.nn.embedding_lookup(parameters_layer, inputs)
    # [batch_size, embedding_size], [batch_size, embedding_size] tf.float32 tensors
    mu, log_sigma_square = tf.split(value=parameters, num_or_size_splits=2, axis=1)
    embedding_size = mu.get_shape()[1].value
    eps = tf.random_normal((1, embedding_size), 0, 1, dtype=tf.float32)
    sigma = tf.sqrt(tf.exp(log_sigma_square))
    return mu + sigma * eps


def main(argv):
    subject_input = tf.placeholder(tf.int32, shape=[None])
    predicate_input = tf.placeholder(tf.int32, shape=[None])
    object_input = tf.placeholder(tf.int32, shape=[None])

    logger.info('Building Inference Network q(h|X) ..')
    entity_parameters_layer = tf.get_variable('entities', shape=[nb_entities, entity_embedding_size * 2],
                                             initializer=tf.contrib.layers.xavier_initializer())
    predicate_parameters_layer = tf.get_variable('predicates', shape=[nb_predicates, predicate_embedding_size * 2],
                                                initializer=tf.contrib.layers.xavier_initializer())

    sampled_subject_embeddings = sample_embedding(subject_input, entity_parameters_layer)
    sampled_predicate_embeddings = sample_embedding(predicate_input, predicate_parameters_layer)
    sampled_object_embeddings = sample_embedding(object_input, entity_parameters_layer)

    logger.info('Building Inference Network p(X|h) ..')

    model = models.BilinearDiagonalModel(subject_embeddings=sampled_subject_embeddings,
                                         predicate_embeddings=sampled_predicate_embeddings,
                                         object_embeddings=sampled_object_embeddings)
    p = tf.sigmoid(model())

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        p_values = session.run([p], feed_dict={
            subject_input: [0, 0, 0],
            predicate_input: [0, 0, 0],
            object_input: [0, 0, 0],
        })

        print(p_values)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

