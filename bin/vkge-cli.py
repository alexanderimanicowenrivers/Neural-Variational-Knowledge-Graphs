#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from vkge.io import read_triples
from vkge.knowledgebase import Fact, KnowledgeBaseParser
from vkge import VKGE

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    train_path = 'data/wn18/wordnet-mlj12-train.txt'
    triples = read_triples(train_path)

    nb_batch_size = 8192

    facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
    parser = KnowledgeBaseParser(facts)

    nb_entities = len(parser.entity_vocabulary)
    nb_predicates = len(parser.predicate_vocabulary)

    sequences = parser.facts_to_sequences(facts)
    triples_idx = {(s, p, o) for (p, [s, o]) in sequences}

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    vkge = VKGE(nb_entities, 10, nb_predicates, 10, optimizer)

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)

        for i in range(1024):
            Xs = 1 + np.random.choice(nb_entities, size=nb_batch_size)
            Xp = 1 + np.random.choice(nb_predicates, size=nb_batch_size)
            Xo = 1 + np.random.choice(nb_entities, size=nb_batch_size)

            y = np.array([(s, p, o) in triples_idx for s, p, o in zip(Xs, Xp, Xo)], dtype=np.int8)

            elbo_value = vkge.train(session, Xs, Xp, Xo, y)
            logger.info('[{}] ELBO: {}'.format(i, elbo_value))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

