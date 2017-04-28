#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from inferbeddings.knowledgebase import Fact, KnowledgeBaseParser
from vkge import VKGE

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))

def main(argv):
    triples = [
        ('a', 'p', 'b'),
        ('b', 'p', 'c'),
        ('a', 'p', 'c')
    ]

    nb_triples = len(triples)

    facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
    parser = KnowledgeBaseParser(facts)

    nb_entities = len(parser.entity_vocabulary)
    nb_predicates = len(parser.predicate_vocabulary)

    sequences = parser.facts_to_sequences(facts)

    Xs = np.array([s[1][0] for s in sequences], dtype=np.int32)
    Xp = np.array([s[0] for s in sequences], dtype=np.int32)
    Xo = np.array([s[1][1] for s in sequences], dtype=np.int32)
    y = np.array([1] * nb_triples, dtype=np.int32)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    vkge = VKGE(nb_entities, 10, nb_predicates, 10, optimizer)

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        vkge.train(session, Xs, Xp, Xo, y, nb_epochs=1000)

    print(Xs, Xp, Xo)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

