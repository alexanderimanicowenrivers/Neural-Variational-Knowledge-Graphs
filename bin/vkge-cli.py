#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf

from vkge.io import read_triples
from vkge import VKGE

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    train_path = 'data/wn18/wordnet-mlj12-train.txt'
    triples = read_triples(train_path)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    vkge = VKGE(triples, 10, 10, optimizer)

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)

        vkge.train(session, nb_epochs=100)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

