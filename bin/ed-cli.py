#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf

from edward.models import Normal
from edward.models import Bernoulli

from vkge.knowledgebase import Fact, KnowledgeBaseParser
from vkge.io import read_triples

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def dot3(x, y, z):


def main(argv):
    train_path = 'data/wn18/wordnet-mlj12-train.txt'
    triples = read_triples(train_path)

    facts = [Fact(predicate_name=p, argument_names=[s, o]) for s, p, o in triples]
    parser = KnowledgeBaseParser(facts)

    nb_entities, nb_predicates = len(parser.entity_vocabulary), len(parser.predicate_vocabulary)
    embedding_size = 10

    E = Normal(loc=tf.zeros([nb_entities, embedding_size]), scale=tf.ones([nb_entities, embedding_size]))
    R = Normal(loc=tf.zeros([nb_predicates, embedding_size]), scale=tf.ones([nb_predicates, embedding_size]))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

