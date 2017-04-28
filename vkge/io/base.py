# -*- coding: utf-8 -*-

import gzip
import bz2
import pickle

import logging

logger = logging.getLogger(__name__)


def iopen(file, *args, **kwargs):
    _open = open
    if file.endswith('.gz'):
        _open = gzip.open
    elif file.endswith('.bz2'):
        _open = bz2.open
    return _open(file, *args, **kwargs)


def read_triples(path):
    triples = []
    with iopen(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split('\t')
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples
