# -*- coding: utf-8 -*-

import abc
import numpy as np


class ACorruptor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, Xs, Xp, Xo):
        raise NotImplementedError


class SimpleCorruptor(ACorruptor):
    def __init__(self, index_generator=None, candidate_indices=None, corr_obj=False):
        self.index_generator, self.candidate_indices = index_generator, candidate_indices
        self.corr_obj = corr_obj

    def __call__(self, Xs, Xp, Xo):
        neg_Xe = np.copy(Xs)
        neg_Xe[:] = self.index_generator(Xs.shape[0], self.candidate_indices)
        return (Xs if self.corr_obj else neg_Xe), Xp, (neg_Xe if self.corr_obj else Xo)
