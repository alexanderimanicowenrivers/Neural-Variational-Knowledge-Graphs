# -*- coding: utf-8 -*-

import abc
import tensorflow as tf
from vkge.models.similarities import negative_l1_distance
import sys


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, subject_embeddings=None, predicate_embeddings=None, object_embeddings=None, *args, **kwargs):
        """
        Abstract class inherited by all models.

        :param subject_embeddings: (batch_size, entity_embedding_size) Tensor.
        :param predicate_embeddings: (batch_size, predicate_embedding_size) Tensor.
        :param object_embeddings: (batch_size, entity_embedding_size) Tensor.
        """
        self.subject_embeddings = subject_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.object_embeddings = object_embeddings

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    @property
    def parameters(self):
        return []


class TranslatingModel(BaseModel):
    def __init__(self, similarity_function=negative_l1_distance, *args, **kwargs):
        """
        Implementation of the Translating Embeddings model [1].
        [1] Bordes, A. et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013

        :param similarity_function: Similarity function.
        """
        super().__init__(*args, **kwargs)
        self.similarity_function = similarity_function

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        translated_subject_embedding = self.subject_embeddings + self.predicate_embeddings
        return self.similarity_function(translated_subject_embedding, self.object_embeddings)


class BilinearDiagonalModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of the Bilinear-Diagonal model [1]
        [1] Yang, B. et al. - Embedding Entities and Relations for Learning and Inference in Knowledge Bases - ICLR 2015

        :param similarity_function: Similarity function.
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        scaled_subject_embedding = self.subject_embeddings * self.predicate_embeddings
        return tf.reduce_sum(scaled_subject_embedding * self.object_embeddings, axis=1)


class BilinearModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of the Bilinear model [1]
        [1] Nickel, M. et al. - A Three-Way Model for Collective Learning on Multi-Relational Data - ICML 2011

        :param similarity_function: Similarity function.
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        es, emb_size = tf.expand_dims(self.subject_embeddings, 1), tf.shape(self.subject_embeddings)[1]
        W = tf.reshape(self.predicate_embeddings, (-1, emb_size, emb_size))
        sW = tf.matmul(es, W)[:, 0, :]
        return tf.reduce_sum(sW * self.object_embeddings, axis=1)


class ComplexModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of the ComplEx model [1]
        [1] Trouillon, T. et al. - Complex Embeddings for Simple Link Prediction - ICML 2016

        :param embedding size: Embedding size.
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        es_re, es_im = tf.split(value=self.subject_embeddings, num_or_size_splits=2, axis=1)
        eo_re, eo_im = tf.split(value=self.object_embeddings, num_or_size_splits=2, axis=1)
        ew_re, ew_im = tf.split(value=self.predicate_embeddings, num_or_size_splits=2, axis=1)

        def dot3(arg1, rel, arg2):
            return tf.reduce_sum(arg1 * rel * arg2, axis=1)

        score = dot3(es_re, ew_re, eo_re) + dot3(es_re, ew_im, eo_im) + dot3(es_im, ew_re, eo_im) - dot3(es_im, ew_im, eo_re)
        return score

def Quaternator(predicate_embeddings,subject_embeddings,object_embeddings):
    """
    Implementation of the Quaternator
    :param embedding size: Embedding size.
    :return: (batch_size) Tensor containing the scores associated by the models to the walks.
     """
    emb_rel, emb_arg1, emb_arg2=predicate_embeddings,subject_embeddings,object_embeddings
    rel_r, rel_x, rel_y, rel_z = tf.split(emb_rel, 4, axis=1)
    arg1_r, arg1_x, arg1_y, arg1_z = tf.split(emb_arg1, 4, axis=1)
    arg2_r, arg2_x, arg2_y, arg2_z = tf.split(emb_arg2, 4, axis=1)

    score1 = tf.einsum("ij,ij->i", arg1_r * arg2_r, rel_x)
    score2 = tf.einsum("ij,ij->i", arg1_r * arg2_x, rel_r)
    score3 = tf.einsum("ij,ij->i", arg1_r * arg2_y, rel_z)
    score4 = tf.einsum("ij,ij->i", arg1_r * arg2_z, rel_y)
    score5 = tf.einsum("ij,ij->i", arg1_x * arg2_r, rel_r)
    score6 = tf.einsum("ij,ij->i", arg1_x * arg2_x, rel_x)
    score7 = tf.einsum("ij,ij->i", arg1_x * arg2_y, rel_y)
    score8 = tf.einsum("ij,ij->i", arg1_x * arg2_z, rel_z)
    score9 = tf.einsum("ij,ij->i", arg1_y * arg2_r, rel_z)
    score10 = tf.einsum("ij,ij->i", arg1_y * arg2_x, rel_y)
    score11 = tf.einsum("ij,ij->i", arg1_y * arg2_y, rel_x)
    score12 = tf.einsum("ij,ij->i", arg1_y * arg2_z, rel_r)
    score13 = tf.einsum("ij,ij->i", arg1_z * arg2_r, rel_y)
    score14 = tf.einsum("ij,ij->i", arg1_z * arg2_x, rel_z)
    score15 = tf.einsum("ij,ij->i", arg1_z * arg2_y, rel_r)
    score16 = tf.einsum("ij,ij->i", arg1_z * arg2_z, rel_x)

    return (score1 - score2 + score3 - score4 +
            score5 + score6 - score7 - score8 -
            score9 + score10 + score11 - score12 +
            score13 + score14 + score15 + score16)

# class Quaternator(BaseModel):
#     def __init__(self, *args, **kwargs):
#         """
#         Implementation of the Quaternator
#         :param embedding size: Embedding size.
#         """
#         super().__init__(*args, **kwargs)
#
#     def __call__(self):
#         """
#         :return: (batch_size) Tensor containing the scores associated by the models to the walks.
#          """
#         emb_rel, emb_arg1, emb_arg2=self.predicate_embeddings,self.subject_embeddings,self.object_embeddings
#         rel_r, rel_x, rel_y, rel_z = tf.split(emb_rel, 4, axis=1)
#         arg1_r, arg1_x, arg1_y, arg1_z = tf.split(emb_arg1, 4, axis=1)
#         arg2_r, arg2_x, arg2_y, arg2_z = tf.split(emb_arg2, 4, axis=1)
#
#         score1 = tf.einsum("ij,ij->i", arg1_r * arg2_r, rel_x)
#         score2 = tf.einsum("ij,ij->i", arg1_r * arg2_x, rel_r)
#         score3 = tf.einsum("ij,ij->i", arg1_r * arg2_y, rel_z)
#         score4 = tf.einsum("ij,ij->i", arg1_r * arg2_z, rel_y)
#         score5 = tf.einsum("ij,ij->i", arg1_x * arg2_r, rel_r)
#         score6 = tf.einsum("ij,ij->i", arg1_x * arg2_x, rel_x)
#         score7 = tf.einsum("ij,ij->i", arg1_x * arg2_y, rel_y)
#         score8 = tf.einsum("ij,ij->i", arg1_x * arg2_z, rel_z)
#         score9 = tf.einsum("ij,ij->i", arg1_y * arg2_r, rel_z)
#         score10 = tf.einsum("ij,ij->i", arg1_y * arg2_x, rel_y)
#         score11 = tf.einsum("ij,ij->i", arg1_y * arg2_y, rel_x)
#         score12 = tf.einsum("ij,ij->i", arg1_y * arg2_z, rel_r)
#         score13 = tf.einsum("ij,ij->i", arg1_z * arg2_r, rel_y)
#         score14 = tf.einsum("ij,ij->i", arg1_z * arg2_x, rel_z)
#         score15 = tf.einsum("ij,ij->i", arg1_z * arg2_y, rel_r)
#         score16 = tf.einsum("ij,ij->i", arg1_z * arg2_z, rel_x)
#
#         return (score1 - score2 + score3 - score4 +
#                 score5 + score6 - score7 - score8 -
#                 score9 + score10 + score11 - score12 +
#                 score13 + score14 + score15 + score16)

# Aliases
TransE = TranslatingEmbeddings = TranslatingModel
DistMult = BilinearDiagonal = BilinearDiagonalModel
RESCAL = Bilinear = BilinearModel
ComplEx = ComplexE = ComplexModel
Qt=Quaternator

def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown model: {}'.format(function_name))
    return getattr(this_module, function_name)
