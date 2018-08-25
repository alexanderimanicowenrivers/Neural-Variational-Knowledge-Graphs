# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
def renorm_update_var(log_var_matrix, norm=1.0, axis=0):
    #limits each variance vector having a spherical norm of greater than one -- unit variance.
    #first transform to origingal variance representation
    var_matrix=tf.exp(log_var_matrix)
    #norm sphere
    row_norms = tf.reduce_sum(var_matrix, axis=axis)
    scaled = var_matrix * tf.expand_dims(norm / row_norms, axis=axis)
    #transform back
    scaled=tf.log(scaled)
    return tf.assign(log_var_matrix, scaled)

def renorm_update_clip(log_var_matrix, norm=1.0, axis=0):
    #limits each variance vector having a spherical variance of greater than one, spherical covariance .
    #first transform to origingal variance representation
    var_matrix=tf.exp(log_var_matrix)
    #norm sphere
    scaled=tf.clip_by_norm(var_matrix,1.0,axes=1)
    scaled=tf.log(scaled)
    return tf.assign(log_var_matrix, scaled)

def renorm_unitvariance(log_var_matrix, norm=1.0, axis=0):
    #limits each variance vector having a spherical norm of greater than one -- unit variance.
    #first transform to origingal variance representation
    var_matrix=tf.sqrt(tf.exp(log_var_matrix))
    #norm sphere
    scaled=tf.clip_by_norm(var_matrix,1.0,axes=1)
    scaled=tf.log(scaled**2)
    return tf.assign(log_var_matrix, scaled)


def renorm_update(var_matrix, norm=1.0, axis=1):
    row_norms = tf.sqrt(tf.reduce_sum(tf.square(var_matrix), axis=axis))
    scaled = var_matrix * tf.expand_dims(norm / row_norms, axis=axis)
    return tf.assign(var_matrix, scaled)


def pseudoboolean_linear_update(var_matrix):
    pseudoboolean_linear = tf.minimum(1., tf.maximum(var_matrix, 0.))
    return tf.assign(var_matrix, pseudoboolean_linear)

def renorm_cubevariance(log_var_matrix, val=1.0):
    #limits each variance vector having a spherical norm of greater than one -- unit variance.
    #first transform to origingal variance representation
    var_matrix=tf.sqrt(tf.exp(log_var_matrix))
    #norm sphere
    scaled=tf.clip_by_value(var_matrix,val)
    scaled=tf.log(scaled**2)
    return tf.assign(log_var_matrix, scaled)

def pseudoboolean_sigmoid_update(var_matrix):
    pseudoboolean_sigmoid = tf.nn.sigmoid(var_matrix)
    return tf.assign(var_matrix, pseudoboolean_sigmoid)

unit_sphere_logvar=renorm_unitvariance
unit_sphere = renorm = renorm_update
unit_cube = pseudoboolean_linear = pseudoboolean_linear_update
unit_cube_logvar = renorm_cubevariance
pseudoboolean_sigmoid = pseudoboolean_sigmoid_update


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown constraint: {}'.format(function_name))
    return getattr(this_module, function_name)
