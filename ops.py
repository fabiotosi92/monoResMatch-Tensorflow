from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def correlation_map(x, y, max_disp, stride=1, name='corr'):
    with tf.variable_scope(name):
        corr_tensors = []
        y_shape = tf.shape(y)
        y_feature = tf.pad(y,[[0,0],[0,0],[max_disp,max_disp],[0,0]])
        for i in range(-max_disp, max_disp+1,stride):
            shifted = tf.slice(y_feature, [0, 0, i + max_disp, 0], [-1, y_shape[1], y_shape[2], -1])
            corr_tensors.append(tf.reduce_mean(shifted*x, axis=-1, keepdims=True))

        result = tf.concat(corr_tensors,axis=-1)
        return result


def conv2d(x, kernel_shape, strides=1, relu=True, padding='SAME'):
    W = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))
    with tf.name_scope("conv"):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)

    return x


def conv2d_transpose(x, kernel_shape, strides=1, relu=True):
    W = tf.get_variable("weights_transpose", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable("biases_transpose", kernel_shape[2], initializer=tf.constant_initializer(0.0))
    output_shape = [x.get_shape()[0].value,
                    x.get_shape()[1].value*strides, x.get_shape()[2].value*strides, kernel_shape[2]]
    with tf.name_scope("deconv"):
        x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1],
                                   padding='SAME')
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)
    return x


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


